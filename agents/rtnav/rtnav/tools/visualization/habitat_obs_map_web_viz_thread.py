"""Web MJPEG viz for a standalone HabitatObstacleMap (VLFM-port).

Differs from :class:`WebVLFMMapVizThread` in one way only: it takes a
``HabitatObstacleMap`` (and a lock + an agent-xy getter) directly, instead of
reaching through an Agent + frontier-detector graph. Use this when you want to
render a HabitatObstacleMap that's being maintained outside the rtnav Agent
(e.g. a standalone teleop / map-only test script).

Color scheme is verbatim from ``HabitatObstacleMap.visualize`` / VLFM:
  white         unknown
  light green   explored
  gray          inflated-unnavigable
  black         obstacle hits
  blue circles  frontier waypoints
  red dot       agent

Optional top-left HUD: target goal + most-confirmed scene-graph objects.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, List, Optional, Set, Tuple

import cv2
import numpy as np

from rtnav.tools.visualization.map_visualizer_thread import _HTML_PAGE


class HabitatObsMapWebVizThread(threading.Thread):
    """MJPEG/WebSocket viewer for a standalone HabitatObstacleMap.

    Parameters
    ----------
    habitat_map:
        The :class:`HabitatObstacleMap` instance to render.
    map_lock:
        Lock held while ``habitat_map`` is being updated. The viz acquires it
        before calling ``visualize()`` so we don't read partially-written maps.
    agent_xy_getter:
        Optional zero-arg callable returning ``(x, y)`` in world XY (the same
        frame the map is in). Used to overlay the red agent dot. Return None
        to skip the dot for that frame.
    shutdown_event:
        Setting this stops the server.
    port:
        Starting port. Falls back +1..+15 if busy.
    """

    def __init__(
        self,
        habitat_map=None,
        map_lock: Optional[threading.Lock] = None,
        shutdown_event: Optional[threading.Event] = None,
        habitat_map_getter: Optional[Callable[[], object]] = None,
        agent_xy_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        selected_goal_xy_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        target_nav_xy_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        goal_text_getter: Optional[Callable[[], Optional[str]]] = None,
        found_objects_getter: Optional[Callable[[], Optional[List[Tuple[str, int]]]]] = None,
        sg_nodes_getter: Optional[Callable[[], Optional[List[Tuple]]]] = None,
        shared_state_getter: Optional[Callable[[], object]] = None,
        port: int = 8771,
        jpeg_quality: int = 80,
        view_size: int = 1100,
        fps: int = 4,
    ):
        """Two modes:

        * Pass ``habitat_map`` (+ optional ``map_lock``) for a fixed map
          maintained outside the agent. Original standalone-test use case.
        * Pass ``habitat_map_getter`` instead — a zero-arg callable returning
          the current :class:`HabitatObstacleMap` (or None). Useful when the
          map lives inside an Agent that's rebuilt every episode; the viz
          pulls the active instance per-frame so what it draws is always
          the same state the agent is selecting from.
        """
        super().__init__(name="HabitatObsMapWebVizThread", daemon=True)
        if habitat_map is None and habitat_map_getter is None:
            raise ValueError("HabitatObsMapWebVizThread needs habitat_map or habitat_map_getter")
        self._map = habitat_map
        self._map_getter = habitat_map_getter
        self._lock = map_lock  # may be None when using map_getter
        self._agent_xy_getter = agent_xy_getter
        self._selected_goal_xy_getter = selected_goal_xy_getter
        self._target_nav_xy_getter = target_nav_xy_getter
        self._goal_text_getter = goal_text_getter
        self._found_objects_getter = found_objects_getter
        self._sg_nodes_getter = sg_nodes_getter
        # Agent-map side panel rendered beside the Habitat map.
        self._shared_state_getter = shared_state_getter
        self._map_helper = None
        self.shutdown_event = shutdown_event if shutdown_event is not None else threading.Event()
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.view_size = int(view_size)
        self._period = 1.0 / max(1, int(fps))

        self._jpeg_params = np.array([cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality], dtype=np.int32)
        self._clients: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[HabitatObsMapWebViz] Server error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._loop.close()

    async def _serve(self):
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_http)
        app.router.add_get("/ws", self._handle_websocket)
        runner = web.AppRunner(app)
        await runner.setup()

        site = None
        for candidate in range(self.port, self.port + 16):
            try:
                site = web.TCPSite(runner, "0.0.0.0", candidate)
                await site.start()
                if candidate != self.port:
                    print(
                        f"[HabitatObsMapWebViz] port {self.port} in use - fell back to {candidate}"
                    )
                self.port = candidate
                break
            except OSError as e:
                if e.errno != 98:
                    raise
                site = None
        if site is None:
            raise OSError(f"[HabitatObsMapWebViz] no free port in {self.port}..{self.port + 15}")

        print(f"[HabitatObsMapWebViz] Server running on http://0.0.0.0:{self.port}")

        try:
            while not self.shutdown_event.is_set():
                t0 = time.time()
                await self._broadcast_frame()
                elapsed = time.time() - t0
                await asyncio.sleep(max(0.01, self._period - elapsed))
        finally:
            await runner.cleanup()

    async def _handle_http(self, request):
        from aiohttp import web

        return web.Response(body=_HTML_PAGE, content_type="text/html")

    async def _handle_websocket(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        print(f"[HabitatObsMapWebViz] Client connected ({len(self._clients)} total)")
        try:
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
            print(f"[HabitatObsMapWebViz] Client disconnected ({len(self._clients)} total)")
        return ws

    def _render_and_encode(self) -> Optional[bytes]:
        """Render the frame + LIVE badge and JPEG-encode it. Runs in a worker
        thread (see _broadcast_frame) so the heavy cv2 render never blocks the
        asyncio event loop — a blocked loop stalls the websocket and the client
        drops the connection ("Disconnected")."""
        img = self._render_frame()
        if img is None:
            return None
        cv2.circle(img, (img.shape[1] - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(
            img,
            "LIVE",
            (img.shape[1] - 60, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        ok, buf = cv2.imencode(".jpg", img, self._jpeg_params)
        if not ok:
            return None
        return buf.tobytes()

    async def _broadcast_frame(self):
        if not self._clients:
            return
        # Offload render+encode to a thread so the event loop keeps servicing
        # the websocket (heartbeats/sends) while we draw — otherwise a slow
        # frame blocks the loop and aiohttp resets the socket.
        loop = asyncio.get_event_loop()
        frame_bytes = await loop.run_in_executor(None, self._render_and_encode)
        if frame_bytes is None:
            return

        dead = set()
        for client in self._clients.copy():
            try:
                await client.send_bytes(frame_bytes)
            except Exception:
                dead.add(client)
        self._clients -= dead

    def _render_frame(self) -> Optional[np.ndarray]:
        agent_xy = None
        if self._agent_xy_getter is not None:
            try:
                agent_xy = self._agent_xy_getter()
            except Exception:
                agent_xy = None

        selected_goal_xy = None
        if self._selected_goal_xy_getter is not None:
            try:
                selected_goal_xy = self._selected_goal_xy_getter()
            except Exception:
                selected_goal_xy = None

        target_nav_xy = None
        if self._target_nav_xy_getter is not None:
            try:
                target_nav_xy = self._target_nav_xy_getter()
            except Exception:
                target_nav_xy = None

        sg_confirmed_xys = None
        if self._sg_nodes_getter is not None:
            try:
                sg_confirmed_xys = self._sg_nodes_getter()
            except Exception:
                sg_confirmed_xys = None

        # Pull the active map (dynamic getter) or use the one we own.
        if self._map_getter is not None:
            try:
                habitat_map = self._map_getter()
            except Exception:
                habitat_map = None
            if habitat_map is None:
                return self._placeholder("waiting for agent map ...")
        else:
            habitat_map = self._map

        try:
            if self._lock is not None:
                with self._lock:
                    img = habitat_map.visualize(
                        agent_xy=agent_xy,
                        selected_goal_xy=selected_goal_xy,
                        sg_confirmed_xys=sg_confirmed_xys,
                    )
            else:
                # No lock available (agent's internal map). visualize() reads
                # bool/uint8 arrays; worst case is a mid-update frame, never
                # a crash. The detector updates at ~10 Hz, the viz at 4 Hz.
                img = habitat_map.visualize(
                    agent_xy=agent_xy,
                    selected_goal_xy=selected_goal_xy,
                    target_nav_xy=target_nav_xy,
                    sg_confirmed_xys=sg_confirmed_xys,
                )
        except Exception as e:
            return self._placeholder(f"visualize() raised: {e}")

        if img is None or img.size == 0:
            return self._placeholder("HabitatObstacleMap empty")

        H, W = img.shape[:2]
        scale = self.view_size / max(H, W)
        if scale != 1.0:
            new_w = max(1, int(round(W * scale)))
            new_h = max(1, int(round(H * scale)))
            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )

        self._draw_legend(img)
        self._draw_hud(img)

        # Append the agent's own map as a second, independently rendered panel.
        try:
            right = self._render_map_panel(img.shape[0])
            if right is not None:
                img = self._concat_panels(img, right)
        except Exception as e:
            if not getattr(self, "_map_panel_warned", False):
                self._map_panel_warned = True
                print(f"[HabitatObsMapWebViz] map panel disabled: {e}")
        return img

    @staticmethod
    def _concat_panels(left, right):
        """hconcat two images, normalizing channels (gray/BGRA→BGR) and height."""

        def _bgr(a):
            if a.ndim == 2:
                return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            if a.shape[2] == 4:
                return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
            return a

        left, right = _bgr(left), _bgr(right)
        if right.shape[0] != left.shape[0]:
            s = left.shape[0] / float(right.shape[0])
            right = cv2.resize(
                right,
                (max(1, int(round(right.shape[1] * s))), left.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.hconcat([left, right])

    def _render_map_panel(self, target_h):
        get = self._shared_state_getter
        if get is None:
            return None
        try:
            ss = get()
        except Exception:
            ss = None
        if ss is None:
            return None
        # (Re)build the helper if shared_state changed (new episode).
        if self._map_helper is None or self._map_helper.shared_state is not ss:
            from rtnav.tools.visualization.map_visualizer_thread import MapVisualizerBase

            self._map_helper = MapVisualizerBase(ss)
        panel = self._map_helper.render_decision_frame_clean()
        if panel is None:
            return None
        ph, pw = panel.shape[:2]
        if ph != target_h:
            scale = target_h / float(ph)
            panel = cv2.resize(
                panel,
                (max(1, int(round(pw * scale))), int(target_h)),
                interpolation=cv2.INTER_LINEAR,
            )
        return panel

    def _placeholder(self, msg: str) -> np.ndarray:
        H = self.view_size
        W = self.view_size
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            msg,
            (12, H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_hud(self, img: np.ndarray) -> None:
        """Top-left overlay: 'Target: <goal>' + top confirmed scene-graph objects."""
        goal_text: Optional[str] = None
        if self._goal_text_getter is not None:
            try:
                goal_text = self._goal_text_getter()
            except Exception:
                goal_text = None

        primary: List[Tuple[str, int]] = []
        if self._found_objects_getter is not None:
            try:
                primary = list(self._found_objects_getter() or [])
            except Exception:
                primary = []

        if not goal_text and not primary:
            return

        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        if goal_text:
            lines.append((f"Target: {goal_text}", (0, 215, 255)))  # amber-ish
        else:
            lines.append(("Target: (none)", (180, 180, 180)))

        if primary:
            lines.append(("Primary objects:", (255, 255, 255)))
            for label, count in primary:
                if count > 0:
                    # Seen in scene graph — light green.
                    lines.append((f"  + {label} (x{count})", (180, 255, 180)))
                else:
                    # In active search set but not detected yet — gray.
                    lines.append((f"  · {label}", (200, 200, 200)))
        else:
            lines.append(("Primary objects: (none yet)", (180, 180, 180)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        line_h = 16
        pad = 6
        # Measure widest line for the background box.
        widths = [cv2.getTextSize(s, font, scale, thick)[0][0] for s, _ in lines]
        box_w = max(widths) + 2 * pad
        box_h = line_h * len(lines) + 2 * pad
        x0, y0 = 4, 4
        x1, y1 = x0 + box_w, y0 + box_h
        # Semi-transparent dark background for legibility.
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

        for i, (text, color) in enumerate(lines):
            ty = y0 + pad + (i + 1) * line_h - 4
            cv2.putText(
                img,
                text,
                (x0 + pad, ty),
                font,
                scale,
                (0, 0, 0),
                thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                text,
                (x0 + pad, ty),
                font,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )

    def _draw_legend(self, img: np.ndarray) -> None:
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thick = 1
        items = [
            ("unknown", (255, 255, 255)),
            ("explored", (200, 255, 200)),
            ("unnav", (100, 100, 100)),
            ("obstacle", (0, 0, 0)),
            ("frontier", (200, 0, 0)),
            ("sel frontier", (0, 215, 255)),
            ("nav target", (255, 0, 255)),
            ("agent", (0, 0, 255)),
        ]
        line_h = 14
        sw = 12
        x0 = 8
        y_bottom = H - 8
        for i, (label, color) in enumerate(reversed(items)):
            y = y_bottom - i * line_h
            cv2.rectangle(
                img,
                (x0, y - sw + 2),
                (x0 + sw, y + 2),
                color,
                -1,
            )
            cv2.rectangle(
                img,
                (x0, y - sw + 2),
                (x0 + sw, y + 2),
                (0, 0, 0),
                1,
            )
            tx = x0 + sw + 4
            cv2.putText(
                img,
                label,
                (tx, y),
                font,
                scale,
                (0, 0, 0),
                thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                (tx, y),
                font,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )
