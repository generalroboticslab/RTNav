"""Async EnvNode interactive UI (cv2-only; not imported by sync).

Three concerns, env-var gated:
  1. Live viz             — SIM_LIVE_VIZ + SIM_VIZ_WEB (cv2 window + MJPEG)
  2. Keyboard teleop      — ASYNC_KEYBOARD_CONTROL (wasd/qe/x/esc)
  3. Floor-plan PNG dump  — per-episode (top-down navmesh + GT markers)

``VizState`` / ``KbdState`` are owned by env_node and passed by reference;
helpers mutate them without touching node attributes.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# 1. Live viz (cv2 window + MJPEG web server)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class VizState:
    """Viz toggles + JPEG slot for the MJPEG server."""

    live_viz: bool = False
    web_viz: bool = False
    web_port: int = 8767
    # JPEG slot — lock-guarded (server-thread reads, env-thread writes).
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_jpeg: Optional[bytes] = None
    warned: bool = False  # log render errors once


def render_panel(obs: dict, step_id: int, episode_count: int) -> Optional[np.ndarray]:
    """``RGB | depth-heatmap`` BGR canvas with step/episode caption,
    or None if neither rgb nor depth is renderable."""
    import cv2

    rgb = obs.get("rgb")
    depth = obs.get("depth")
    panels = []

    if rgb is not None:
        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3:
            panels.append(cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2BGR))

    if depth is not None:
        d_arr = np.asarray(depth).squeeze()
        if d_arr.ndim == 2:
            d_max = float(np.nanmax(d_arr)) if np.isfinite(d_arr).any() else 1.0
            d_max = max(d_max, 1e-3)
            d_u8 = np.clip(d_arr / d_max, 0.0, 1.0)
            d_u8 = (d_u8 * 255).astype(np.uint8)
            panels.append(cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO))

    if not panels:
        return None

    h = max(p.shape[0] for p in panels)
    panels = [
        cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) if p.shape[0] != h else p
        for p in panels
    ]
    canvas = np.hstack(panels)

    cv2.putText(
        canvas, f"step {step_id}  ep {episode_count}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return canvas


def show_live_viz(
    obs: dict,
    step_id: int,
    episode_count: int,
    viz_state: VizState,
    *,
    skip_waitkey: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """Render + dispatch one frame to cv2 window and/or MJPEG slot.
    ``skip_waitkey=True`` when caller already polls via :func:`poll_keyboard_cmd`."""
    if not (viz_state.live_viz or viz_state.web_viz):
        return
    try:
        import cv2

        canvas = render_panel(obs, step_id, episode_count)
        if canvas is None:
            return

        if viz_state.live_viz:
            cv2.imshow("sim: RGB | depth", canvas)
            if not skip_waitkey:
                cv2.waitKey(1)

        if viz_state.web_viz:
            ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with viz_state.lock:
                    viz_state.latest_jpeg = jpeg.tobytes()
    except Exception as e:
        if not viz_state.warned:
            viz_state.warned = True
            if logger is not None:
                logger.warn(f"[sim-viz] render failed (suppressing further warnings): {e}")


def start_viz_web_server(
    viz_state: VizState,
    step_id_getter: Callable[[], int],
    *,
    logger: Optional[Any] = None,
) -> Optional[ThreadingHTTPServer]:
    """MJPEG HTTP server in a daemon thread. Returns the server (or None on
    bind failure — we flip ``viz_state.web_viz`` off so subsequent
    :func:`show_live_viz` calls skip JPEG encoding)."""
    state = viz_state

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw):  # silence default access log
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    b"<!doctype html><meta charset=utf-8>"
                    b"<title>env_node sim viz</title>"
                    b"<style>body{margin:0;background:#111;color:#ddd;"
                    b"font-family:sans-serif;text-align:center}"
                    b"img{max-width:100vw;max-height:100vh;display:block;"
                    b"margin:0 auto}</style>"
                    b"<h3>env_node: RGB | depth</h3>"
                    b'<img src="/stream.mjpg" alt="stream">'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/frame.jpg":
                with state.lock:
                    jpeg = state.latest_jpeg
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path == "/stream.mjpg":
                boundary = "frame"
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={boundary}",
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                last_id = -1
                try:
                    while True:
                        with state.lock:
                            jpeg = state.latest_jpeg
                        step_id = step_id_getter()
                        if jpeg is not None and step_id != last_id:
                            last_id = step_id
                            self.wfile.write(b"--" + boundary.encode() + b"\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            )
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.03)  # ~30 fps cap; producer caps too
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_error(404)

    # Try the requested port; if it's taken (orphan from a prior crashed
    # run, or another viz on the same port), step up to web_port+15. This
    # mirrors WebMapVisualizerThread's fallback so a stale TIME_WAIT socket
    # never wedges sim-viz-web for the user.
    requested_port = state.web_port
    srv = None
    last_err: Optional[Exception] = None
    for candidate in range(requested_port, requested_port + 16):
        try:
            srv = ThreadingHTTPServer(("0.0.0.0", candidate), _Handler)
            if candidate != requested_port and logger is not None:
                logger.warn(
                    f"[sim-viz-web] port {requested_port} in use → "
                    f"fell back to {candidate}"
                )
            state.web_port = candidate
            break
        except OSError as e:
            if e.errno != 98:  # EADDRINUSE — keep trying for that one only
                last_err = e
                break
            last_err = e
            srv = None
    if srv is None:
        if logger is not None:
            logger.error(
                f"[sim-viz-web] no free port in "
                f"{requested_port}..{requested_port + 15} ({last_err})"
            )
        state.web_viz = False
        return None

    thread = threading.Thread(target=srv.serve_forever, name="SimVizWebServer", daemon=True)
    thread.start()
    if logger is not None:
        logger.info(
            f"[sim-viz-web] streaming at http://0.0.0.0:{state.web_port}/  "
            f"(also /stream.mjpg, /frame.jpg)"
        )
    return srv


# ─────────────────────────────────────────────────────────────────────────
# 2. Keyboard teleop (cv2.waitKeyEx-based)
# ─────────────────────────────────────────────────────────────────────────

# Active-key model — one key per axis. (axis, sign) per key:
#   axis 0 = linear x  | w/s = +/- forward
#   axis 1 = linear y  | a/d = +/- strafe (off unless ASYNC_KEYBOARD_STRAFE_SPEED>0)
#   axis 2 = angular z | q/e = +/- yaw (CCW/CW)
_KEY_MAP = {
    ord("w"): (0, +1.0),
    ord("s"): (0, -1.0),
    ord("a"): (1, -1.0),
    ord("d"): (1, +1.0),
    ord("q"): (2, +1.0),
    ord("e"): (2, -1.0),
}
_AXIS_BY_KEY = {k: v[0] for k, v in _KEY_MAP.items()}


@dataclass
class KbdState:
    """Keyboard teleop state (owned by env_node)."""

    enabled: bool = False
    quit_requested: bool = False
    lin: float = 1.0
    ang: float = 1.5
    strafe: float = 0.0
    active_keys: list = field(default_factory=list)
    cv2: object = None  # imported cv2 module, or None if unavailable


def make_kbd_state(enabled: bool, *, logger=None) -> KbdState:
    """Build KbdState from env vars + try to import cv2."""
    st = KbdState(
        enabled=enabled,
        lin=float(os.environ.get("ASYNC_KEYBOARD_LINEAR_SPEED", "1.0")),
        ang=float(os.environ.get("ASYNC_KEYBOARD_ANGULAR_SPEED", "1.5")),
        strafe=float(os.environ.get("ASYNC_KEYBOARD_STRAFE_SPEED", "0.0")),
    )
    if enabled:
        try:
            import cv2
            st.cv2 = cv2
            if logger is not None:
                logger.info(
                    "Keyboard teleop enabled (async): "
                    "w/s=fwd/back, a/d=strafe left/right, "
                    "q/e=yaw ccw/cw, f=clear, x|esc=stop episode"
                )
        except Exception as e:
            st.enabled = False
            if logger is not None:
                logger.warn(f"ASYNC_KEYBOARD_CONTROL enabled but cv2 unavailable: {e}")
    return st


def poll_keyboard_cmd(state: KbdState) -> Optional[Tuple[float, float, float]]:
    """Read one keypress (cv2.waitKeyEx) and return ``(lin, lin_y, ang)``.

    Key mapping:
      - w / s            → forward / back     (sets axis 0)
      - a / d            → strafe left/right  (sets axis 1; off unless ASYNC_KEYBOARD_STRAFE_SPEED>0)
      - q / e            → yaw CCW / CW       (sets axis 2)
      - f / F            → stop motion        (returns 0, 0, 0)
      - x / X / ESC      → end the episode    (sets state.quit_requested)

    For w/s/a/d/q/e a new press REPLACES the previous active key on the same
    axis (so pressing ``d`` after ``a`` flips strafe direction without you
    needing to release the first key). Returned tuple is the current velocity
    setpoint synthesised from whichever keys are still active across the 3
    axes — so 0/0/0 means "no axis active right now", which is what ``f``
    produces.

    Returns None when teleop is disabled or cv2 is missing.
    """
    if not state.enabled or state.cv2 is None:
        return None

    cv2 = state.cv2
    key = cv2.waitKeyEx(1)
    if 0 <= key <= 255:
        ch = chr(key)
        if ch.isalpha():
            key = ord(ch.lower())
    key = key & 0xFF

    # f / F → emit a hard zero command immediately. Clearing active_keys
    # would also produce zero on the next tick, but returning (0, 0, 0)
    # here makes the "stop now" intent explicit at the call site.
    if key in (ord("f"), ord("F")):
        state.active_keys.clear()
        return (0.0, 0.0, 0.0)

    # x / X / ESC → flag quit + emit zero. The env_node tick will see
    # quit_requested before it consumes this command, but we still send
    # zeros in case the caller bypasses the quit check.
    if key in (ord("x"), ord("X"), 27):
        state.quit_requested = True
        state.active_keys.clear()
        return (0.0, 0.0, 0.0)

    # w/s/a/d/q/e → replace any previous press on the same axis. Falls
    # through to the synthesise step below so the new setpoint is emitted
    # right away without waiting for the next tick.
    if key in _KEY_MAP:
        axis = _AXIS_BY_KEY[key]
        state.active_keys = [k for k in state.active_keys if _AXIS_BY_KEY[k] != axis]
        state.active_keys.append(key)

    # Synthesise the current (lin, lin_y, ang) setpoint from whichever
    # keys are still active across the 3 axes.
    cmd = [0.0, 0.0, 0.0]
    for k in state.active_keys:
        axis, sign = _KEY_MAP[k]
        if axis == 0:
            cmd[axis] = sign * state.lin
        elif axis == 1:
            cmd[axis] = sign * state.strafe
        else:
            cmd[axis] = sign * state.ang
    return (cmd[0], cmd[1], cmd[2])


# ─────────────────────────────────────────────────────────────────────────
# 3. Floor-plan PNG dump (per episode)
# ─────────────────────────────────────────────────────────────────────────

# Per-scene topdown navmesh cache: scene_id → (tdmap_color, (h, w)). The
# navmesh doesn't change across episodes within a scene, so the slow
# get_topdown_map_from_sim raycast only runs once per scene. Big win on
# OVON / HM3D where hundreds of episodes share each scene.
_TDMAP_CACHE: dict = {}


def _scene_cache_key(sim) -> str:
    try:
        return str(sim.curr_scene_name)
    except Exception:
        try:
            return str(sim.config.sim_cfg.scene_id)
        except Exception:
            return "unknown"


def save_floor_plan(
    env_wrapper: Any,
    episode: Any,
    reset_info: dict,
    output_dir: str,
    *,
    logger: Optional[Any] = None,
) -> None:
    """Write ``<output_dir>/floor_plan.png``: top-down navmesh + start (blue),
    episode goal (yellow ✕), other same-category instances (orange).

    Sim queries (navmesh, semantic_annotations, to_grid) run synchronously on
    the caller's thread — they MUST, since Habitat-sim isn't thread-safe.
    Everything CPU-only (cv2 drawing, PNG encode) is dispatched to a daemon
    thread so the env's sim tick doesn't stall waiting on matplotlib /
    imwrite. Same-scene navmesh is cached so subsequent episodes only pay
    the per-episode marker overlay cost (~tens of ms).

    Best-effort — any exception is logged at warning and we return silently."""
    try:
        from habitat.utils.visualizations import maps

        sim = env_wrapper._env.unwrapped.habitat_env.sim
        cache_key = _scene_cache_key(sim)
        cached = _TDMAP_CACHE.get(cache_key)
        if cached is None:
            tdmap = maps.get_topdown_map_from_sim(sim, map_resolution=1024, draw_border=True)
            tdmap_color = maps.colorize_topdown_map(tdmap)
            h, w = tdmap.shape[:2]
            _TDMAP_CACHE[cache_key] = (tdmap_color, (h, w))
        else:
            tdmap_color, (h, w) = cached

        def hab_pos_to_px(pos):
            r, c = maps.to_grid(pos[2], pos[0], (h, w), sim=sim)
            return int(c), int(r)

        # Snapshot per-episode markers (sim-bound — must be on this thread).
        markers = []
        sp = reset_info.get("start_position")
        if sp:
            markers.append(("start", hab_pos_to_px(sp)))

        goal_cat = str(reset_info.get("goal_category", "") or "")
        try:
            sem_scene = sim.semantic_annotations()
            target_ids = set()
            for goal in (episode.goals or []):
                oid = getattr(goal, "object_id", None)
                if oid is not None:
                    target_ids.add(int(oid))
            target_cat_names = set()
            for obj in (sem_scene.objects or []):
                if obj is None:
                    continue
                if int(obj.id) in target_ids:
                    cat = getattr(obj, "category", None)
                    if cat is not None:
                        target_cat_names.add(cat.name().lower())
            if not target_cat_names:
                target_cat_names = {goal_cat.lower()}

            seen_px = set()
            for obj in (sem_scene.objects or []):
                if obj is None:
                    continue
                aabb = getattr(obj, "aabb", None)
                if aabb is None:
                    continue
                cat = getattr(obj, "category", None)
                cat_name = cat.name().lower() if cat is not None else ""
                if not any(t in cat_name for t in target_cat_names):
                    continue
                pos = aabb.center
                px_key = (round(float(pos[0]), 2), round(float(pos[2]), 2))
                if px_key in seen_px:
                    continue
                seen_px.add(px_key)
                ox, oy = hab_pos_to_px([float(pos[0]), float(pos[1]), float(pos[2])])
                kind = "goal" if int(obj.id) in target_ids else "other"
                markers.append((kind, (ox, oy)))
        except Exception:
            for goal in (episode.goals or []):
                pos = getattr(goal, "position", None)
                if pos is None:
                    continue
                gx, gy = hab_pos_to_px([float(pos[0]), float(pos[1]), float(pos[2])])
                markers.append(("goal", (gx, gy)))

        label = f"{goal_cat}  ep={episode.episode_id}"
        path = os.path.join(output_dir, "floor_plan.png")
        # Pass a COPY of the cached base — bg thread mutates it via cv2.draw*.
        base_img = tdmap_color.copy()

        # Async: pure CPU drawing + PNG encode. Daemon so it doesn't block
        # env_node shutdown.
        import threading
        threading.Thread(
            target=_render_floor_plan_async,
            args=(base_img, markers, label, path, logger),
            daemon=True,
            name=f"floor_plan_{episode.episode_id}",
        ).start()
    except Exception as exc:
        if logger is not None:
            logger.warning(f"[floor_plan] failed to save: {exc}")


def _render_floor_plan_async(base_img, markers, label, path, logger):
    """Pure CPU: draw markers on the colored navmesh, save PNG. No sim access."""
    try:
        import cv2
        goal_color = (0, 215, 255)   # BGR yellow
        other_color = (0, 140, 255)  # BGR orange
        for kind, (x, y) in markers:
            if kind == "start":
                cv2.circle(base_img, (x, y), 12, (0, 0, 255), -1)
                cv2.circle(base_img, (x, y), 14, (255, 255, 255), 2)
            elif kind == "goal":
                r = 12
                cv2.line(base_img, (x - r, y), (x + r, y), goal_color, 3)
                cv2.line(base_img, (x, y - r), (x, y + r), goal_color, 3)
                d = int(r * 0.7)
                cv2.line(base_img, (x - d, y - d), (x + d, y + d), goal_color, 3)
                cv2.line(base_img, (x + d, y - d), (x - d, y + d), goal_color, 3)
                cv2.circle(base_img, (x, y), r + 3, goal_color, 2)
            elif kind == "other":
                cv2.circle(base_img, (x, y), 8, other_color, -1)
                cv2.circle(base_img, (x, y), 10, (255, 255, 255), 1)
        cv2.putText(base_img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        out = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, out)
        if logger is not None:
            logger.info(f"[floor_plan] saved → {path}")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"[floor_plan] async save failed: {exc}")
