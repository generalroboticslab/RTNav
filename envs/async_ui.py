"""Async EnvNode interactive UI (cv2-only; not imported by sync).

Two concerns, env-var gated:
  1. Live viz             — SIM_LIVE_VIZ + SIM_VIZ_WEB (cv2 window + MJPEG)
  2. Keyboard teleop      — ASYNC_KEYBOARD_CONTROL (wasd/qe/x/esc)

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
        def log_message(self, *_args, **_kwargs):  # silence default access log
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
