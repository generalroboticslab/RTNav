"""Per-decision-step recorder — captures one captioned frame (egocentric RGB |
rendered map) each time the agent makes a *meaningful* decision (a new frontier
LLM pick, a target commit, or a mode switch). The frames are assembled, offline,
into a per-episode "filmstrip" figure (see results/assemble_decision_filmstrip.py).

When ``--record_decisions`` is enabled, agent_node creates one recorder and
publishes it through ``shared_state.system.decision_recorder``.
  - DecisionThread._send_goal() calls recorder.record(source, goal_xy, caption)
    on every published goal; the recorder dedupes consecutive same-goal picks so
    only genuinely new decisions produce a frame.

Rendering is done on a worker thread so the decision tick is never blocked: the
egocentric RGB is copied at decision time (cheap); the map is rendered shortly
after from shared_state (it changes slowly, so the small lag is harmless).
"""

import os
import queue
import shutil
import threading

import cv2
import numpy as np

from rtnav.tools.visualization.map_visualizer_thread import MapVisualizerBase


class DecisionStepRecorder:
    def __init__(self, shared_state, enabled: bool = False):
        self.shared_state = shared_state
        self.enabled = enabled
        self._renderer = MapVisualizerBase(shared_state)
        self._ep_dir = None
        self._scene = ""
        self._episode = ""
        self._goal_text = ""
        self._step = 0
        self._last_key = None  # (source, round(x,1), round(y,1)) → dedupe
        self._q: "queue.Queue" = queue.Queue(maxsize=256)
        self._write_lock = threading.Lock()  # serialize disk writes (worker + record_sync)
        self._shutdown = threading.Event()
        self._worker = None
        if self.enabled:
            self._worker = threading.Thread(
                target=self._worker_loop, name="DecisionStepRecorder", daemon=True
            )
            self._worker.start()

    # ── public API (mirrors EpisodeVideoRecorder) ────────────────────────────
    def new_episode(self, scene_id: str, episode_id, output_dir: str = None):
        if not self.enabled:
            return
        self._scene = os.path.basename(str(scene_id)).split(".")[0]
        self._episode = str(episode_id)
        self._step = 0
        self._last_key = None
        with self.shared_state.lock:
            self._goal_text = str(getattr(self.shared_state.task, "goal_category", "") or "")
        base = output_dir or self._default_run_dir()
        self._ep_dir = os.path.join(
            base, "decision_steps", f"{self._scene}_ep{self._episode}_{self._goal_text}"
        )
        try:
            # Overwrite fresh each run: wipe any frames/captions left from a
            # previous run of this same episode so old and new don't mix.
            shutil.rmtree(self._ep_dir, ignore_errors=True)
            os.makedirs(self._ep_dir, exist_ok=True)
        except Exception as exc:
            print(f"[DecisionRec] could not create {self._ep_dir}: {exc}")
            self._ep_dir = None

    def record(self, source: str, goal_xy, caption: str):
        """Called from the decision thread on every published goal. Fast: dedupes
        and enqueues a copy of the current RGB; the worker renders + saves."""
        if not self.enabled or self._ep_dir is None or goal_xy is None:
            return
        try:
            key = (str(source), round(float(goal_xy[0]), 1), round(float(goal_xy[1]), 1))
        except (TypeError, ValueError, IndexError):
            return
        if key == self._last_key:
            return  # same decision as last time → not a new decision
        self._last_key = key
        rgb, env_step = self._grab_obs()  # copied now, at decision time
        seq = self._step  # decision order, for stable filename sorting
        self._step += 1
        try:
            self._q.put_nowait((seq, env_step, source, tuple(goal_xy[:2]), str(caption), rgb))
        except queue.Full:
            pass

    def record_sync(self, source: str, goal_xy, caption: str):
        """Like record() but renders + writes on the calling thread, so the frame is on
        disk before this returns. Used for the commit 'stop' frame, which the async queue
        would otherwise drop when the episode resets immediately after committing."""
        if not self.enabled or self._ep_dir is None or goal_xy is None:
            return
        try:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
        except (TypeError, ValueError, IndexError):
            return
        rgb, env_step = self._grab_obs()
        seq = self._step
        self._step += 1
        try:
            with self._write_lock:
                self._save_frame(seq, env_step, source, (gx, gy), str(caption), rgb)
        except Exception as exc:
            print(f"[DecisionRec] sync stop frame failed: {exc}")

    def record_image(self, source: str, rgb, caption: str, goal_xy=(0.0, 0.0)):
        """Save a CALLER-PROVIDED image (e.g. the VLM-verify frame with its box)
        plus a caption — unlike record(), which snapshots the current obs. Not
        deduped (every verify is its own frame)."""
        if not self.enabled or self._ep_dir is None or rgb is None:
            return
        seq = self._step
        self._step += 1
        _, env_step = self._grab_obs()  # for the step-number annotation only
        try:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
        except (TypeError, ValueError, IndexError):
            gx, gy = 0.0, 0.0
        try:
            self._q.put_nowait(
                (seq, env_step, source, (gx, gy), str(caption), np.asarray(rgb).copy())
            )
        except queue.Full:
            pass

    def save_episode(self, metrics: dict = None):
        # Frames are written as they happen; nothing to flush. Kept for symmetry
        # with the video recorder's call sites.
        return

    def shutdown(self):
        self._shutdown.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    # ── internals ────────────────────────────────────────────────────────────
    def _default_run_dir(self):
        return os.environ.get("RUN_DIR") or "/tmp/rtnav_decisions"

    def _grab_obs(self):
        """Return (rgb_copy, env_step) from the latest observation. env_step is
        the env/sim step_id (elapsed-time proxy), -1 if unavailable."""
        try:
            with self.shared_state.lock:
                hab = getattr(self.shared_state.sensor, "habitat_obs", None)
                rgb = getattr(hab, "rgb", None) if hab is not None else None
                env_step = int(getattr(hab, "step_id", -1)) if hab is not None else -1
            rgb = np.asarray(rgb).copy() if rgb is not None else None
            return rgb, env_step
        except Exception:
            return None, -1

    def _worker_loop(self):
        while not self._shutdown.is_set():
            try:
                seq, env_step, source, goal_xy, caption, rgb = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with self._write_lock:
                    self._save_frame(seq, env_step, source, goal_xy, caption, rgb)
            except Exception as exc:
                print(f"[DecisionRec] frame seq={seq} failed: {exc}")

    def _render_map(self):
        # Compact agent map; the Habitat trajectory is rendered separately.
        try:
            return self._renderer.render_decision_frame_clean()
        except Exception:
            return None

    def _save_frame(self, seq, env_step, source, goal_xy, caption, rgb):
        if self._ep_dir is None:
            return
        map_img = self._render_map()
        panels = []
        if rgb is not None and rgb.size:
            panels.append(cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR))
        if map_img is not None and getattr(map_img, "size", 0):
            panels.append(np.asarray(map_img, np.uint8))
        if not panels:
            return
        # Match heights, hconcat egocentric | map
        H = max(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            if p.shape[0] != H:
                w = int(p.shape[1] * H / p.shape[0])
                p = cv2.resize(p, (w, H))
            resized.append(p)
        body = resized[0] if len(resized) == 1 else cv2.hconcat(resized)

        # Caption banner along the top — header line + one line per caption row
        # (frontier selections are multi-line: one row per candidate frontier).
        step_str = f"env step {env_step}" if env_step >= 0 else "env step ?"
        head = f"{step_str}  [{source}]  goal=({goal_xy[0]:+.1f},{goal_xy[1]:+.1f})"
        cap_lines = [ln for ln in str(caption).split("\n") if ln.strip()]
        line_h = 18
        banner_h = 30 + line_h * (len(cap_lines) + 1)
        banner = np.full((banner_h, body.shape[1], 3), 30, np.uint8)
        cv2.putText(
            banner, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA
        )
        y = 22 + line_h + 6
        for ln in cap_lines:
            ln = ln if len(ln) <= 90 else ln[:87] + "..."
            cv2.putText(
                banner, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
            )
            y += line_h
        frame = cv2.vconcat([banner, body])

        # seq keeps frames in decision order for the assembly's lexical sort;
        # env_step (elapsed-time proxy) is in the name and banner.
        fname = f"step_{seq:04d}_env{max(env_step, 0):05d}_{source}.png"
        cv2.imwrite(os.path.join(self._ep_dir, fname), frame)
        # Sidecar caption text for the assembly script (and easy grepping).
        try:
            cap_oneline = str(caption).replace("\n", " | ")
            with open(os.path.join(self._ep_dir, "captions.tsv"), "a") as f:
                f.write(
                    f"{seq}\t{env_step}\t{source}\t"
                    f"{goal_xy[0]:.2f}\t{goal_xy[1]:.2f}\t{cap_oneline}\n"
                )
        except Exception:
            pass
