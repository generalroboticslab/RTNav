"""Mapping thread: integrate perception point clouds into the obstacle map."""

import threading
import time

from rtnav.modules.mapping.obstacle_map.obstacle_map import ObstacleMap
from rtnav.utils.task_gate import wait_for_task_ready


class MappingThread(threading.Thread):
    # Skip the first few frames each episode while scene rendering settles.
    _WARMUP_FRAMES = 3

    def __init__(self, shared_state, shutdown_event, cfg, verbose=False):
        super().__init__(name="MappingThread", daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        self.verbose = verbose
        self.mapping_cfg = cfg.mapping
        self.obstacle_map = self._new_obstacle_map()
        with self.shared_state.lock:
            self.shared_state.mapping.obstacle_map = self.obstacle_map
            self._last_episode_index = self.shared_state.system.episode_index
            self._last_wipe_counter = self.shared_state.system.exploration_wipe_counter

    def _new_obstacle_map(self) -> ObstacleMap:
        m = self.mapping_cfg
        return ObstacleMap(
            size=m.obstacle_map_size,
            pixels_per_meter=m.pixels_per_meter,
            config=m,
        )

    def reset_episode(self) -> None:
        """Synchronous episode-boundary reset, called by the runner."""
        self._reset()

    def _reset(self):
        """Rebuild the obstacle map when the per-episode reset counter advances.
        Guarded by the lock so the runner's sync call and this thread's own loop
        cannot both rebuild."""
        with self.shared_state.lock:
            counter = self.shared_state.system.episode_index
            if counter <= self._last_episode_index:
                return
            self._last_episode_index = counter
        self.obstacle_map = self._new_obstacle_map()
        with self.shared_state.lock:
            self.shared_state.mapping.obstacle_map = self.obstacle_map
            self.shared_state.mapping.mapping_output = None

    def _consume_wipe_request(self):
        """Frontier-exhaustion restart: wipe cumulative explored state, keep obstacles."""
        with self.shared_state.lock:
            counter = self.shared_state.system.exploration_wipe_counter
            if counter <= self._last_wipe_counter:
                return
            self._last_wipe_counter = counter
        self.obstacle_map.clear_explored()
        print("[Mapping] Explored map cleared (frontier-exhaustion restart)")

    def run(self):
        last_version = -1
        warmup_counter = self._last_episode_index
        frames = 0

        while not self.shutdown_event.is_set():
            self._reset()
            self._consume_wipe_request()
            if not wait_for_task_ready(self.shared_state, "MappingThread", self.shutdown_event):
                break

            with self.shared_state.lock:
                episode = self.shared_state.system.episode_index
                perception = self.shared_state.perception.perception_output
                version = self.shared_state.perception.perception_version
                fallen = self.shared_state.system.robot_fallen

            if episode != warmup_counter:  # re-arm warmup at each episode boundary
                warmup_counter = episode
                last_version = -1
                frames = 0

            if perception is None or version == last_version:
                time.sleep(0.01)
                continue
            last_version = version

            frames += 1
            if frames <= self._WARMUP_FRAMES or fallen:
                continue

            t0 = time.time()
            mapping_output = self.obstacle_map.update_from_perception(perception)

            # Drop the result if an episode reset landed during integration.
            with self.shared_state.lock:
                if self.shared_state.system.episode_index != episode:
                    continue
                self.shared_state.mapping.mapping_output = mapping_output

            if self.verbose:
                print(f"[Mapping] step {(time.time() - t0) * 1000:.0f}ms")
