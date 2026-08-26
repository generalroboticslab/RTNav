import threading
import time

import numpy as np

from rtnav.core.data_types import FrontierOutput


class FrontierDetectionThread(threading.Thread):
    """Continuously detects frontiers from the shared obstacle map at a fixed rate."""

    # Rolling window of per-frontier camera snapshots kept for the image-VLM selector.
    CAMERA_HISTORY_LIMIT = 24

    def __init__(self, shared_state, shutdown_event, cfg):
        super().__init__(name="FrontierDetectionThread", daemon=True)
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        with shared_state.lock:
            self._last_episode_index = shared_state.system.episode_index
            self._last_wipe_counter = shared_state.system.exploration_wipe_counter
            self._spin_was_complete = bool(shared_state.system.bootstrap_spin_complete)
        from rtnav.modules.frontier.vlfm_frontier_detector import (
            VLFMFrontierDetector,
        )

        frontier_cfg = cfg.frontier
        self.detector = VLFMFrontierDetector(
            shared_state,
            min_obstacle_height=frontier_cfg.vlfm_min_obstacle_height,
            max_obstacle_height=frontier_cfg.vlfm_max_obstacle_height,
            floor_drop_height=frontier_cfg.vlfm_floor_drop_height,
            area_thresh_m2=frontier_cfg.vlfm_area_thresh_m2,
            use_local_height_range=frontier_cfg.vlfm_use_local_height_range,
            update_every_n_steps=frontier_cfg.vlfm_update_every_n_steps,
            camera_cfg=cfg.camera,
        )
        self.cfg = frontier_cfg
        self._update_every_n_obs = max(1, frontier_cfg.update_every_n_obs)
        # Publish the first accumulated VLFM map immediately; throttle only
        # the subsequent frontier outputs.
        self._obs_tick_count: int = self._update_every_n_obs - 1

    def reset_episode(self) -> None:
        """Synchronous episode-boundary reset, called by the runner."""
        self._reset()

    def _reset(self) -> bool:
        """Drop detector + frontier shared_state on episode boundary; returns True if reset.
        Atomic check-and-update prevents the runner and this thread from rebuilding twice."""
        with self.shared_state.lock:
            counter = self.shared_state.system.episode_index
            if counter <= self._last_episode_index:
                return False
            self._last_episode_index = counter
        print("[Frontier] Episode reset: clearing detector state")
        self._clear_detector_state()
        return True

    def _consume_wipe_request(self) -> bool:
        """Frontier-exhaustion restart: rebuild the VLFM map mid-episode."""
        with self.shared_state.lock:
            counter = self.shared_state.system.exploration_wipe_counter
            if counter <= self._last_wipe_counter:
                return False
            self._last_wipe_counter = counter
        print("[Frontier] Exploration wipe: rebuilding frontier map")
        self._clear_detector_state()
        return True

    def _clear_detector_state(self) -> None:
        try:
            self.detector.reset()
        except Exception as e:
            print(f"[Frontier] detector.reset() raised: {e}")
        with self.shared_state.lock:
            self.shared_state.frontier.frontier_output = None
            self.shared_state.frontier.frontier_output_cameras = {}
            self.shared_state.frontier.frontier_output_camera_history = []
            self.shared_state.frontier.chosen_frontier_xy = None
            self._spin_was_complete = bool(
                self.shared_state.system.bootstrap_spin_complete
            )
        self._obs_tick_count = self._update_every_n_obs - 1

    def _feed_latest_observation(self) -> None:
        """Keep the VLFM map current independently of frontier publication."""
        try:
            self.detector.update_map()
        except Exception as e:
            print(f"[Frontier] VLFM map update raised: {e}")

    def run(self):
        print("FrontierDetectionThread started")
        from rtnav.utils.task_gate import wait_for_task_ready

        if not wait_for_task_ready(self.shared_state, "Frontier", self.shutdown_event):
            return
        with self.shared_state.lock:
            self._last_episode_index = self.shared_state.system.episode_index

        # Wait for mapping to be ready, but feed VLFM throughout the bootstrap spin.
        while not self.shutdown_event.is_set():
            self._feed_latest_observation()
            with self.shared_state.lock:
                obstacle_map = self.shared_state.mapping.obstacle_map
                mapping_output = self.shared_state.mapping.mapping_output
            if obstacle_map is not None and mapping_output is not None:
                explored_count = int(np.sum(obstacle_map.explored > 0))
                if explored_count > self.cfg.min_explored_threshold:
                    print(
                        f"[Frontier] Map ready (explored={explored_count} cells, "
                        f"threshold={self.cfg.min_explored_threshold}), starting detection"
                    )
                    break
            self.shutdown_event.wait(0.02)

        last_map_timestamp = None
        _logged_first_clusters = False

        while not self.shutdown_event.is_set():
            # Drop accumulated map state at episode boundaries (esp. vlfm,
            # whose HabitatObstacleMap would otherwise carry over to next scene).
            if self._reset() or self._consume_wipe_request():
                last_map_timestamp = None
                _logged_first_clusters = False
            # Per-iteration task_ready gate: pause for the whole reset
            # window, then resume once the new goal has been applied.
            if not wait_for_task_ready(self.shared_state, "Frontier", self.shutdown_event):
                break
            with self.shared_state.lock:
                spin_complete_before_update = bool(
                    self.shared_state.system.bootstrap_spin_complete
                )
            force_post_spin_publish = (
                spin_complete_before_update and not self._spin_was_complete
            )
            self._spin_was_complete = spin_complete_before_update
            self._feed_latest_observation()
            with self.shared_state.lock:
                obstacle_map = self.shared_state.mapping.obstacle_map
                mapping_output = self.shared_state.mapping.mapping_output

            if obstacle_map is None:
                time.sleep(0.1)
                continue

            current_timestamp = mapping_output.timestamp if mapping_output else None
            if (
                not force_post_spin_publish
                and current_timestamp is not None
                and current_timestamp == last_map_timestamp
            ):
                time.sleep(0.05)
                continue
            last_map_timestamp = current_timestamp

            # Throttle: only run the (expensive) detector pipeline every
            # Nth fresh map output. In between, leave the previous
            # FrontierOutput in shared_state.
            self._obs_tick_count += 1
            if (
                not force_post_spin_publish
                and (self._obs_tick_count % self._update_every_n_obs) != 0
            ):
                time.sleep(0.01)
                continue

            try:
                frontier_mask, frontier_clusters = self.detector.detect(obstacle_map)
                vlfm_map_snapshot = self.detector.map_snapshot()
                unexplored_directions = list(
                    self.detector.last_frontier_unexplored_directions or []
                )

                if len(frontier_clusters) > 0:
                    centroids = self.detector.cluster_centroids(frontier_clusters)
                else:
                    centroids = np.empty((0, 2))

                frontier_output = FrontierOutput(
                    frontier_mask=frontier_mask,
                    frontier_clusters=frontier_clusters,
                    frontier_centroids=centroids,
                    frontier_unexplored_directions=unexplored_directions,
                    vlfm_map_snapshot=vlfm_map_snapshot,
                    timestamp=time.time(),
                )

                with self.shared_state.lock:
                    # If the spin ended while this detection was running,
                    # PointNav has invalidated its result. The next loop forces
                    # a replacement after the latest view has been ingested.
                    if (
                        not spin_complete_before_update
                        and self.shared_state.system.bootstrap_spin_complete
                    ):
                        continue
                    self.shared_state.frontier.frontier_output = frontier_output
                    camera_snapshot = dict(self.shared_state.sensor.latest_cameras or {})
                    camera_history = list(
                        self.shared_state.frontier.frontier_output_camera_history or []
                    )
                    if camera_snapshot:
                        camera_history.append(camera_snapshot)
                        camera_history = camera_history[-self.CAMERA_HISTORY_LIMIT :]
                    self.shared_state.frontier.frontier_output_cameras = camera_snapshot
                    self.shared_state.frontier.frontier_output_camera_history = camera_history

                if force_post_spin_publish:
                    print("[Frontier] Post-spin VLFM catch-up map published")

                if len(frontier_clusters) > 0 and not _logged_first_clusters:
                    _logged_first_clusters = True
                    print(
                        f"[Frontier] First frontier clusters: {len(frontier_clusters)} "
                        f"(map had enough explored/unexplored boundary)"
                    )

            except Exception as e:
                print(f"FrontierDetectionThread error: {e}")
                import traceback

                traceback.print_exc()
