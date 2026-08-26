"""RtNavAgent owns the RtNav worker threads and per-episode reset flow."""

import threading

from rtnav.config.environments.config_loader import load_config
from rtnav.config.model_paths import get_llm_model_dir
from rtnav.core.shared_state import SharedState
from rtnav.modules.decision.decision_thread import DecisionThread
from rtnav.modules.frontier.frontier_detection_thread import FrontierDetectionThread
from rtnav.modules.mapping.mapping_thread import MappingThread
from rtnav.modules.navigation.pointnav.policy_driver import PointNavDriverThread
from rtnav.modules.perception.detector_thread import OpenVocabDetectorThread
from rtnav.modules.perception.perception_thread import PerceptionThread
from rtnav.modules.scenegraph.scene_graph_thread import SceneGraphThread
from rtnav.task.episode_goal import GoalParser
from rtnav.utils.vllm_utils import ensure_vllm_server


class RtNavAgent:
    """Main RtNav runtime: build once, reset per episode, shut down at the end."""

    def __init__(self, env_name: str = "hm3d", verbose: bool = False, robot_api=None):
        self.cfg = load_config(env_name)
        self._robot_api = robot_api
        self.shared_state = SharedState()
        self.shutdown_event = threading.Event()
        self.shared_state.system.robot_api = robot_api

        self._load_models()
        self._build_workers(verbose=verbose)
        self._start_workers()

        print(f"[agent] ready — env={env_name} threads={[t.name for t in self.threads]}")

    def _build_workers(self, verbose: bool) -> None:
        ss, se = self.shared_state, self.shutdown_event

        self.perception = PerceptionThread(ss, se, self.cfg, verbose=verbose)
        self.mapping = MappingThread(ss, se, self.cfg, verbose=verbose)
        self.frontier = FrontierDetectionThread(ss, se, self.cfg)
        self.navigation = PointNavDriverThread(ss, se, robot_api=self._robot_api)
        self.decision = DecisionThread(ss, se, self.cfg)
        self.detector = OpenVocabDetectorThread(ss, se, self.cfg)
        self.scene_graph = SceneGraphThread(ss, se, self.cfg)

        self.threads = [
            self.perception,
            self.mapping,
            self.frontier,
            self.navigation,
            self.detector,
            self.scene_graph,
        ]

    def _start_workers(self) -> None:
        for thread in self.threads:
            thread.start()
        self.decision.start()

    def _load_models(self):
        ensure_vllm_server(get_llm_model_dir())
        print("[agent] LLM ready")
        self._goal_parser = GoalParser.build()

    def reset_episode(
        self,
        episode_hash: int | None = None,
        goal_info: dict | None = None,
        scene_id: str | None = None,
        episode_id: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Soft episode boundary: wipe per-episode state, reset workers,
        then apply the new goal if provided."""
        self.shared_state.task_ready.clear()
        self.shared_state.reset_episode()
        self.perception.reset_episode()
        self.detector.reset_episode()
        self.mapping.reset_episode()
        self.scene_graph.reset_episode()
        self.frontier.reset_episode()
        self.navigation.reset_episode()
        self.decision.reset_episode()  # Decision reset after all workers are cleared.

        logger = self.shared_state.system.vlm_logger
        if logger is not None:
            logger.new_episode(scene_id, episode_id, output_dir=output_dir)
        self._robot_api.reset_episode(episode_hash, preserve_step_id=True)
        self._goal_parser.set_goal(self.shared_state, goal_info)

    def shutdown(self, timeout: float = 3.0):
        self.shutdown_event.set()
        try:
            self.decision.shutdown()
        except Exception as exc:
            print(f"[agent] shutdown: decision teardown failed: {exc}")
        for thread in self.threads:
            try:
                thread.join(timeout=timeout)
            except RuntimeError:
                pass  # thread never started
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            print(f"[agent] WARNING: {len(alive)} threads still alive after shutdown: {alive}")
