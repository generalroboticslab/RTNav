"""Benchmark entrypoint for the RtNav agent."""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

# Bootstrap the agent so ROS and RtNav imports resolve from this package's own root.
_RTNAV_ROOT = os.environ.get("RTNAV_ROOT", str(Path(__file__).resolve().parent))
sys.path.insert(0, _RTNAV_ROOT)
os.chdir(_RTNAV_ROOT)

from base_agent import BaseAgentNode, run_agent
from rclpy.executors import MultiThreadedExecutor
from ros_msgs.msg import Observation as ObservationMsg

from rtnav.modules.ros_interface.observation_subscriber import ObservationSubscriberNode
from rtnav.runner import RtNavAgent
from rtnav.utils.run_dir import resolve_agent_run_dir


class RTNavEvalNode(BaseAgentNode):
    """Owns the benchmark loop and episode bookkeeping."""

    def __init__(self, args: argparse.Namespace, mode: str = "async"):
        super().__init__("rtnav_eval", mode)
        self._args = args
        self._episodes_remaining = True
        self._agent: RtNavAgent | None = None
        self._observation_subscriber: ObservationSubscriberNode | None = None
        self._ros_executor: MultiThreadedExecutor | None = None
        self._ros_spin_thread: threading.Thread | None = None
        self._decision_recorder = None
        self._vlm_logger = None
        self._run_dir: Path | None = None
        self._visualizer_threads: list[threading.Thread] = []
        self._episode_done = threading.Event()
        self._video = None

    def run_eval(self, num_episodes: int = -1) -> None:
        self._wait_for_environment()
        self._build_runtime()

        episode_index = 0
        try:
            while self._episodes_remaining:
                if 0 < num_episodes <= episode_index:
                    break

                episode = self._start_episode()
                if episode is None:
                    break

                episode_index += 1
                print(
                    f"[eval] [ep {episode_index}] scene={os.path.basename(episode.scene_id)} "
                    f"id={episode.episode_id} goal='{self._agent.shared_state.task.goal_category}' "
                    f"remaining={episode.episodes_remaining}"
                )
                self._wait_for_episode_end(episode_index)
                self._save_episode()
        finally:
            self._shutdown()

        print(f"[eval] complete — {episode_index} episodes run.")

    def _wait_for_environment(self) -> None:
        self.get_logger().info("Waiting for env services …")
        if not self._cli_reset.wait_for_service(timeout_sec=300):
            raise RuntimeError("Timed out waiting 300s for /reset_episode.")

    def _build_runtime(self) -> None:
        self._agent = RtNavAgent(
            env_name=os.environ["BENCHMARK"],
            verbose=self._args.verbose,
            robot_api=self._robot_api,
        )

        from rtnav.tools.visualization.vlm_decision_logger import VLMDecisionLogger

        self._vlm_logger = VLMDecisionLogger()
        with self._agent.shared_state.lock:
            self._agent.shared_state.system.vlm_logger = self._vlm_logger

        if self._record_async_observations:
            from rtnav.tools.agent_video import AgentVideoRecorder

            self._video = AgentVideoRecorder(
                self._video_recorder,
                lambda: self._agent,
            )

        self._observation_subscriber = ObservationSubscriberNode(
            self._agent.shared_state, on_episode_done=self._handle_episode_done
        )
        self._ros_executor = MultiThreadedExecutor(num_threads=4)
        self._ros_executor.add_node(self._observation_subscriber)
        self._ros_spin_thread = threading.Thread(
            target=self._ros_executor.spin, daemon=True, name="obs_subscriber_spin"
        )
        self._ros_spin_thread.start()

        if self._args.record_decisions:
            from rtnav.tools.visualization.decision_step_recorder import DecisionStepRecorder

            self._decision_recorder = DecisionStepRecorder(
                self._agent.shared_state,
                enabled=True,
            )
            with self._agent.shared_state.lock:
                self._agent.shared_state.system.decision_recorder = self._decision_recorder

        if self._args.rr_viz:
            from rtnav.tools.visualization.scene_graph_detection_visualizer_thread import (
                SceneGraphRerunVisualizer,
            )

            rr_viz = SceneGraphRerunVisualizer(self._agent.shared_state, self._agent.shutdown_event)
            rr_viz.start()
            self._visualizer_threads.append(rr_viz)
            print("[eval] Rerun viz started — run `rerun` on host first")

        if self._args.det_viz_web:
            from rtnav.tools.visualization.detection_visualizer_thread import (
                WebDetectionVisualizerThread,
            )

            det_viz = WebDetectionVisualizerThread(
                self._agent.shared_state, self._agent.shutdown_event
            )
            det_viz.start()
            self._visualizer_threads.append(det_viz)
            print("[eval] detection web viz — http://localhost:8765")

        if self._args.map_viz_web:
            from rtnav.tools.obs_viz import start_persistent_viz

            start_persistent_viz(
                self._args,
                self._agent,
                self._agent.shutdown_event,
                self._visualizer_threads,
            )

    def _start_episode(self):
        self._observation_subscriber.pause()

        try:
            response = self._call_reset()
            if response is None:
                print("[eval] reset returned None — aborting")
                return None

            self._episodes_remaining = response.episodes_remaining

            output_dir = self._episode_output_dir(response)
            self._agent.reset_episode(
                episode_hash=int(response.episode_id_hash),
                goal_info=json.loads(response.observation.info_json),
                scene_id=response.scene_id,
                episode_id=response.episode_id,
                output_dir=output_dir,
            )
            self._episode_done.clear()
            self._prepare_episode_outputs(response, output_dir)
            return response

        finally:
            self._observation_subscriber.resume()

    def _episode_output_dir(self, response) -> str:
        if self._run_dir is None:
            self._run_dir = Path(resolve_agent_run_dir())
            print(f"[eval] run dir: {self._run_dir}")
        scene_tag = Path(response.scene_id).stem
        episode_dir = self._run_dir / f"episode_{scene_tag}_{response.episode_id}" / "agent"
        episode_dir.mkdir(parents=True, exist_ok=True)
        return str(episode_dir)

    def _prepare_episode_outputs(self, response, output_dir: str) -> None:
        output_users = [viz for viz in self._visualizer_threads if hasattr(viz, "new_episode")]
        if self._decision_recorder is not None:
            output_users.append(self._decision_recorder)

        for output_user in output_users:
            output_user.new_episode(
                response.scene_id,
                response.episode_id,
                output_dir=output_dir,
            )

        if self._video is not None:
            self._video.new_episode(response.scene_id, response.episode_id)

    def _wait_for_episode_end(self, episode_index: int) -> None:
        # Block until the env sends DONE_SENTINEL, which ObservationSubscriberNode detects and delivers to _handle_episode_done.
        self._episode_done.wait()
        print(f"[eval] [ep {episode_index}] done ✓")

    def _save_episode(self) -> None:
        for viz in self._visualizer_threads:
            if hasattr(viz, "save_episode"):
                viz.save_episode()
        if self._video is not None:
            self._video.save()
        self._vlm_logger.save_episode()

    def _shutdown(self) -> None:
        # Stop ROS input before shutting down the RTNav workers.
        self._observation_subscriber.pause()
        self._ros_executor.shutdown()
        self._ros_spin_thread.join(timeout=3.0)
        self._observation_subscriber.destroy_node()
        self._agent.shutdown(timeout=3.0)
        if self._video is not None:
            self._video.shutdown()
        if self._decision_recorder is not None:
            self._decision_recorder.shutdown()
        if self._vlm_logger is not None:
            self._vlm_logger.shutdown()

        for thread in self._visualizer_threads:
            thread.join(timeout=3.0)

    def _handle_episode_done(self, _msg: ObservationMsg) -> None:
        """DONE_SENTINEL from ObservationSubscriberNode — episode has ended."""
        self._episode_done.set()

    def _on_obs_video(self, msg: ObservationMsg) -> None:
        if self._video is not None:
            self._video.on_obs(msg)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--map_viz_web", action="store_true", help="VLFM obstacle-map viz on :8766")
    p.add_argument("--det_viz_web", action="store_true", help="detector viz on :8765")
    p.add_argument(
        "--rr_viz", action="store_true", help="rerun scene-graph + 3D point cloud visualizer"
    )
    p.add_argument("--record_decisions", action="store_true", help="save decision filmstrips")
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    print(f"[eval] RTNav running on benchmark={os.environ['BENCHMARK']}")

    run_agent(lambda mode: RTNavEvalNode(args, mode=mode))


if __name__ == "__main__":
    main()
