"""The central data bus and its per-module state containers.

Every module reads and writes through the grouped containers on a single
SharedState instance. Each container owns a reset() that decides what is
per-episode vs. persistent; SharedState.reset_episode() calls them all.

Value objects passed between modules (observations, frames, outputs) live in
data_types.py — this file holds only the mutable shared state.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from rtnav.core.data_types import CameraFrame, HabitatObservation
from rtnav.core.inference_gate import InferenceGate


# ── Sensor ───────────────────────────────────────────────────────────────────
@dataclass
class SensorState:
    """Shared sensor bus: the raw Habitat observation plus the pose/camera
    signals PerceptionThread derives from it."""

    # Raw observation: written by ObservationSubscriberNode, read by PerceptionThread,
    # the VLFM frontier detector, and the viz/recorder threads.
    habitat_obs: Optional[HabitatObservation] = None

    # Derived by PerceptionThread; read by frontier/detector/viz.
    latest_cameras: Dict[str, CameraFrame] = field(default_factory=dict)
    camera_history: List[Dict[str, CameraFrame]] = field(default_factory=list)
    latest_odom: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)

    def reset(self):
        self.habitat_obs = None
        self.latest_cameras = {}
        self.camera_history = []
        self.latest_odom = None


# ── Perception ───────────────────────────────────────────────────────────────
@dataclass
class PerceptionState:
    """Perception + detection outputs."""

    perception_output: Any = None  # PerceptionOutput, set by PerceptionThread
    perception_version: int = 0  # bumped per publish; MappingThread watches it
    detection_result: Any = None  # MultiCameraDetectionResult from DetectorThread

    def reset(self):
        self.perception_output = None
        self.perception_version = 0
        self.detection_result = None


# ── Mapping ──────────────────────────────────────────────────────────────────
@dataclass
class MappingState:
    """Mapping module outputs."""

    obstacle_map: Any = None  # ObstacleMap instance
    mapping_output: Any = None  # MappingOutput instance

    def reset(self):
        self.mapping_output = None


# ── Frontier ─────────────────────────────────────────────────────────────────
@dataclass
class FrontierState:
    """Frontier detection and selection."""

    frontier_output: Any = None  # FrontierOutput instance
    frontier_output_cameras: Dict[str, CameraFrame] = field(default_factory=dict)
    frontier_output_camera_history: List[Dict[str, CameraFrame]] = field(default_factory=list)
    chosen_frontier_xy: Optional[Tuple[float, float]] = None
    # Image-VLM frontier selector telemetry.
    frontier_images: Any = None
    frontier_vlm_query_text: str = ""
    frontier_vlm_query_time: float = 0.0

    def reset(self):
        self.frontier_output = None
        self.frontier_output_cameras = {}
        self.frontier_output_camera_history = []
        self.chosen_frontier_xy = None
        self.frontier_images = None
        self.frontier_vlm_query_text = ""
        self.frontier_vlm_query_time = 0.0


# ── Scene graph ──────────────────────────────────────────────────────────────
@dataclass
class SceneGraphState:
    """Scene graph module outputs."""

    scene_graph: Any = None  # SceneGraph instance
    # VLM-confirmed detections queued for SG insertion (continuous verification).
    pending_verified_detections: List[Dict] = field(default_factory=list)

    def reset(self):
        self.scene_graph = None
        self.pending_verified_detections = []


# ── Navigation ───────────────────────────────────────────────────────────────
@dataclass
class NavigationState:
    """Navigation state. The PointNav policy drives toward goal_xy; classical
    path/waypoint planning was removed."""

    status: str = "idle"  # idle | navigating | reached | failed
    goal_xy: Optional[Tuple[float, float]] = None
    goal_px: Optional[Tuple[int, int]] = None
    goal_cluster: Optional[np.ndarray] = None
    goal_source: Optional[str] = None  # target | frontier
    nav_id: int = 0
    failure_reason: Optional[str] = None

    def reset(self):
        self.status = "idle"
        self.goal_xy = self.goal_px = self.goal_cluster = self.goal_source = self.failure_reason = (
            None
        )


# ── Target ───────────────────────────────────────────────────────────────────
@dataclass
class TargetState:
    """Target tracking and pursuit."""

    current_target_xy: Optional[Tuple[float, float]] = None
    current_target_label: Optional[str] = None
    # SG node id of the currently-chosen find target, for resolving live centroid.
    current_target_node_id: Optional[int] = None
    target_goals: List[Dict] = field(default_factory=list)
    # Last successfully visited temporary target: (approach origin, target).
    last_temporary_target_ray: Optional[
        Tuple[Tuple[float, float], Tuple[float, float]]
    ] = None
    # SG node ids permanently ignored after 3 VLM scores <= 0.5.
    target_node_blacklist_ids: Set[int] = field(default_factory=set)

    def reset(self):
        self.current_target_xy = self.current_target_label = self.current_target_node_id = None
        self.target_goals = []
        self.last_temporary_target_ray = None
        self.target_node_blacklist_ids.clear()


# ── Task ─────────────────────────────────────────────────────────────────────
@dataclass
class TaskState:
    """The one target assigned to the current ObjectNav episode."""

    # synonym.lower() -> canonical name, populated by GoalParser each episode.
    synonym_to_canonical: Dict[str, str] = field(default_factory=dict)
    goal_category: str = ""
    complete: bool = False

    def reset(self):
        self.goal_category = ""
        self.synonym_to_canonical = {}
        self.complete = False


# ── System & config ──────────────────────────────────────────────────────────
@dataclass
class SystemState:
    """System control state."""

    robot_fallen: bool = False
    episode_index: int = (
        0  # Incremented by --per-episode-reset; threads reinit when they see a new value
    )
    # Incremented by decision on frontier exhaustion; mapping/frontier threads
    # wipe their explored state when they see a new value.
    exploration_wipe_counter: int = 0
    # True once the nav bootstrap scan completes; gates decision reasoning.
    bootstrap_spin_complete: bool = True

    # Optional per-decision filmstrip recorder (DecisionStepRecorder).
    decision_recorder: Any = None
    # Always-on visual record of frontier and target VLM calls.
    vlm_logger: Any = None
    # Startup service handles: set once by the agent, persist across episodes.
    robot_api: Any = None  # RobotAPI, for publishing /agent_done

    def reset(self):
        self.episode_index += 1
        self.robot_fallen = False
        self.bootstrap_spin_complete = False


class SharedState:
    """Central data bus for all modules.

    Access pattern:
        shared_state.nav.status = "navigating"
        shared_state.nav.goal_xy = (1.0, 2.0)
        shared_state.task.goal_category = "chair"
    """

    def __init__(self):
        self.lock = threading.Lock()
        # Set after GoalParser applies a goal (synonyms resolved,
        # goal_category populated). Movement / planning / reasoning gates
        # on this so we don't move or plan before the task is parsed.
        # Cleared at every episode boundary.
        self.task_ready = threading.Event()

        # Module state containers
        self.sensor = SensorState()
        self.perception = PerceptionState()
        self.mapping = MappingState()
        self.frontier = FrontierState()
        self.scenegraph = SceneGraphState()
        self.nav = NavigationState()
        self.target = TargetState()
        self.task = TaskState()
        self.system = SystemState()
        self.inference = InferenceGate()

    def reset_episode(self):
        """Reset every container's per-episode state under the lock.

        Each container decides which runtime handles persist across episodes.
        """
        with self.lock:
            for state in (
                self.system,
                self.sensor,
                self.perception,
                self.mapping,
                self.scenegraph,
                self.frontier,
                self.nav,
                self.target,
                self.task,
            ):
                state.reset()
