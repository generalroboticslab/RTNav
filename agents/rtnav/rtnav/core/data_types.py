"""Value objects passed between modules (observations, frames, outputs).

The mutable shared-bus state containers (SensorState, PerceptionState, …)
live in shared_state.py, not here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Sensors & raw observations ───────────────────────────────────────────────
@dataclass
class HabitatObservation:
    """Raw /obs from env_node — sim outputs before any geometry is computed."""

    step_id: int
    rgb: Optional[np.ndarray]  # HxWx3 uint8
    depth: np.ndarray  # HxW float32, meters
    gps: Tuple[float, float]  # (x, y) displacement from episode start, meters
    compass: float  # radians, 0 = forward (+Y world), CW positive
    timestamp: float = 0.0


@dataclass
class CameraFrame:
    """Single camera's RGB-D frame (geometry resolved: T_world_cam + intrinsics)."""

    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    T_world_cam: np.ndarray
    intrinsics: np.ndarray
    # Per-camera valid depth range (meters) for rejecting depth sentinel values.
    min_depth_m: float = 0.0
    max_depth_m: float = 10.0


# ── Perception & detection ───────────────────────────────────────────────────
@dataclass
class PerceptionOutput:
    """One perception frame for mapping: world point cloud + robot pose +
    per-camera extrinsics/FOV metadata."""

    points_world: np.ndarray
    camera_extrinsics: Dict[str, np.ndarray]
    robot_pose: np.ndarray
    camera_meta: Dict[str, Dict[str, float]]


@dataclass
class DetectionEntity:
    """Single detected object."""

    name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    embeddings: Optional[np.ndarray] = None
    # Per-detection top-k (label, probability) from the prompt ensemble.
    top_k_label_probs: Optional[List[Tuple[str, float]]] = None


@dataclass
class DetectionOutput:
    """Full detector output."""

    detections: List[DetectionEntity]


@dataclass
class CameraDetectionResult:
    """Detection results for a single camera."""

    camera_name: str
    timestamp: float
    detections: List[DetectionEntity]
    rgb_image: np.ndarray
    depth_image: np.ndarray
    T_world_camera: np.ndarray
    # Depth camera intrinsics (for 2D->3D back-projection; matches depth_image resolution)
    intrinsics_depth: np.ndarray
    # RGB camera intrinsics when different from depth (optional)
    intrinsics_rgb: Optional[np.ndarray] = None
    min_depth_m: float = 0.0
    max_depth_m: float = 10.0
    # Rotation applied to RGB (0=none, 1=90CCW, -1=90CW, 2=180)
    rotation_k: int = 0
    # Exact image fed to detector (rotated/masked RGB); bboxes in detections_detector_input
    rgb_image_detector_input: Optional[np.ndarray] = None
    detections_detector_input: Optional[List[DetectionEntity]] = None

    @property
    def intrinsics(self) -> np.ndarray:
        """The K used for 2D->3D back-projection (depth-frame intrinsics)."""
        return self.intrinsics_depth


@dataclass
class MultiCameraDetectionResult:
    """Detection results from all cameras."""

    timestamp: float
    camera_results: Dict[str, CameraDetectionResult]
    episode_index: int = 0


# ── Mapping ──────────────────────────────────────────────────────────────────
@dataclass
class MappingOutput:
    """2D map for navigation."""

    timestamp: float = 0.0


# ── Frontier ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FrontierMapSnapshot:
    """Copied VLFM explored-map state used to produce one frontier output."""

    explored: np.ndarray
    pixels_per_meter: float
    episode_pixel_origin: Tuple[float, float]


@dataclass
class FrontierOutput:
    """Frontier detection output."""

    frontier_mask: np.ndarray
    frontier_clusters: list
    frontier_centroids: np.ndarray
    frontier_unexplored_directions: list = field(default_factory=list)
    vlfm_map_snapshot: Optional[FrontierMapSnapshot] = None
    timestamp: float = 0.0


# ── Scene graph ──────────────────────────────────────────────────────────────
@dataclass
class SceneGraphNode:
    """Tracked object in scene graph."""

    node_id: int
    past_labels: list  # list of label strings with repetition (frequency-preserving)
    chosen_label: str
    confidence: float
    centroid: np.ndarray
    bbox_3d: np.ndarray
    point_cloud: np.ndarray
    colors: np.ndarray
    view_count: int
    is_confirmed: bool
    # World (x, y) the camera was at when this node was last detected.
    observed_from_xy: Optional[Tuple[float, float]] = None
    # Uncapped lifetime label tally (label → count), unlike windowed past_labels.
    label_counts: Dict[str, int] = field(default_factory=dict)
    label_history: List[Tuple] = field(default_factory=list)
    label_prob_sums: Dict[str, float] = field(default_factory=dict)
    label_prob_count: int = 0


@dataclass
class SceneGraphOutput:
    """Complete scene graph output."""

    nodes: List[SceneGraphNode]
    timestamp: float
