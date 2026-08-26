"""Object-centric 3D map built incrementally from RGB-D detections."""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np


class VoxelGrid:
    """Incremental voxel grid for point cloud dedup. Dict-backed (O(1) insert);
    caches exported arrays for repeated access."""

    def __init__(self, voxel_size=0.05, max_voxels=2000):
        self.voxel_size = voxel_size
        self.voxels = {}  # Dict[voxel_key, (point, color)]
        self._max_voxels = max_voxels
        self._cache_valid = False
        self._cached_points = None
        self._cached_colors = None
        self._cached_keys_np = None

    def add_points(self, points, colors):
        """Add points; last write wins per voxel."""
        if len(points) == 0:
            return
        voxel_keys = (points / self.voxel_size).astype(np.int32)
        key_tuples = list(map(tuple, voxel_keys.tolist()))
        pts_list = points.tolist()
        col_list = colors.tolist()
        self.voxels.update(zip(key_tuples, zip(pts_list, col_list)))
        if len(self.voxels) > self._max_voxels:
            excess = len(self.voxels) - self._max_voxels
            for k in list(self.voxels.keys())[:excess]:
                del self.voxels[k]
        self._cache_valid = False
        self._cached_keys_np = None

    def get_points(self):
        """Export all voxel points as arrays. Uses cache for repeated access."""
        if self._cache_valid and self._cached_points is not None:
            return self._cached_points, self._cached_colors

        if len(self.voxels) == 0:
            self._cached_points = np.zeros((0, 3))
            self._cached_colors = np.zeros((0, 3), dtype=np.uint8)
        else:
            self._cached_points = np.array([v[0] for v in self.voxels.values()])
            self._cached_colors = np.array([v[1] for v in self.voxels.values()])

        self._cache_valid = True
        return self._cached_points, self._cached_colors

    def __len__(self):
        return len(self.voxels)

    def get_keys_np(self):
        """Get cached numpy array of voxel keys (rebuilt only when dirty)."""
        if self._cached_keys_np is None:
            if len(self.voxels) > 0:
                self._cached_keys_np = np.array(list(self.voxels.keys()), dtype=np.int32)
            else:
                self._cached_keys_np = np.zeros((0, 3), dtype=np.int32)
        return self._cached_keys_np


@dataclass
class ObjectNode:
    """One object in the 3D scene graph (geometry + OWLv2 features + history)."""

    node_id: int
    label: str = ""

    _commit_window: ClassVar[int] = 12
    _switch_margin: ClassVar[int] = 2

    # A detection counts as a DISTINCT vantage unless its camera POSE is nearly identical
    # to one already recorded — i.e. within BOTH _confirm_min_separation_m AND
    # _confirm_min_angle_rad. Only true duplicate poses (holding still, re-detecting the
    # same frame) collapse; any real movement or turn is a new view. This stops a stuck
    # re-detection from re-confirming and spamming the VLM verify while still crediting
    # genuinely different perspectives.
    _confirm_min_separation_m: ClassVar[float] = 0.1
    _confirm_min_angle_rad: ClassVar[float] = 0.05  # ~3°

    bbox_3d: Optional[np.ndarray] = None  # (2, 3): [min_xyz, max_xyz]
    centroid: Optional[np.ndarray] = None  # (3,)
    point_cloud: Optional[np.ndarray] = None  # (N, 3)
    colors: Optional[np.ndarray] = None  # (N, 3)
    owl_features: Optional[np.ndarray] = None
    confidence: float = 0.0
    view_count: int = 0  # DISTINCT vantages (not raw frames) — gates confirm/target/reject
    _raw_obs: int = field(default=0, init=False, repr=False)  # raw frame count, for averaging
    first_seen: int = 0
    last_seen: int = 0
    label_history: List[Tuple] = field(
        default_factory=list
    )  # (label, confidence, frame_idx, top_k_label_probs)
    frames_since_detection: int = 0
    is_confirmed: bool = False
    voting_window: int = 5
    number_to_be_confirmed: int = 2
    all_labels: Set[str] = field(default_factory=set)
    # Lifetime label tally, never truncated; powers the dominance guard.
    label_counts: Dict[str, int] = field(default_factory=dict)
    label_prob_sums: Dict[str, float] = field(default_factory=dict)
    label_prob_count: int = 0
    # Closest distance this node was ever detected at; biases centroid updates.
    _best_detection_dist: float = field(default=float("inf"), init=False, repr=False)
    # Camera world (x, y) at the most recent detection of this node.
    observed_from_xy: Optional[Tuple[float, float]] = None
    # Distinct camera world-xy positions this object was detected from.
    observed_locations: List[Tuple[float, float, float]] = field(
        default_factory=list
    )  # (x, y, yaw)

    _voxel_grid: Optional[VoxelGrid] = field(default=None, init=False, repr=False)

    def update_from_detection(self, detection: "ObjectDetection", frame_idx: int):
        """Fold a new detection into this node (centroid, label, features, bbox)."""
        self._raw_obs += 1  # raw frames folded into the running centroid average
        self.last_seen = frame_idx
        self.frames_since_detection = 0
        # view_count counts DISTINCT vantages: only bump it when this detection is from a
        # camera position ≥ _confirm_min_separation_m from every one already seen. So a
        # stationary/churned re-detection of the same object doesn't rack up views — which
        # otherwise lets it re-confirm and re-trigger the target/VLM verify without moving.
        obs_xy = detection.observed_from_xy
        obs_yaw = detection.observed_from_yaw
        if obs_xy is not None:
            self.observed_from_xy = obs_xy
            if self._register_location(obs_xy, obs_yaw):
                self.view_count += 1
        else:
            self.view_count += 1  # no camera pose → can't dedupe, count the frame

        if self.view_count >= self.number_to_be_confirmed:
            self.is_confirmed = True

        if detection.label:
            top_k = list(detection.top_k_label_probs or [])
            self.label_history.append((detection.label, detection.confidence, frame_idx, top_k))
            if len(self.label_history) > 50:
                self.label_history = self.label_history[-50:]
            self.all_labels.add(detection.label)
            self.label_counts[detection.label] = self.label_counts.get(detection.label, 0) + 1
            self.label_prob_count += 1
            for label, probability in top_k:
                self.label_prob_sums[label] = self.label_prob_sums.get(label, 0.0) + float(
                    probability
                )

        self.label = self.get_consensus_label()

        # Confidence EMA.
        alpha = 0.3
        self.confidence = alpha * detection.confidence + (1 - alpha) * self.confidence

        # 3D bounding box union.
        if self.bbox_3d is None:
            self.bbox_3d = detection.bbox_3d.copy()
        else:
            self.bbox_3d[0] = np.minimum(self.bbox_3d[0], detection.bbox_3d[0])
            self.bbox_3d[1] = np.maximum(self.bbox_3d[1], detection.bbox_3d[1])

        # Distance-biased centroid: a much closer detection replaces rather than
        # averages, since depth accuracy degrades with distance.
        det_dist = float(detection.robot_distance)
        if self.centroid is None:
            self.centroid = detection.centroid.copy()
            self._best_detection_dist = det_dist
        elif det_dist < self._best_detection_dist * 0.75:
            self.centroid = detection.centroid.copy()
            self._best_detection_dist = det_dist
        else:
            self.centroid = (
                self.centroid * (self._raw_obs - 1) + detection.centroid
            ) / self._raw_obs
            if det_dist < self._best_detection_dist:
                self._best_detection_dist = det_dist

        # Feature EMA.
        if self.owl_features is None:
            self.owl_features = detection.owl_features.copy()
        else:
            self.owl_features = alpha * detection.owl_features + (1 - alpha) * self.owl_features
            self.owl_features /= np.linalg.norm(self.owl_features) + 1e-8

        if self._voxel_grid is None:
            self._voxel_grid = VoxelGrid(voxel_size=0.05)

        self._voxel_grid.add_points(detection.point_cloud, detection.colors)

        # Export points lazily; later reads go through get_points() on the grid.
        if self.point_cloud is None or len(self.point_cloud) == 0:
            self.point_cloud, self.colors = self._voxel_grid.get_points()

    def _register_location(self, xy, yaw=None) -> bool:
        """Record a camera pose (x, y, yaw), but only if it differs from every pose already
        recorded — position ≥ _confirm_min_separation_m OR heading ≥ _confirm_min_angle_rad.
        So a detection collapses to an existing vantage only when it's close in BOTH
        position and heading; moving OR turning to a new angle counts as a new vantage.
        Returns True iff a new vantage was added."""
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, IndexError, ValueError):
            return False
        yaw = float(yaw) if yaw is not None else None
        sep2 = self._confirm_min_separation_m * self._confirm_min_separation_m
        for loc in self.observed_locations:
            px, py = loc[0], loc[1]
            pyaw = loc[2] if len(loc) > 2 else None
            if (px - x) ** 2 + (py - y) ** 2 >= sep2:
                continue  # far enough in position → distinct from this one
            if yaw is not None and pyaw is not None:
                dyaw = abs((yaw - pyaw + np.pi) % (2 * np.pi) - np.pi)
                if dyaw >= self._confirm_min_angle_rad:
                    continue  # same spot but turned enough → distinct from this one
            return False  # close in position AND heading → same vantage
        self.observed_locations.append((x, y, yaw if yaw is not None else 0.0))
        return True

    def get_consensus_label(self) -> str:
        """
        Consensus label, biased hard toward stability once the node is confirmed.

        - UNCONFIRMED node: responsive — returns the highest cumulative-confidence
          label over the recent ``voting_window`` views so a new object's label
          converges quickly before it's committed.
        - CONFIRMED node: stable — votes over the last 12 labels and switches
          only when another label leads the current label by at least two votes.
        """
        if not self.label_history:
            return ""

        recent_labels = self.label_history[-self.voting_window :]

        label_scores: Dict[str, float] = {}
        for entry in recent_labels:
            label, confidence = entry[:2]
            if label not in label_scores:
                label_scores[label] = 0.0
            label_scores[label] += confidence

        if not label_scores:
            return ""

        best_label = max(label_scores.items(), key=lambda x: x[1])[0]

        # Confirmed node: commit to the majority label over the recent
        # _commit_window, switching only when another leads by _switch_margin.
        if self.is_confirmed:
            window = self.label_history[-self._commit_window :]
            counts: Dict[str, int] = {}
            for entry in window:
                lbl = entry[0]
                counts[lbl] = counts.get(lbl, 0) + 1
            if not counts:
                return self.label or best_label
            dom_label, dom_n = max(counts.items(), key=lambda kv: kv[1])
            if not self.label or dom_label == self.label:
                return self.label or dom_label
            cur_n = counts.get(self.label, 0)
            if dom_n >= cur_n + self._switch_margin:
                return dom_label
            return self.label

        return best_label

    def get_voxel_grid(self) -> Optional[VoxelGrid]:
        """The node's voxel grid (None until it has points)."""
        return self._voxel_grid


@dataclass
class ObjectDetection:
    """One object detection from a single frame."""

    label: str
    bbox_3d: np.ndarray
    centroid: np.ndarray
    point_cloud: np.ndarray
    colors: np.ndarray
    owl_features: np.ndarray
    confidence: float
    robot_distance: float = float("inf")  # camera-to-centroid distance at detection
    # Camera world (x, y) and viewing heading (yaw, rad) this detection came from.
    observed_from_xy: Optional[Tuple[float, float]] = None
    observed_from_yaw: Optional[float] = None
    top_k_label_probs: Optional[List[Tuple[str, float]]] = None


class SceneGraph:
    """Object-centric 3D map of tracked object nodes."""

    def __init__(self):
        self.nodes: Dict[int, ObjectNode] = {}
        self._node_counter: int = 0
        self._version: int = 0  # bumped on structure change, for caching
        self._added_nodes: List[int] = []
        self._removed_nodes: set = set()

    def add_node(self, node: ObjectNode) -> int:
        """Add a node; assign and return its unique id."""
        if node.node_id == 0:  # not assigned yet
            node.node_id = self._node_counter
            self._node_counter += 1

        self.nodes[node.node_id] = node
        self._version += 1
        self._added_nodes.append(node.node_id)
        return node.node_id

    def update_node(self, node_id: int, detection: ObjectDetection, frame_idx: int):
        """Fold a new detection into an existing node (raises if unknown)."""
        if node_id in self.nodes:
            self.nodes[node_id].update_from_detection(detection, frame_idx)
        else:
            raise ValueError(f"Node {node_id} not found in scene graph")

    def remove_node(self, node_id: int):
        """Remove a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._version += 1
            self._removed_nodes.add(node_id)
            if node_id in self._added_nodes:
                self._added_nodes.remove(node_id)

    def get_statistics(self) -> Dict:
        """Get scene graph statistics."""
        return {
            "num_nodes": len(self.nodes),
            "total_points": sum(
                len(n.point_cloud) if n.point_cloud is not None else 0 for n in self.nodes.values()
            ),
            "avg_confidence": (
                np.mean([n.confidence for n in self.nodes.values()]) if self.nodes else 0.0
            ),
            "avg_view_count": (
                np.mean([n.view_count for n in self.nodes.values()]) if self.nodes else 0.0
            ),
        }

    def __len__(self) -> int:
        """Number of nodes in graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"SceneGraph(nodes={stats['num_nodes']}, avg_confidence={stats['avg_confidence']:.2f})"
        )
