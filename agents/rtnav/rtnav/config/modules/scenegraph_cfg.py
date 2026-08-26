from rtnav.config import configclass


@configclass
class SceneGraphConfig:
    # Confirmation.
    min_confirmation_views: int = 3  # distinct-view detections before a node is confirmed
    confirm_min_separation_m: float = 0.1  # min camera-move (m) for two views to count as distinct

    # Stale-node pruning.
    remove_stale_objects: bool = False
    max_frames_without_detection: int = 5

    # Voxel grid + merge.
    voxel_size_m: float = 0.05
    merge_max_distance_m: float = 3.0  # max distance between same-label centroids to merge
