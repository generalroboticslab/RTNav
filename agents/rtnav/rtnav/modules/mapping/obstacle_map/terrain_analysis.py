import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


class TerrainAnalyzer:
    """Slope-based traversability: cells whose smoothed-height gradient exceeds
    slope_threshold are marked as obstacles."""

    def __init__(
        self, grid_size: float, slope_threshold: float = 0.3, erode_boundary: bool = False
    ):
        self.grid_size = grid_size
        self.slope_threshold = slope_threshold
        self.erode_boundary = erode_boundary

    def analyze(self, height_map) -> np.ndarray:
        """Return an obstacle mask (1.0 = obstacle) shaped like height_map."""
        valid = np.isfinite(height_map)
        h = height_map.astype(np.float32, copy=True)

        # Nearest-neighbor fill so slope across the valid/invalid border stays ~0.
        if (~valid).any() and valid.any():
            from scipy.ndimage import distance_transform_edt

            _, nearest = distance_transform_edt(~valid, return_indices=True)
            h[~valid] = h[nearest[0][~valid], nearest[1][~valid]]

        # Smooth before slope so per-pixel sensor jitter doesn't read as obstacles.
        slope = self._compute_slope(gaussian_filter(h, sigma=0.5))

        # Restrict to the eroded interior of valid cells, dropping noisy borders.
        interior = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if interior.sum() == 0:
            interior = valid
        if self.erode_boundary:
            interior = cv2.erode(interior.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if interior.sum() == 0:
                interior = valid
        slope[~interior] = 0

        obstacle = (slope > self.slope_threshold).astype(np.float32)
        obstacle[~valid] = 0
        return obstacle

    def _compute_slope(self, height_map) -> np.ndarray:
        dzdx = sobel(height_map, axis=1) / (8 * self.grid_size)
        dzdy = sobel(height_map, axis=0) / (8 * self.grid_size)
        return np.sqrt(dzdx**2 + dzdy**2)
