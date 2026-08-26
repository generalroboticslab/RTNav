"""In-process MobileSAM wrapper for RTNav target mask projection."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _clip_bbox(bbox: list[float], image_shape: tuple[int, int]) -> list[int]:
    h, w = image_shape
    if len(bbox) != 4:
        raise ValueError(f"[MobileSAM] bbox must have four values, got {bbox!r}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    clipped = [
        max(0, min(w - 1, int(round(left)))),
        max(0, min(h - 1, int(round(top)))),
        max(0, min(w - 1, int(round(right)))),
        max(0, min(h - 1, int(round(bottom)))),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"[MobileSAM] bbox is empty after clipping: {bbox!r}")
    return clipped


class MobileSAM:
    def __init__(
        self,
        sam_checkpoint: str | None = None,
        model_type: str = "vit_t",
        device: Any | None = None,
    ) -> None:
        import torch
        from mobile_sam import SamPredictor, sam_model_registry

        if sam_checkpoint is None:
            sam_checkpoint = os.environ.get(
                "MOBILE_SAM_CHECKPOINT",
                "data/mobile_sam.pt",
            )
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"[MobileSAM] checkpoint not found: {sam_checkpoint}")
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        model.to(device=device)
        model.eval()
        self.predictor = SamPredictor(model)

    def segment_bbox(self, image: np.ndarray, bbox: list[float]) -> np.ndarray:
        import torch

        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"[MobileSAM] image must be HxWxC, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        clipped_bbox = _clip_bbox(bbox, image.shape[:2])
        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(
                box=np.array(clipped_bbox),
                multimask_output=False,
            )
        return masks[0]
