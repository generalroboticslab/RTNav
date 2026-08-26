# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
except ModuleNotFoundError:
    print("Could not import transformers/Owlv2. This is OK if you are only using the client.")

OWLV2_MODEL_PATH = "data/owlv2/owlv2-base-patch16-ensemble"

_VIS_FRAME_COUNTER = 0
_VIS_OUTPUT_DIR = os.environ.get("OWLV2_VIS_DIR", "owlv2_vis_output")


def _parse_caption_to_texts(caption: str) -> List[str]:
    classes = [c.strip() for c in caption.replace(" .", ".").split(".") if c.strip()]
    return classes


def _classes_from_caption(caption: Optional[str]) -> List[str]:
    if caption is None or caption.strip() == "":
        raise ValueError("OWLv2 detection requires a non-empty caption.")
    classes = _parse_caption_to_texts(caption)
    if len(classes) == 0:
        raise ValueError(f"OWLv2 caption produced no classes: {caption!r}")
    return classes


def _save_owlv2_visualization(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    phrases: List[str],
    save_dir: str = _VIS_OUTPUT_DIR,
) -> None:
    global _VIS_FRAME_COUNTER

    try:
        os.makedirs(save_dir, exist_ok=True)
        pil_img = Image.fromarray(image).copy()
        draw = ImageDraw.Draw(pil_img)

        boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
        scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)

        for i in range(len(boxes_np)):
            x1, y1, x2, y2 = boxes_np[i]
            conf = float(scores_np[i])
            label = phrases[i] if i < len(phrases) else "?"
            color = "red" if conf < 0.3 else "orange" if conf < 0.6 else "lime"
            draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=color, width=3)
            draw.text((float(x1), float(y1) - 14), f"{label} ({conf:.2f})", fill=color)

        save_path = os.path.join(save_dir, f"owlv2_frame_{_VIS_FRAME_COUNTER:06d}.png")
        pil_img.save(save_path)
        print(f"[OWLv2 Vis] Saved {save_path} ({len(boxes_np)} detections)")
    except Exception as e:
        print(f"[OWLv2 Vis] WARNING: visualization failed: {e}")
    finally:
        _VIS_FRAME_COUNTER += 1


class OWLv2:
    def __init__(
        self,
        model_path: str = OWLV2_MODEL_PATH,
        caption: Optional[str] = None,
        box_threshold: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.processor = Owlv2Processor.from_pretrained(model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(device)
        self.model.eval()
        self.caption = caption
        self.box_threshold = box_threshold
        self.device = device

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        classes = _classes_from_caption(caption or self.caption)
        print(f"OWLv2 input classes: {classes}")

        inputs = self.processor(text=[classes], images=Image.fromarray(image), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(outputs=outputs, threshold=self.box_threshold)
        result = results[0]
        boxes = result["boxes"].cpu()
        scores = result["scores"].cpu()
        labels = result["labels"].cpu()

        h, w = image.shape[:2]
        max_dim = max(h, w)
        boxes[:, 0] = (boxes[:, 0] * max_dim).clamp(0, w) / w
        boxes[:, 1] = (boxes[:, 1] * max_dim).clamp(0, h) / h
        boxes[:, 2] = (boxes[:, 2] * max_dim).clamp(0, w) / w
        boxes[:, 3] = (boxes[:, 3] * max_dim).clamp(0, h) / h

        phrases = [classes[label.item()] for label in labels]
        detections = ObjectDetections(
            boxes=boxes,
            logits=scores,
            phrases=phrases,
            image_source=image,
            fmt="xyxy",
        )
        detections.filter_by_class(classes)
        return detections


class OWLv2Client:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/owlv2"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        _classes_from_caption(caption)
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading OWLv2 model...")

    class OWLv2Server(ServerMixin, OWLv2):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image, caption=payload["caption"]).to_json()

    owlv2 = OWLv2Server()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owlv2, name="owlv2", port=args.port)
