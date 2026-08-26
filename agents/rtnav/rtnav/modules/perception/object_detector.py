"""OWLv2 detector with prompt ensembling and duplicate suppression."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F_nn
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from rtnav.core.data_types import DetectionEntity, DetectionOutput
from rtnav.modules.perception.detection_policy import (
    MAX_BOX_AREA_FRACTION,
    NMS_IOU_THRESHOLD,
    PROMPT_TEMPLATES,
    TOP_K_LABELS,
    average_prompt_logits,
    canonical_top_k,
    prompt_queries,
    square_padding,
)


def resolve_owl_weights() -> str:
    here = Path(__file__).resolve().parent
    path = here / "weights" / "owlv2" / "base"
    if path.is_dir():
        return str(path)
    raise FileNotFoundError(f"OWLv2 weights not found: {path}")


class OpenVocabularyDetector:
    def __init__(
        self,
        owl_dir: str | None = None,
        inference_res: int = 960,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.inference_res = int(inference_res) if inference_res else 0
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        owl_dir = owl_dir or resolve_owl_weights()
        self.processor = Owlv2Processor.from_pretrained(owl_dir, use_fast=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            owl_dir,
            local_files_only=True,
            low_cpu_mem_usage=False,
        )
        meta_params = [
            name
            for name, param in self.model.named_parameters()
            if getattr(param, "is_meta", False)
        ]
        if meta_params:
            preview = ", ".join(meta_params[:5])
            raise RuntimeError(
                f"OWLv2 loaded with meta parameters despite low_cpu_mem_usage=False: {preview}"
            )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self._text_cache = OrderedDict()
        self._class_cache = OrderedDict()
        self._mean = torch.tensor(
            self.processor.image_processor.image_mean,
            device=self.device,
            dtype=self.dtype,
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            self.processor.image_processor.image_std,
            device=self.device,
            dtype=self.dtype,
        ).view(1, 3, 1, 1)
        self._image_size = self.inference_res or self._processor_image_size()

    def _processor_image_size(self) -> int:
        size = getattr(self.processor.image_processor, "size", None)
        if isinstance(size, dict):
            for key in ("height", "shortest_edge"):
                value = size.get(key)
                if isinstance(value, int):
                    return value
        return int(self.model.config.vision_config.image_size)

    @staticmethod
    def _as_rgb3_batch(image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise RuntimeError(
                f"Detector image batch must be BxCxHxW, got shape={getattr(image, 'shape', None)}"
            )
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] == 4:
            image = image[:, :3]
        elif image.shape[1] != 3:
            raise RuntimeError(
                f"Detector image batch must have 1, 3, or 4 channels, got shape={image.shape}"
            )

        if torch.is_floating_point(image):
            if image.numel() and image.max().item() > 1.1:
                image = image / 255.0
            return image.clamp(0.0, 1.0)
        return image.float().clamp(0, 255) / 255.0

    def _class_metadata(
        self,
        detection_classes: dict,
        batch_size: int,
    ) -> tuple[list[str], dict, list[list[str]], list[str], int]:
        valid_lookup = detection_classes.get("targets", {})
        additional = [
            label
            for label in dict.fromkeys(detection_classes.get("additional", []))
            if label not in valid_lookup
        ]
        cache_key = (
            PROMPT_TEMPLATES,
            tuple(valid_lookup.items()),
            tuple(additional),
            batch_size,
        )
        cached = self._class_cache.get(cache_key)
        if cached is not None:
            self._class_cache.move_to_end(cache_key)
            return cached

        texts = list(valid_lookup.keys()) + list(additional)
        if not texts:
            raise RuntimeError("OWLv2 detector requires at least one text query")
        prompt_texts = prompt_queries(texts)
        metadata = (
            texts,
            dict(valid_lookup),
            [texts for _ in range(batch_size)],
            prompt_texts,
            len(PROMPT_TEMPLATES),
        )
        self._class_cache[cache_key] = metadata
        self._class_cache.move_to_end(cache_key)
        while len(self._class_cache) > 8:
            self._class_cache.popitem(last=False)
        return metadata

    def _get_text_embeddings(self, texts: list[str], batch_size: int):
        cache_key = tuple(texts)
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            self._text_cache.move_to_end(cache_key)
            query_embeds, query_mask = cached
            return query_embeds.repeat(batch_size, 1, 1), query_mask.repeat(batch_size, 1)

        text_inputs = self.processor(
            text=[texts],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs.get("attention_mask")
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

        text_outputs = self.model.owlv2.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = self.model.owlv2.text_projection(text_outputs.pooler_output)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        max_queries = input_ids.shape[0]
        query_embeds = text_embeds.reshape(1, max_queries, text_embeds.shape[-1])
        input_ids_reshaped = input_ids.reshape(1, max_queries, input_ids.shape[-1])
        query_mask = input_ids_reshaped[..., 0] > 0

        self._text_cache[cache_key] = (query_embeds, query_mask)
        self._text_cache.move_to_end(cache_key)
        while len(self._text_cache) > 8:
            self._text_cache.popitem(last=False)
        return query_embeds.repeat(batch_size, 1, 1), query_mask.repeat(batch_size, 1)

    def _preprocess_square_torch(self, image: torch.Tensor):
        image = self._as_rgb3_batch(image.detach().cpu())
        batch_size, _, orig_h, orig_w = image.shape
        image = F_nn.pad(
            image,
            pad=square_padding(orig_h, orig_w),
            mode="constant",
            value=0.5,
        )
        image = F_nn.interpolate(
            image,
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixel_values = image.to(self.device, dtype=self.dtype)
        pixel_values = (pixel_values - self._mean) / self._std
        if pixel_values.shape[0] != batch_size:
            raise RuntimeError("Batch size changed during OWLv2 preprocessing")
        return pixel_values, [(orig_h, orig_w) for _ in range(batch_size)]

    def _forward_owl(
        self,
        pixel_values: torch.Tensor,
        prompt_texts: list[str],
        num_classes: int,
        num_templates: int,
    ):
        batch_size = pixel_values.shape[0]
        query_embeds, query_mask = self._get_text_embeddings(prompt_texts, batch_size)
        feature_map, vision_outputs = self.model.image_embedder(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        _, patches_h, patches_w, hidden_dim = feature_map.shape
        image_feats = feature_map.reshape(batch_size, patches_h * patches_w, hidden_dim)
        pred_logits, _ = self.model.class_predictor(image_feats, query_embeds, query_mask)
        if num_templates != len(PROMPT_TEMPLATES):
            raise ValueError(f"Expected {len(PROMPT_TEMPLATES)} prompt templates")
        pred_logits = average_prompt_logits(pred_logits, num_classes)
        return SimpleNamespace(
            logits=pred_logits,
            pred_boxes=self.model.box_predictor(image_feats, feature_map, True),
            objectness_logits=self.model.objectness_predictor(image_feats),
            vision_model_output=vision_outputs,
            num_patches_height=patches_h,
            num_patches_width=patches_w,
        )

    @staticmethod
    def _post_process_with_query_indices(
        outputs,
        threshold: float,
        target_sizes: torch.Tensor,
        text_labels: list[list[str]],
    ) -> list[dict]:
        class_logits = torch.max(outputs.logits, dim=-1)
        scores = torch.sigmoid(class_logits.values)
        labels = class_logits.indices

        cxcy = outputs.pred_boxes[..., :2]
        wh = outputs.pred_boxes[..., 2:]
        boxes = torch.cat((cxcy - wh / 2, cxcy + wh / 2), dim=-1)
        max_size = torch.max(target_sizes[:, 0], target_sizes[:, 1])
        scale = torch.stack((max_size, max_size, max_size, max_size), dim=1).to(
            device=boxes.device,
            dtype=boxes.dtype,
        )
        boxes = boxes * scale[:, None, :]

        results = []
        for image_scores, image_labels, image_boxes, image_text_labels in zip(
            scores,
            labels,
            boxes,
            text_labels,
        ):
            keep = torch.nonzero(image_scores > threshold, as_tuple=False).flatten()
            kept_labels = image_labels[keep]
            results.append(
                {
                    "scores": image_scores[keep],
                    "labels": kept_labels,
                    "boxes": image_boxes[keep],
                    "query_indices": keep,
                    "text_labels": [
                        image_text_labels[int(label)]
                        for label in kept_labels.detach().cpu().tolist()
                    ],
                }
            )
        return results

    @staticmethod
    def _filter_detections(results: list[dict], orig_sizes: list[tuple[int, int]]) -> list[dict]:
        filtered = []
        for result, (height, width) in zip(results, orig_sizes):
            boxes = result["boxes"]
            if len(boxes) == 0:
                filtered.append(result)
                continue

            box_widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            box_heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            keep = (box_widths * box_heights) <= (MAX_BOX_AREA_FRACTION * height * width)
            kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
            if len(kept_indices) > 0:
                nms_indices = nms(
                    boxes[kept_indices].float(),
                    result["scores"][kept_indices].float(),
                    NMS_IOU_THRESHOLD,
                )
                kept_indices = kept_indices[nms_indices]

            item = {
                "boxes": result["boxes"][kept_indices],
                "scores": result["scores"][kept_indices],
                "labels": result["labels"][kept_indices],
                "query_indices": result["query_indices"][kept_indices],
            }
            if "text_labels" in result:
                item["text_labels"] = [
                    result["text_labels"][int(idx)] for idx in kept_indices.detach().cpu().tolist()
                ]
            filtered.append(item)
        return filtered

    def _extract_region_embeddings(
        self,
        outputs,
        results: list[dict],
        orig_sizes: list[tuple[int, int]],
    ) -> list[list[np.ndarray]]:
        if not any(len(result["boxes"]) for result in results):
            return [[] for _ in results]

        last_hidden = outputs.vision_model_output.last_hidden_state.detach().cpu().float().numpy()
        pooler_output = outputs.vision_model_output.pooler_output.detach().cpu().float().numpy()
        patches_h = int(outputs.num_patches_height)
        patches_w = int(outputs.num_patches_width)
        patch_size = int(self.model.config.vision_config.patch_size)

        all_embeddings = []
        for image_idx, (result, (orig_h, orig_w)) in enumerate(zip(results, orig_sizes)):
            boxes_np = result["boxes"].detach().cpu().float().numpy()
            spatial = last_hidden[image_idx]
            pooler = pooler_output[image_idx]
            max_patch_idx = spatial.shape[0]
            scale = self._image_size / max(orig_h, orig_w)
            image_embeddings = []

            for x1, y1, x2, y2 in boxes_np:
                px1 = int(np.clip((x1 * scale) / patch_size, 0, patches_w - 1))
                py1 = int(np.clip((y1 * scale) / patch_size, 0, patches_h - 1))
                px2 = max(px1 + 1, min(int((x2 * scale) / patch_size) + 1, patches_w))
                py2 = max(py1 + 1, min(int((y2 * scale) / patch_size) + 1, patches_h))

                py_indices = np.arange(py1, py2)
                px_indices = np.arange(px1, px2)
                patch_indices = 1 + py_indices[:, None] * patches_w + px_indices
                patch_indices = patch_indices.ravel()
                patch_indices = patch_indices[patch_indices < max_patch_idx]
                if len(patch_indices) > 0:
                    embedding = spatial[patch_indices].mean(axis=0)
                else:
                    embedding = pooler
                image_embeddings.append(np.asarray(embedding, dtype=np.float32))

            all_embeddings.append(image_embeddings)
        return all_embeddings

    @staticmethod
    def _canonical_top_k_label_probs(
        logits: torch.Tensor,
        texts: list[str],
        synonym_to_canonical: dict,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        return canonical_top_k(texts, probs, synonym_to_canonical, k)

    @torch.inference_mode()
    def detect_batch(
        self,
        image: torch.Tensor,
        detection_classes: dict,
        threshold: float = 0.3,
    ) -> list[DetectionOutput]:
        pixel_values, orig_sizes = self._preprocess_square_torch(image)
        texts, synonym_to_canonical, text_labels, prompt_texts, num_templates = (
            self._class_metadata(
                detection_classes,
                batch_size=len(orig_sizes),
            )
        )
        outputs = self._forward_owl(
            pixel_values,
            prompt_texts,
            num_classes=len(texts),
            num_templates=num_templates,
        )
        results = self._post_process_with_query_indices(
            outputs=outputs,
            threshold=threshold,
            target_sizes=torch.tensor(orig_sizes, device=self.device),
            text_labels=text_labels,
        )
        results = self._filter_detections(results, orig_sizes)
        embeddings_per_image = self._extract_region_embeddings(outputs, results, orig_sizes)

        outputs_per_image = []
        for batch_idx, ((height, width), result, embeddings) in enumerate(
            zip(orig_sizes, results, embeddings_per_image)
        ):
            result_labels = result.get("text_labels")
            if result_labels is None:
                result_labels = [texts[int(idx)] for idx in result["labels"].tolist()]
            if len(embeddings) != len(result["boxes"]):
                raise RuntimeError(
                    "OWLv2 embedding extraction count mismatch: "
                    f"embeddings={len(embeddings)} boxes={len(result['boxes'])}"
                )

            detections = []
            for box, score, text_label, embedding, query_idx in zip(
                result["boxes"],
                result["scores"],
                result_labels,
                embeddings,
                result["query_indices"],
            ):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                name = synonym_to_canonical.get(text_label, text_label)
                detections.append(
                    DetectionEntity(
                        name=name,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score.item()),
                        embeddings=embedding,
                        top_k_label_probs=self._canonical_top_k_label_probs(
                            outputs.logits[batch_idx, int(query_idx)],
                            texts,
                            synonym_to_canonical,
                            k=TOP_K_LABELS,
                        ),
                    )
                )

            detections.sort(key=lambda entity: entity.confidence, reverse=True)
            outputs_per_image.append(
                DetectionOutput(detections=detections)
            )

        return outputs_per_image
