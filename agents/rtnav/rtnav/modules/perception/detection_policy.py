"""The fixed OWLv2 behavior used by RTNav."""

DEFAULT_THRESHOLD = 0.25
DEFAULT_INFERENCE_RES = 960
MAX_BOX_AREA_FRACTION = 0.8
NMS_IOU_THRESHOLD = 0.6
TOP_K_LABELS = 3

PROMPT_TEMPLATES = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)


def prompt_queries(labels):
    return [template.format(label) for label in labels for template in PROMPT_TEMPLATES]


def average_prompt_logits(logits, num_classes: int):
    """Average each class's seven adjacent prompt-template logits."""
    return logits.reshape(
        logits.shape[0],
        logits.shape[1],
        num_classes,
        len(PROMPT_TEMPLATES),
    ).mean(-1)


def square_padding(height: int, width: int):
    """Right/bottom padding tuple accepted by ``torch.nn.functional.pad``."""
    side = max(height, width)
    return 0, side - width, 0, side - height


def canonical_top_k(labels, probabilities, canonical_by_label, k=TOP_K_LABELS):
    """Collapse aliases by maximum probability and retain the best labels."""
    canonical_probs = {}
    for label, probability in zip(labels, probabilities):
        canonical = canonical_by_label.get(label, label)
        canonical_probs[canonical] = max(
            canonical_probs.get(canonical, 0.0),
            float(probability),
        )
    return sorted(
        canonical_probs.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]
