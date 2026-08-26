"""Exploitation utilities for target selection."""

import re
from typing import Any, Dict, List, Optional


def normalize_target_names(param: Any) -> List[str]:
    """Normalize target parameter to list of lowercase names."""
    if param is None:
        return []
    if isinstance(param, str):
        return [param.lower()]
    if isinstance(param, (list, tuple)):
        return [str(p).lower() for p in param]
    return [str(param).lower()]


def target_matches(target: Dict, names: List[str]) -> bool:
    """Check if target matches any name (exact or partial)."""
    canonical = target.get("canonical", "").lower()
    label = target.get("label", "").lower()

    for name in names:
        if name == canonical or name == label:
            return True
        if name in canonical or name in label:
            return True
        if canonical in name or label in name:
            return True
    return False


def compact_target_name(name: Any) -> str:
    """Lowercase label with spaces/punctuation removed."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def label_matches_any(label: Any, names) -> bool:
    """Exact target-label match, allowing only spacing/punctuation variants."""
    label_l = str(label).lower().strip()
    if not label_l:
        return False
    names_l = {str(n).lower().strip() for n in (names or []) if str(n).strip()}
    if label_l in names_l:
        return True
    label_c = compact_target_name(label_l)
    names_c = {compact_target_name(n) for n in names_l}
    return bool(label_c and label_c in names_c)


TARGET_CONFIRMATION_MIN_TARGET_MASS = 3.0
TARGET_OBSERVE_MIN_TARGET_MASS = 2.0
TARGET_OBSERVE_PROMOTE_MASS = TARGET_CONFIRMATION_MIN_TARGET_MASS


def _history_label_masses(history) -> tuple[Dict[str, float], float]:
    label_masses: Dict[str, float] = {}
    total_weight = 0.0
    for entry in history:
        total_weight += 1.0
        if len(entry) >= 4 and entry[3]:
            label_probs = entry[3]
        elif len(entry) >= 2 and entry[0]:
            label_probs = [(entry[0], entry[1])]
        else:
            label_probs = []
        for label, prob in label_probs:
            label_s = str(label).lower().strip()
            label_masses[label_s] = label_masses.get(label_s, 0.0) + float(prob)
    return label_masses, total_weight


def target_label_probability_stats(
    node,
    target_set,
    history_start_index: int = 0,
) -> Dict[str, Any]:
    prob_count = int(getattr(node, "label_prob_count", 0) or 0)
    prob_sums = getattr(node, "label_prob_sums", None) or {}
    history = getattr(node, "label_history", None) or []
    # ObjectNode already bounds label_history to the selected voting window.
    if history:
        start = max(0, int(history_start_index or 0))
        used_history = history[start:]
        views = len(used_history)
        label_masses, evidence_denom = _history_label_masses(used_history)
    elif prob_count > 0 and prob_sums:
        views = prob_count
        label_masses = {
            str(label).lower().strip(): float(prob) for label, prob in prob_sums.items()
        }
        evidence_denom = float(prob_count)
    else:
        views = 0
        label_masses = {}
        evidence_denom = 0.0
    target_mass = sum(
        mass for label, mass in label_masses.items() if label_matches_any(label, target_set)
    )
    non_targets = {
        label: mass
        for label, mass in label_masses.items()
        if not label_matches_any(label, target_set)
    }
    max_non_target_label, max_non_target_mass = (
        max(non_targets.items(), key=lambda item: item[1]) if non_targets else ("", 0.0)
    )
    target_frac = target_mass / evidence_denom if evidence_denom else 0.0
    max_non_target_frac = max_non_target_mass / evidence_denom if evidence_denom else 0.0
    return {
        "views": views,
        "target_mass": target_mass,
        "target_frac": target_frac,
        "max_non_target_label": max_non_target_label,
        "max_non_target_mass": max_non_target_mass,
        "max_non_target_frac": max_non_target_frac,
        "target_dominates_non_target": target_mass > max_non_target_mass,
    }


def target_confirmation_summary(
    node,
    target_set,
) -> Dict[str, Any]:
    """Human-readable target-confirmation state for one SG node."""
    stats = target_label_probability_stats(node, target_set)
    current_label = (
        str(getattr(node, "chosen_label", None) or getattr(node, "label", "")).lower().strip()
    )
    current_label_matches = label_matches_any(current_label, target_set)
    mass_passes = stats["target_mass"] >= TARGET_CONFIRMATION_MIN_TARGET_MASS
    dominance_passes = stats["target_dominates_non_target"]
    regular_passes = current_label_matches and mass_passes and dominance_passes
    vote_passes = regular_passes
    passes = vote_passes
    if not mass_passes:
        reason = (
            f"target evidence {stats['target_mass']:.2f} < "
            f"{TARGET_CONFIRMATION_MIN_TARGET_MASS:.2f}"
        )
    elif not current_label_matches:
        reason = f"current label '{current_label}' is not a target label"
    elif not dominance_passes:
        reason = (
            f"target evidence {stats['target_mass']:.2f} not > "
            f"non-target '{stats['max_non_target_label']}' "
            f"({stats['max_non_target_mass']:.2f})"
        )
    else:
        reason = f"target evidence {stats['target_mass']:.2f} passes"

    return {
        "views": int(stats["views"]),
        "target_mass": float(stats["target_mass"]),
        "target_frac": float(stats["target_frac"]),
        "max_non_target_label": stats["max_non_target_label"],
        "max_non_target_mass": float(stats["max_non_target_mass"]),
        "target_dominates_non_target": bool(dominance_passes),
        "current_label": current_label,
        "current_label_matches": current_label_matches,
        "vote_passes": vote_passes,
        "passes": passes,
        "reason": reason,
    }


def target_confirmation_passes(node, target_set) -> bool:
    """Target confirmation rule: target probability mass must dominate."""
    return target_confirmation_summary(node, target_set)["passes"]


def target_observe_summary(
    node,
    target_set,
    history_start_index: int = 0,
    threshold: float = TARGET_OBSERVE_MIN_TARGET_MASS,
) -> Dict[str, Any]:
    """Lower-confidence target-of-interest gate over the recent window."""
    stats = target_label_probability_stats(
        node,
        target_set,
        history_start_index=history_start_index,
    )
    current_label = (
        str(getattr(node, "chosen_label", None) or getattr(node, "label", "")).lower().strip()
    )
    current_label_matches = label_matches_any(current_label, target_set)
    mass_passes = stats["target_mass"] >= threshold
    dominance_passes = stats["target_dominates_non_target"]
    passes = current_label_matches and mass_passes and dominance_passes
    if not mass_passes:
        reason = f"target evidence {stats['target_mass']:.2f} < {threshold:.2f}"
    elif not current_label_matches:
        reason = f"current label '{current_label}' is not a target label"
    elif not dominance_passes:
        reason = (
            f"target evidence {stats['target_mass']:.2f} not > "
            f"non-target '{stats['max_non_target_label']}' "
            f"({stats['max_non_target_mass']:.2f})"
        )
    else:
        reason = f"target evidence {stats['target_mass']:.2f} passes"
    return {
        **stats,
        "current_label": current_label,
        "current_label_matches": current_label_matches,
        "passes": passes,
        "reason": reason,
    }


def target_observe_abort_reason(node, target_set, history_start_index: int) -> Optional[str]:
    stats = target_label_probability_stats(
        node,
        target_set,
        history_start_index=history_start_index,
    )
    target_mass = float(stats["target_mass"])
    non_label = str(stats["max_non_target_label"])
    non_mass = float(stats["max_non_target_mass"])
    if target_mass < TARGET_OBSERVE_MIN_TARGET_MASS:
        return f"observe target evidence {target_mass:.2f} < {TARGET_OBSERVE_MIN_TARGET_MASS:.2f}"
    if non_label and non_mass > target_mass:
        return (
            f"observe non-target '{non_label}' evidence {non_mass:.2f} "
            f"> target evidence {target_mass:.2f}"
        )
    return None
