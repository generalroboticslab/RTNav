"""rt_ovn author addition: opt-in determinism for reproduction probes.

Two official BeliefMapNav runs of the same episode were observed to diverge at
step 110 (500 vs 193 actions), which makes exact action comparison against the
ROS port impossible. The cause is float nondeterminism — TF32, cuDNN autotuning
and non-deterministic CUDA atomics — amplified by the OpenFusion belief map,
which integrates every frame so a tiny difference eventually flips a frontier.

This must be applied in *every* process that touches CUDA, including the
GroundingDINO / MobileSAM / YOLOv7 servers, since the detectors run out of
process and their outputs feed the map.

No-op unless BMN_DETERMINISTIC=1, so upstream behaviour is unchanged.
"""

import os
import random


def enable_determinism(seed: int = 0) -> bool:
    """Pin every RNG and disable nondeterministic CUDA kernels. Returns whether
    determinism was actually enabled."""
    if os.environ.get("BMN_DETERMINISTIC", "").strip() != "1":
        return False

    # Must be set before the first cuBLAS handle is created; the probe also
    # exports it, this is the in-process fallback.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # warn_only: several ops in this stack have no deterministic kernel, and a
    # hard failure would just prevent the run. They are reported at runtime so
    # remaining nondeterminism can be traced to a specific op.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return True
