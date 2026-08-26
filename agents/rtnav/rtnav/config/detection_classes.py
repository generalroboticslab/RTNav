"""OWLv2's always-on common-object vocabulary and LVIS aliases."""

import json
from pathlib import Path

LVIS_CATEGORIES = json.loads(Path(__file__).with_name("lvis_common.json").read_text())

_EXCLUDED = {"hinge", "latch", "wall socket", "doorknob", "knob", "handle"}
OBJECT_CLASSES = [
    row["name"].replace("_", " ")
    for row in LVIS_CATEGORIES
    if row["name"].replace("_", " ") not in _EXCLUDED
]
