# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from pathlib import Path

import torch

from vlfm.run import get_config


def save_dummy_policy(filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # Save a dummy state_dict using torch.save
    config = get_config("config/experiments/vlfm_objectnav_hm3d.yaml")
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, filename)


if __name__ == "__main__":
    filename = os.environ.get("VLFM_DUMMY_POLICY_PATH", "data/dummy_policy.pth")
    save_dummy_policy(filename)
    print(f"Dummy policy weights saved to {filename}")
