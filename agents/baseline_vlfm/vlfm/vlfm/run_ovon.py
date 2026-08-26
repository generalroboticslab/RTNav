# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Entry point for running VLFM evaluation on HM3D-OVON."""

import os

import frontier_exploration  # noqa
import hydra
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401

# Register OVON components
from ovon.config import HabitatConfigPlugin as OVONConfigPlugin

register_hydra_plugin(OVONConfigPlugin)

import ovon.task.sensors  # noqa: F401
import ovon.measurements  # noqa: F401
import ovon.dataset  # noqa: F401
import ovon.task.simulator  # noqa: F401


class VLFMConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(VLFMConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/vlfm_objectnav_ovon",
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)

    cfg = patch_config(cfg)
    with read_write(cfg):
        cfg.habitat.task.lab_sensors.pop("objectgoal_sensor")
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    execute_exp(cfg, "eval")



if __name__ == "__main__":
    main()
