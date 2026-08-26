"""Load the fixed configuration for a supported benchmark."""


def load_config(env_name: str):
    if env_name in ("hm3d", "hm3d_v1", "hm3d_v2"):
        from rtnav.config.environments.hm3d_config import HM3DConfig

        return HM3DConfig()
    if env_name == "ovon":
        from rtnav.config.environments.ovon_config import OVONConfig

        return OVONConfig()
    raise ValueError(
        f"Unsupported env_name={env_name!r}; expected 'hm3d_v1', 'hm3d_v2', or 'ovon'"
    )
