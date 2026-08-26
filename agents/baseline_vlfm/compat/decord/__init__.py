"""Image-only compatibility surface for LAVIS on ARM64.

VLFM does not use LAVIS video datasets.  The real decord package does not
publish ARM64 wheels, but LAVIS imports it unconditionally from data_utils.
"""


class _Bridge:
    @staticmethod
    def set_bridge(_name: str) -> None:
        return None


bridge = _Bridge()


def cpu(index: int = 0):
    """Return an opaque CPU context for import compatibility only."""
    return ("cpu", index)


class VideoReader:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "LAVIS video loading requires decord, which is unavailable on ARM64; "
            "VLFM's image models do not use this path"
        )
