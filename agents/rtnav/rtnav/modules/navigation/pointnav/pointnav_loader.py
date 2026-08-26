"""Loads the VLFM/BDAI PointNav policy without a habitat install.

The checkpoint is a habitat-baselines-wrapped PointNav ResNet with a discrete
4-way action head (STOP/FORWARD/LEFT/RIGHT). The agent container has neither
habitat-sim nor habitat-baselines, so three workarounds live here:

  1. a stub ``gym`` module — vlfm does ``from gym import spaces`` at import time;
  2. stub ``habitat`` / ``habitat_baselines`` modules while unpickling the ckpt;
  3. a one-time "flat" state-dict extracted from the wrapped checkpoint and
     cached to disk, so only the env container (which has habitat) pays for it.

Public API: ``resolve_ckpt``, ``load_policy``, ``prepare_depth``, ``SUCCESS_DIST``.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Make ``from vlfm.policy... import ...`` resolve. In Docker rtnav is bind-mounted
# at /opt/rtnav (away from the repo root), so __file__ can't reach the repo —
# use RT_OVN_ROOT, falling back to the dev-checkout layout.
_rt_ovn_root = os.environ.get("RT_OVN_ROOT", "").strip()
_REPO_ROOT = Path(_rt_ovn_root) if _rt_ovn_root else Path(__file__).resolve().parents[6]
_VLFM_ROOT = _REPO_ROOT / "agents" / "baseline_vlfm" / "vlfm"
for _p in (_REPO_ROOT, _VLFM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


SUCCESS_DIST = 0.2  # m — standard PointNav success threshold
_MAX_DEPTH = 5.0  # m — depth normalization range
_DEPTH_HW = (224, 224)  # PointNav was trained on 224x224 depth
_DEPTH_METERS_THRESHOLD = 1.5  # max > this ⇒ raw depth is in meters, not [0,1]
_DISCRETE_EMBED_SHAPE = (5, 32)  # prev-action-embedding weight shape of the 4-way head

_CKPT_SEARCH_PATHS = (
    os.environ.get("POINTNAV_CKPT"),
    str(_VLFM_ROOT / "data" / "pointnav_weights.pth"),
    str(Path.home() / "pointnav_ckpts" / "pointnav_weights.pth"),
    "/opt/vlfm/data/pointnav_weights.pth",
)


def find_ckpt() -> Optional[str]:
    """First existing checkpoint path from the search list, or None."""
    for path in _CKPT_SEARCH_PATHS:
        if path and Path(path).exists():
            return path
    return None


def resolve_ckpt(ckpt_path: Optional[str] = None) -> str:
    """Return a usable checkpoint path, or raise listing the paths tried."""
    resolved = ckpt_path or find_ckpt()
    if resolved is None:
        raise FileNotFoundError(
            "PointNav checkpoint not found in any of: "
            + ", ".join(repr(p) for p in _CKPT_SEARCH_PATHS if p)
        )
    return resolved


# ── gym stub ─────────────────────────────────────────────────────────
# vlfm does a top-level ``from gym import spaces``; install a stub so that
# import succeeds when gym isn't present. The spaces are never constructed.
def _install_gym_stub() -> None:
    try:
        import gym  # noqa: F401

        return
    except ImportError:
        pass
    gym_mod = types.ModuleType("gym")
    spaces_mod = types.ModuleType("gym.spaces")

    class _StubSpace:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("gym not installed — stub only exists for import.")

    spaces_mod.Box = _StubSpace
    spaces_mod.Dict = _StubSpace
    spaces_mod.Discrete = _StubSpace
    gym_mod.spaces = spaces_mod
    sys.modules["gym"] = gym_mod
    sys.modules["gym.spaces"] = spaces_mod


_install_gym_stub()


# ── habitat import stubs (used only while unpickling the checkpoint) ──
# The wrapped checkpoint pickles references to habitat classes we don't have
# installed. A meta-path finder hands back permissive stub modules whose
# attributes are auto-created no-op classes.
class _HabitatStubClass:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __reduce__(self):
        return (self.__class__, ())


class _HabitatStubModule(types.ModuleType):
    def __init__(self, fullname: str):
        super().__init__(fullname)
        self.__path__ = []

    def __getattr__(self, name: str):
        if name.startswith("_") and name not in {"__all__", "__loader__", "__spec__"}:
            raise AttributeError(name)
        klass = type(name, (_HabitatStubClass,), {})
        setattr(self, name, klass)
        return klass


class _HabitatStubFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes: tuple):
        self._prefixes = prefixes

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._prefixes):
            return importlib.machinery.ModuleSpec(fullname, _HabitatStubLoader(), is_package=True)
        return None


class _HabitatStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _HabitatStubModule(spec.name)

    def exec_module(self, module):
        return


# ── flat state-dict cache ────────────────────────────────────────────
# Unpickling the wrapped checkpoint is expensive (needs the habitat stubs).
# Extract its plain state-dict once and cache it next to the checkpoint so
# later loads (especially in the agent container) just read a flat file.
def _flat_ckpt_candidates(ckpt_path: str) -> list:
    name = Path(ckpt_path).name
    candidates = [
        "/opt/rt_ovn/data/" + name + ".flat.pth",
        ckpt_path + ".flat.pth",
        str(Path.home() / "pointnav_ckpts" / (name + ".flat.pth")),
    ]
    override = os.environ.get("POINTNAV_FLAT_CKPT")
    if override:
        candidates.insert(0, override)
    return candidates


@contextmanager
def _flat_ckpt_lock(ckpt_path: str):
    """Serialize the one-time flat-checkpoint extraction across parallel workers."""
    import fcntl

    fh = None
    try:
        for cand in _flat_ckpt_candidates(ckpt_path):
            try:
                parent = Path(cand).parent
                parent.mkdir(parents=True, exist_ok=True)
                fh = open(str(parent / ".pointnav_flat_ckpt.lock"), "a")
                fcntl.flock(fh, fcntl.LOCK_EX)
                break
            except (PermissionError, OSError):
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                    fh = None
        yield
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            finally:
                fh.close()


def _ensure_flat_ckpt(ckpt_path: str) -> str:
    """Return a flat state-dict path, extracting one from the wrapped
    checkpoint if none is cached yet. Only the env container (real habitat)
    can extract; the agent container just reads the cached file."""
    with _flat_ckpt_lock(ckpt_path):
        for cand in _flat_ckpt_candidates(ckpt_path):
            if Path(cand).exists():
                return cand
        finder = _HabitatStubFinder(("habitat", "habitat_baselines"))
        sys.meta_path.append(finder)
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        finally:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass
        state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        last_err = None
        for cand in _flat_ckpt_candidates(ckpt_path):
            tmp = f"{cand}.tmp.{os.getpid()}"
            try:
                Path(cand).parent.mkdir(parents=True, exist_ok=True)
                print(f"[pointnav] extracting flat state_dict → {cand} (one-time)")
                torch.save(state_dict, tmp)
                os.replace(tmp, cand)
                return cand
            except (PermissionError, OSError, RuntimeError) as e:
                last_err = e
                try:
                    if Path(tmp).exists():
                        Path(tmp).unlink()
                except Exception:
                    pass
        raise RuntimeError(f"Couldn't write flat PointNav state_dict: {last_err}")


# ── discrete-action head patch ───────────────────────────────────────
# vlfm's no-habitat loader only builds the continuous Gaussian PointNav head;
# the BDAI checkpoint has a 4-way categorical head. Temporarily swap vlfm's
# loader for one that builds the discrete variant, then restore it.
_vlfm_orig_loader = None


def _patch_vlfm_loader_for_discrete() -> None:
    import torch.nn as nn
    import vlfm.policy.utils.pointnav_policy as _pnp
    from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetNet,
        PointNavResNetPolicy,
    )

    global _vlfm_orig_loader
    _vlfm_orig_loader = _pnp.load_pointnav_policy

    class _DiscretePointNavPolicy(PointNavResNetPolicy):
        def __init__(self):
            nn.Module.__init__(self)
            self.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
            self.action_distribution = nn.Module()
            self.action_distribution.linear = nn.Linear(512, 4)

        def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False):
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
            logits = self.action_distribution.linear(features)
            if deterministic:
                action = logits.argmax(dim=-1, keepdim=True)
            else:
                from torch.distributions import Categorical

                action = Categorical(logits=logits).sample().unsqueeze(-1)
            return action, rnn_hidden_states

    def _load(file_path):
        policy = _DiscretePointNavPolicy()
        sd = torch.load(file_path, map_location="cpu")
        renamed = {
            (
                k.replace("net.prev_action_embedding.", "net.prev_action_embedding_discrete.")
                if k.startswith("net.prev_action_embedding.")
                else k
            ): v
            for k, v in sd.items()
        }
        cur = policy.state_dict()
        loadable = {k: v for k, v in renamed.items() if k in cur}
        try:
            policy.load_state_dict(loadable, strict=False, assign=True)
        except TypeError:
            policy.load_state_dict(loadable, strict=False)
        return policy

    _pnp.load_pointnav_policy = _load  # type: ignore[assignment]


def _restore_vlfm_loader() -> None:
    global _vlfm_orig_loader
    if _vlfm_orig_loader is None:
        return
    import vlfm.policy.utils.pointnav_policy as _pnp

    _pnp.load_pointnav_policy = _vlfm_orig_loader  # type: ignore[assignment]
    _vlfm_orig_loader = None


def load_policy(ckpt_path: Optional[str], device: torch.device):
    """Load the BDAI PointNav checkpoint via vlfm's loader.

    In the agent container (no habitat) we take vlfm's non-habitat fallback and
    apply the discrete-head patch; in the env container (full habitat) we go
    through vlfm's normal path.
    """
    ckpt_path = resolve_ckpt(ckpt_path)

    from vlfm.policy.utils.pointnav_policy import (  # noqa: E402
        HABITAT_BASELINES_AVAILABLE,
        WrappedPointNavResNetPolicy,
    )

    load_path = ckpt_path
    if not HABITAT_BASELINES_AVAILABLE:
        for cand in _flat_ckpt_candidates(ckpt_path):
            if Path(cand).exists():
                load_path = cand
                break
        else:
            raise RuntimeError(
                "No flat state-dict found. Run the loader once in the env "
                "container to extract weights from the wrapped BDAI checkpoint."
            )
    else:
        try:
            _ensure_flat_ckpt(ckpt_path)
        except Exception as e:
            print(f"[pointnav] flat-state-dict cache prep failed: {e}")

    # vlfm's loader calls torch.load without weights_only=False; force it so the
    # pickled checkpoint still loads on newer torch.
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]
    patched_loader = False
    try:
        if not HABITAT_BASELINES_AVAILABLE:
            sd_probe = torch.load(load_path, map_location="cpu")
            is_discrete = (
                "net.prev_action_embedding.weight" in sd_probe
                and sd_probe["net.prev_action_embedding.weight"].shape == _DISCRETE_EMBED_SHAPE
            )
            if is_discrete:
                _patch_vlfm_loader_for_discrete()
                patched_loader = True
        return WrappedPointNavResNetPolicy(load_path, device=device)
    finally:
        torch.load = _orig_load  # type: ignore[assignment]
        if patched_loader:
            _restore_vlfm_loader()


def prepare_depth(raw: np.ndarray) -> np.ndarray:
    """Habitat raw depth → PointNav-ready 224x224 float32 in [0, 1]."""
    d = np.asarray(raw, dtype=np.float32).squeeze()
    if float(d.max(initial=0.0)) > _DEPTH_METERS_THRESHOLD:
        d = np.clip(d, 0.0, _MAX_DEPTH) / _MAX_DEPTH
    else:
        d = np.clip(d, 0.0, 1.0)
    return cv2.resize(d, (_DEPTH_HW[1], _DEPTH_HW[0]), interpolation=cv2.INTER_AREA)
