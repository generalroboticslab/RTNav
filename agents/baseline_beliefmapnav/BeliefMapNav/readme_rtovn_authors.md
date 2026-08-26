# BeliefMapNav install (rt_ovn authors)

Install notes for CUDA 12 hosts (tested target: 8x NVIDIA L40S, driver 580.119,
`nvcc` 12.4 at `/usr/bin/nvcc`, gcc 11.4, Python 3.9). This replaces the
`Installation` section of [README.md](README.md); everything else in the
upstream README (dataset download, evaluation) still applies.

## Why this differs from the upstream README

- The upstream README installs `torch==1.12.1+cu113`, but `requirements.txt`
  later pins `torch==2.0.1` and `nvidia-*-cu12` libs, so the 1.12.1 line just
  gets overwritten. Neither supports `sm_89` (Ada Lovelace) well. We install
  `torch==2.1.0` from the `cu121` wheel index once and keep it — same as the
  `gamap` and `openfmnav` envs on this host. `salesforce-lavis==1.0.2` and
  habitat-sim 0.2.5 are both fine with torch 2.1.x.
- `pip install -r requirements.txt` cannot run: that file is a raw `pip freeze`
  containing conda `@ file:///home/conda/...` pins (`attrs`, `magnum`,
  `mkl_fft`, ...), three `-e git+` editable entries, `habitat-sim==0.2.5`
  (conda-only, not on PyPI), and `ghfast.top` mirror URLs. Use
  [requirements_by_rt_ovn_authors.txt](requirements_by_rt_ovn_authors.txt)
  instead — same pins, with those lines removed, the torch/`nvidia-*`/`triton`/
  `xformers` pins dropped (they would fight the cu121 torch above), and `cupy`
  swapped for `cupy-cuda12x`. `xformers`, `mmyolo`, `mmengine`, `decord`,
  `webdataset`, `lmdb`, `trimesh`, `streamlit`, `dash`, and `kaggle` are never
  imported by the code. `albucore` is dropped too — it requires
  `opencv-python-headless>=4.9.0.80`, contradicting the `4.5.5.64` pin below,
  and nothing imports it.
- habitat-sim is installed from the `aihabitat` conda channel **first**, before
  any pip step. The upstream README installs it last via
  `pip install git+...habitat-sim@v0.2.5` (a from-source build); the conda
  build is prebuilt, and running the conda solver before pip stops it from
  downgrading `numpy`/`python` underneath the pip packages.

## A. Conda env and habitat-sim

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n bmn python=3.9 -y
conda activate bmn
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 headless -y
```

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"   # 0.2.5
```

## B. PyTorch (cu121)

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## C. LAVIS (BLIP2 backbone)

`--no-deps` because lavis's unpinned `spacy` dependency resolves to the latest
release, which pulls `thinc>=8.3.12` (Python 3.10+ only) and fails on 3.9. Step
F pins `spacy==3.8.3` / `thinc==8.3.4`, so lavis's real transitive deps arrive
there.

```bash
pip install --no-deps salesforce-lavis==1.0.2
```

## D. GroundingDINO

GroundingDINO compiles a CUDA op at install time and its `setup.py` imports
`torch` at module top level. `--no-build-isolation` makes pip reuse the torch
from step B instead of building in an empty env; `CUDA_HOME` points `nvcc` at
the system CUDA headers.

```bash
CUDA_HOME=/usr pip install --no-build-isolation \
  git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67
```

A pip install leaves no `GroundingDINO/` directory, which is where both
`vlfm/vlm/grounding_dino.py` and `scripts/launch_vlm_servers.sh` look for the
model config, so the server dies at startup with `FileNotFoundError`.
`grounding_dino.py` now reads `GROUNDING_DINO_CONFIG` (rt_ovn author addition);
export it before launching any VLM server:

```bash
export GROUNDING_DINO_CONFIG="$(python -c \
  'import os, groundingdino; print(os.path.join(os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py"))')"
```

## E. habitat-lab / habitat-baselines 0.2.5

BeliefMapNav ships a patched `env_utils.py` that makes evaluation resumable; it
has to overwrite the copy in habitat-lab. Run from the directory *containing*
`BeliefMapNav/`.

```bash
cd ..    # -> rt_ovn_baselines/
git clone --depth 1 --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cp BeliefMapNav/habitat_lab/env_utils.py habitat-lab/habitat-lab/habitat/core/env_utils.py
pip install -e habitat-lab/habitat-lab
pip install -e habitat-lab/habitat-baselines
cd BeliefMapNav
```

## F. Remaining Python dependencies

```bash
pip install -r requirements_by_rt_ovn_authors.txt
```

This downgrades `numpy` to 1.26.4 (GroundingDINO pulls numpy 2.x, which
habitat-sim and the pinned `opencv-python-headless==4.5.5.64` cannot use).

## F2. Keep only the headless OpenCV

Required, not optional. Steps D and E pull in `opencv-python` 4.11.0.86, and
step F installs `opencv-python-headless==4.5.5.64` (pinned exactly by lavis).
Both unpack into the same `site-packages/cv2/`, so the headless install
overwrites the shared binaries but leaves 4.11's `cv2/typing/` behind — a
subpackage that did not exist before OpenCV 4.8. It is imported by `import cv2`
and fails with:

```
AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
```

Uninstall both, clear the leftover directory, and reinstall only the headless
build. `--no-deps` keeps pip from pulling `opencv-python` back in.

```bash
pip uninstall -y opencv-python opencv-python-headless
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/cv2"
pip install --no-deps opencv-python-headless==4.5.5.64
```

`pip check` then reports `opencv-python, which is not installed` for
groundingdino, habitat-lab, mmengine, supervision, and ultralytics. Expected and
harmless: headless provides the same `cv2` module. The remaining `pip check`
lines are expected too — `habitat-sim ... requires numpy<1.24.0` (step F),
`groundingdino ... requires supervision==0.6.0`, and `decord 0.6.0 is not
supported on this platform` (`decord` is never imported).

## G. yolov7 source tree

`vlfm/vlm/yolov7.py` imports from the yolov7 repo at runtime, so the source
tree has to sit inside `BeliefMapNav/`.

```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

## H. Model weights

All checkpoints go in `BeliefMapNav/data/`.

```bash
mkdir -p data && cd data

# MobileSAM
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# GroundingDINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# YOLOv7-E6E
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

# PointNav
wget https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth

# SAM 2.1-b, used by vlfm/mapping/openfusion/slam.py (see Upstream patches).
# ultralytics fetches it automatically on first use if you skip this.
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt

# Fine-tuned RedNet segmentation (same weights L3MVN and GAMap use)
curl -L -o rednet_semmap_mp3d_40.pth \
  "https://drive.usercontent.google.com/download?id=1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv&export=download&confirm=t"

cd ..
```

`curl` rather than `gdown`: `vision-aided-loss==0.1.0` hard-pins `gdown==4.4.0`,
which predates Google's current large-file confirm-token flow and reports the
interstitial page as `Access denied`. The file is 627 MB, md5
`1deed657c87fd08f052c0607910b7722` — the same checkpoint as
`L3MVN/RedNet/model/rednet_semmap_mp3d_40.pth`.

## I. LLM key

`vlfm/vlm/openai_api.py` calls GPT-4o for detection refinement and
multi-candidate disambiguation.

```bash
export OPENAI_API_KEY="sk-..."
```

Upstream hard-codes the key and always passes a `base_url` for a third-party
relay, so a plain OpenAI key could not work; both are now read from the
environment. Set `OPENAI_BASE_URL` only if you route through a proxy or gateway
— unset, the SDK uses `https://api.openai.com/v1`.

For the reproducibility probe, `BMN_DUMMY_LLM=1` returns deterministic canned
answers and skips the API call. Note the stubs accept every detection
(`detection_refinement` always answers yes) and pick the first candidate, so
dummy runs are reproducible but their success/SPL numbers are not meaningful.

## Upstream patches

Four source files carry rt_ovn author additions, each marked in-line. All are
no-ops unless the environment variable is set, so upstream behaviour is
unchanged.

| File | Change |
| --- | --- |
| `vlfm/vlm/openai_api.py` | `BMN_DUMMY_LLM=1` stubs the GPT-4o calls; key/endpoint read from `OPENAI_API_KEY`/`OPENAI_BASE_URL` (step I) |
| `vlfm/vlm/grounding_dino.py` | read `GROUNDING_DINO_CONFIG`/`_WEIGHTS` from the env (step D) |
| `vlfm/mapping/openfusion/slam.py` | SAM 2.1-b path was hard-coded to `/home/ubuntu/DATA2/zzb/`; now `data/sam2.1_b.pt`, overridable with `SAM2_CHECKPOINT` |
| `vlfm/utils/vlfm_trainer.py` | `BMN_TRACE_PATH` writes the action trace the rt_ovn probe compares |

## Verify

```bash
conda activate bmn
python -c "
import torch, habitat_sim, habitat, habitat_baselines, groundingdino, lavis
import cupy, open3d, alphashape, shapely, ultralytics, torchshow, cv2, numpy
print('torch     ', torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('habitat_sim', habitat_sim.__version__)
print('numpy     ', numpy.__version__)
print('cv2       ', cv2.__version__)
"
python -c "import vlfm"
```

`vlfm` is not pip-installable — `setup.py` is a bare `setup()` with no
`setup.cfg` or `pyproject.toml`, so there is no package metadata to install
from. `scripts/launch_multi_eval.sh` runs `python -m vlfm.run`, which relies on
the repo root being the working directory. Run everything from
`rt_ovn_baselines/BeliefMapNav/`, or `export PYTHONPATH="$PWD:$PYTHONPATH"`.

Datasets are documented in [README.md](README.md) and in the rt_ovn repo's
`docs/datasets.md`.
