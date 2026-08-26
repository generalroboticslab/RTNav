# TriHelper ROS2 Baseline

This wrapper runs the HM3D TriHelper policy through the shared ROS2 environment
and agent interface. Policy constants are fixed to the paper settings; only the
shared evaluation controls such as `MODE`, `NUM_EPISODES`, and `GPU` are runtime
inputs.

Prepare the RedNet and Qwen-VL-Chat-Int4 models as described in
[`docs/baselines.md`](../../docs/baselines.md), then build and run from this
directory:

```bash
docker compose build agent
DATA_DIR=/path/to/data MODE=sync docker compose up env agent
```

The same image works with the repository's parallel launcher:

```bash
DATA_DIR=/path/to/data bash ../launch_parallel.sh \
  --baseline trihelper --benchmark hm3d_v1 \
  --episodes 2000 --workers 8 --gpus 0,1,2,3,4,5,6,7
```
