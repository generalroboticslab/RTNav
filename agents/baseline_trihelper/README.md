# TriHelper ROS2 Baseline

This wrapper runs the HM3D TriHelper policy through the shared ROS2 environment
and agent interface. Policy constants are fixed to the paper settings; only the
shared evaluation controls such as `MODE`, `NUM_EPISODES`, and `GPU` are runtime
inputs.

Prepare the RedNet and Qwen-VL-Chat-Int4 models as described in
[`docs/baselines.md`](../../docs/baselines.md), then build the shared and
TriHelper images using that guide.

Run evaluations from the repository root through the shared launcher:

```bash
DATA_DIR=/path/to/data bash agents/launch_parallel.sh \
  --baseline trihelper --benchmark hm3d_v1 \
  --episodes 1 --workers 1 --gpus 0 --launch
```

Increase `--episodes`, `--workers`, and `--gpus` for a full evaluation.
