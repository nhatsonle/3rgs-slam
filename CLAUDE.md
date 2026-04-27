# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

The project has a CUDA/C++ backend that must be compiled before any Python code runs:

```bash
pip install --no-build-isolation -e .
```

This compiles `mast3r_slam_backends` — a PyTorch CUDA extension containing three source files:
- `mast3r_slam/backend/src/gn.cpp` — Python bindings
- `mast3r_slam/backend/src/gn_kernels.cu` — Gauss-Newton pose solver + Eigen sparse Cholesky
- `mast3r_slam/backend/src/matching_kernels.cu` — iterative projection matching (LM) + descriptor refinement

The compiled `.so` lands at the repo root as `mast3r_slam_backends.cpython-*.so`. If you modify any `.cu` or `.cpp` file, re-run the build command. The extension is imported directly as `import mast3r_slam_backends` in `matching.py` and `global_opt.py`.

Thirdparty dependencies (must be installed separately):
```bash
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install gsplat #for 3DGS training
```

Checkpoints go in `checkpoints/` — three `.pth`/`.pkl` files from NAVER Labs (see README for URLs).

## Running

```bash
# Basic run (no calibration, monocular)
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml

# Calibrated run (recommended for accuracy)
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room --config config/calib.yaml

# Headless / evaluation mode
python main.py --dataset <path> --config config/eval_calib.yaml --no-viz --save-as results

# With explicit intrinsics (for video/folder inputs)
python main.py --dataset video.mp4 --config config/base.yaml --calib config/intrinsics.yaml
```
Always run on headless no-viz mode, running on IMG_2520.mp4 file for quick demo, no need to test on real dataset right now.

Evaluation scripts run in single-threaded headless mode and save trajectories to `logs/`:
```bash
bash ./scripts/eval_tum.sh            # with calibration
bash ./scripts/eval_tum.sh --no-calib # without calibration
```

There are no unit tests in this repository.

## Architecture

The system has three concurrent processes communicating through shared GPU memory:

**Main thread** (`main.py`): reads frames, runs `FrameTracker.track()` per frame, appends new keyframes to `SharedKeyframes`, queues tasks for the backend.

**Backend process** (`run_backend()` in `main.py`): consumes tasks from `states.global_optimizer_tasks`, calls `FactorGraph.add_factors()` + `solve_GN_rays/calib()`, updates poses.

**Visualization process** (`mast3r_slam/visualization.py`): reads `SharedKeyframes` and `SharedStates` for display only.

### Shared memory contract

`SharedKeyframes` (`mast3r_slam/frame.py:220`) is a fixed-size ring buffer (512 slots) of pre-allocated GPU tensors using `share_memory_()`. All inter-process keyframe data passes through this structure. `SharedStates` holds the current mode, the optimizer task queue, and the current frame for visualization.

**Critical**: `update_T_WCs()` in `SharedKeyframes` updates poses after GN optimization but does **not** set `is_dirty`. The `is_dirty` flag is only set by `__setitem__`. This is a known inconsistency.

### Data flow: MASt3R → SLAM

MASt3R is a ViT-Large encoder + cross-attention decoder. It always decodes **pairs** of frames — the 3D output for frame A depends on which frame B it's decoded against. This means:

- `mast3r_inference_mono()` decodes self→self to get `X_canon (H*W, 3)` stored per frame — these are pair-conditioned and not strictly independent
- `mast3r_match_symmetric()` decodes i→j in all four combinations and returns `X (4,H,W,3)`, `C`, `D (24-dim descriptors)`, `Q`
- Matching is geometry-first: `iter_proj` CUDA kernel projects 3D points into a ray image via Levenberg-Marquardt in 2D pixel space; descriptors only refine in a small local window afterward

### Pose representation

All poses are `lietorch.Sim3` (7-DOF: translation + quaternion + scale). Scale is explicit because MASt3R's per-pair metric outputs have scale ambiguity. When converting to SE3 for output/rendering, `as_SE3()` in `mast3r_slam/lietorch_utils.py` drops the scale component — world-frame points from `T_WC.act(X_canon)` are scale-consistent, but the stripped SE3 is also correct since scale is already baked into the point positions.

### Optimization

`FactorGraph` (`mast3r_slam/global_opt.py`) maintains all edges as growing tensors (`ii`, `jj`, `idx_ii2jj (N_edges, H*W)`). There is no marginalization or window pruning — the full graph is optimized every solve. The CUDA Gauss-Newton kernel (`gauss_newton_rays` / `gauss_newton_calib`) builds a block-sparse Hessian on GPU, transfers to CPU, and solves via Eigen `SimplicialLLT`. Two modes:
- **Uncalibrated** (`use_calib: False`): residuals are 4D ray+distance; no `K` required
- **Calibrated** (`use_calib: True`): residuals are 3D pixel+log-depth; requires `K`; `constrain_points_to_ray()` must be applied to `X_canon` before optimization

### Config system

Configs use YAML inheritance via `inherit:` key. `config/base.yaml` defines all defaults. `config/calib.yaml` inherits base and sets `use_calib: True` and `subsample: 2`. Load order matters: `load_config()` in `mast3r_slam/config.py` merges child over parent.

### Key file locations

| Purpose | File |
|---|---|
| Entry point, process spawning, main loop | `main.py` |
| Frame & shared memory data structures | `mast3r_slam/frame.py` |
| MASt3R inference wrappers | `mast3r_slam/mast3r_utils.py` |
| Tracking (per-frame pose estimation) | `mast3r_slam/tracker.py` |
| Factor graph & global optimization | `mast3r_slam/global_opt.py` |
| Feature matching (iter_proj wrapper) | `mast3r_slam/matching.py` |
| Geometric transforms + Jacobians | `mast3r_slam/geometry.py` |
| Loop closure retrieval (ASMK/IVF) | `mast3r_slam/retrieval_database.py` |
| Trajectory/PLY saving | `mast3r_slam/evaluate.py` |
| CUDA backend source | `mast3r_slam/backend/src/` |

All functions must have clear comments (usage, input, output)