# MASt3R-SLAM Codebase Changes — Overview

This document tracks the 3DGS transition from offline-only mapping to online
incremental mapping. For implementation details, see
`gaussian_splatting_integration.md` and `api_reference.md`.

---

## Current mapping modes

- **Online 3DGS (primary):** `gaussian_splat.online_enabled: true`
  - Runs in a dedicated worker process during SLAM.
  - Consumes keyframes as events and trains continuously with a sliding window.
  - Performs final fine refinement only after SLAM backend finishes.
- **Offline 3DGS (optional fallback):** `gaussian_splat.enabled: true`
  - Legacy batch training after SLAM ends.
  - Kept for comparison and ablation.

---

## Files changed for online migration

| File | Type | Purpose |
|---|---|---|
| `mast3r_slam/online_gaussian_splat.py` | **New** | Online GS worker, keyframe snapshots, incremental mapper, fine stage |
| `main.py` | Modified | Starts online GS process, publishes keyframe events, sends terminate event |
| `config/base.yaml` | Modified | Adds online GS controls and defaults under `gaussian_splat` |
| `config/demo_gs.yaml` | Modified | Demo profile tuned for online GS |
| `config/eval_gs.yaml` | Modified | Evaluation profile for online GS benchmarks |
| `scripts/eval_gs_psnr.py` | Modified | Selects final PLY, parses TUM/repo intrinsics, supports explicit `--ply` |
| `mast3r_slam/gaussian_splat.py` | Existing | Offline batch GS path retained (optional) |
| `mast3r_slam/backend/src/gn_kernels.cu` | Modified | Build compatibility fix for newer PyTorch |

---

## Runtime flow changes (`main.py`)

1. Detect online mode:
   - `dataset.save_results` is true (`--save-as` used)
   - `config["gaussian_splat"]["online_enabled"]` is true
2. Spawn online worker:
   - `mp.Process(target=run_online_gs, ...)`
3. On each new keyframe:
   - `make_keyframe_snapshot(...)`
   - `gs_queue.put(snapshot_event)`
4. After SLAM loop and `backend.join()`:
   - `gs_queue.put({"type": "terminate"})`
   - online worker syncs final optimized poses
   - if `rebuild_from_final_keyframes: true`, it rebuilds Gaussian geometry
     from final `SharedKeyframes` before fine refinement
   - it runs fine refinement
5. Worker writes:
   - `<save_dir>/<seq_name>_online_gs.ply`

Offline trigger in `main.py` is still available and independent:
- `gaussian_splat.enabled: true` calls `train_gaussian_splat(...)` after SLAM.

---

## Config migration summary

`gaussian_splat` now has two control switches:

- `online_enabled`: enable online MASt3R-GS style mapping (recommended)
- `enabled`: enable legacy offline batch mapper

Important online fields:

- `window_size`, `random_history`
- `steps_per_keyframe`, `idle_train_steps`
- `max_gaussians`, `max_gaussians_per_kf`
- `rebuild_from_final_keyframes`, `final_max_gaussians_per_kf`, `save_rebuild_init`
- `n_iters_fine`, `fine_prune_thresh`, `fine_log_interval`
- `alpha_rgb`, `lambda_iso`, `lambda_depth`, `lambda_ssim`
- `disc_z_factor`, `max_log_scale`, `min_log_scale`, `max_scale_ratio`

---

## Safety / memory controls

- Confidence gate: `c_conf_threshold`
- Global cap: `max_gaussians`
- Per-keyframe insertion cap: `max_gaussians_per_kf`
- Scale sanity + needle pruning: `max_log_scale`, `min_log_scale`, `max_scale_ratio`

These controls are the main protection against Gaussian explosion in long
sequences.

---

## Quick run commands

Demo (online mode):

```bash
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml --no-viz --save-as gs_demo
```

Eval profile (online mode):

```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room --config config/eval_gs.yaml --no-viz --save-as gs_eval
```

---

## Extra dependency

```bash
pip install gsplat
pip install plyfile
```
