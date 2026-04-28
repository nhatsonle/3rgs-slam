# MASt3R-SLAM Codebase Changes — Overview

This document summarises every modification made to the original MASt3R-SLAM codebase.
For details on the Gaussian Splatting integration see the sibling docs:
- [gaussian_splatting_integration.md](gaussian_splatting_integration.md) — offline batch mode
- [online_gs_integration.md](online_gs_integration.md) — online/incremental mode + GlobalGaussianMap

---

## Files changed

| File | Type | Purpose |
|---|---|---|
| `mast3r_slam/gaussian_splat.py` | New | Offline 3DGS pipeline + `OnlineGaussianSplat` class |
| `mast3r_slam/gaussian_map.py` | **New** | `GlobalGaussianMap` — 12-step multi-view incremental map |
| `config/demo_gs.yaml` | New/Modified | Demo config; now enables `GlobalGaussianMap` by default |
| `config/base.yaml` | Modified | Extended `gaussian_splat` block (online + GlobalGaussianMap params) |
| `main.py` | Modified | Backend selection, `MappingFrameSelector`, `mapping_queue`, finalize drain |
| `mast3r_slam/backend/src/gn_kernels.cu` | Modified | Build fix for `torch::linalg_norm` API change |
| `CLAUDE.md` | New | Developer guide / AI coding instructions |

---

## 1. `mast3r_slam/gaussian_splat.py` (new, ~970 lines)

The offline 3DGS module and the original online incremental class.

### Public functions

#### `extract_gaussians(keyframes, use_calib, threshold=0.5)`
Converts the final SLAM state into Gaussian parameters.

#### `train_gaussian_splat(keyframes, K, save_dir, seq_name, use_calib, cfg, threshold=None)`
Full offline Adam training loop with adaptive density control and a combined
photometric + structural + geometric loss (L1 + D-SSIM + depth). Used when
`online: false`.

#### `class OnlineGaussianSplat`
Incremental per-keyframe Gaussian extraction and training. Superseded by
`GlobalGaussianMap` for better quality but preserved for backward compatibility.

### Private helpers
`_psnr`, `_ssim`, `_prune_mask`, `_clone`, `_apply_mask`, `_index_data`,
`_normal_to_quat_wxyz`, `_sim3_to_w2c`, `_quat_xyzw_to_matrix`, `_save_splat`

---

## 2. `mast3r_slam/gaussian_map.py` (new, ~1050 lines)

Implements the 12-step GlobalGaussianMap integration plan. See
[online_gs_integration.md](online_gs_integration.md) for the full design.

### Classes

#### `MappingFrameSelector`
Decides which non-keyframe frames enter the mapping pipeline based on baseline,
confidence, and cooldown. Runs in the **main process**.

#### `LocalFusionBuffer`
Sliding-window buffer (default 7 frames) that fuses geometry from multiple
viewpoints before Gaussian candidate generation.

#### `VoxelAssociation`
Python-dict spatial hash (5 cm voxels) that checks 27-cell neighbourhoods to
prevent object duplication when spawning new Gaussians.

#### `GlobalGaussianMap`
Drop-in replacement for `OnlineGaussianSplat`. Activates when
`gaussian_splat.use_global_map: true`. Implements Steps 2–12:
- Step 2 — mapping frame selection (non-KF frames only)
- Step 3 — local sliding-window fusion
- Step 4 — Gaussian candidate generation (confidence-weighted, isotropic init)
- Steps 6–7 — association engine: merge close Gaussians, spawn new ones
- Step 8 — density control (prune + clone from abs-2D-gradient)
- Step 9 — hybrid loss (L1 + D-SSIM + log-depth + regulariser)
- Step 11 — loop-closure re-anchor (propagates per-KF pose corrections)
- Step 12 — periodic cleanup (prune dead + single-observation Gaussians)

---

## 3. `config/base.yaml` — extended `gaussian_splat` block

The block now has three sections: base training, online incremental, and
GlobalGaussianMap parameters.

```yaml
gaussian_splat:
  enabled: false
  # ... base training params unchanged ...
  online: false
  online_iters_per_kf: 50
  online_finalize_iters: 0
  aux_ratio: 0.3
  aux_max_frames: 200
  # ── GlobalGaussianMap (use_global_map: true) ──
  use_global_map: false
  map_baseline_thresh: 0.02
  map_conf_thresh: 0.20
  map_cooldown_frames: 3
  map_window_size: 7
  map_min_frames: 2
  assoc_thresh: 0.05
  assoc_voxel: 0.05
  assoc_max_check: 5000
  merge_ema: 0.30
  lambda_reg: 0.01
  reanchor_on_loop_closure: true
  cleanup_interval_kf: 10
```

All existing configs are unaffected (`enabled: false`, `use_global_map: false` by default).

---

## 4. `config/demo_gs.yaml` (updated)

Now enables `GlobalGaussianMap` with a fast demo profile:
- `subsample: 10` → ~120 frames from the 1207-frame video
- `online_iters_per_kf: 20` — keeps the backend from falling behind during SLAM
- `online_finalize_iters: 2000` — bulk training after all KFs are registered
- `use_global_map: true` — activates the 12-step pipeline

```bash
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml \
  --no-viz --save-as results
# Output: logs/results/IMG_2520_gs.ply  (separate from SLAM trajectory PLY)
```

---

## 5. `main.py` — major updates

| Change | Purpose |
|---|---|
| `_drain_queue(q)` helper | Atomically drains a `mp.Queue` into a list |
| `run_backend` adds `mapping_queue=None` parameter | Receives mapping frames from main |
| Backend selects `GlobalGaussianMap` vs `OnlineGaussianSplat` via `use_global_map` | Config-driven module switch |
| Drain `mapping_queue` inside GN-solve loop (before `sync_poses`) | Multi-view fusion runs at correct time |
| RELOC mode: `capture_poses` before GN, `reanchor` after success | Loop-closure pose corrections propagate to Gaussians |
| Finalize: drain remaining KF tasks + `mapping_queue` before save | Prevents data loss when backend lags behind main |
| Print save path after `backend.join()` | User knows which file to open |
| Main creates `MappingFrameSelector` + `mapping_queue` when `use_global_map=True` | Sends non-KF frames to backend |
| Main enqueues qualifying non-KF frames via `mapping_queue.put_nowait()` | Never blocks tracking thread |

---

## 6. `mast3r_slam/backend/src/gn_kernels.cu` — build fix

PyTorch's C++ API changed the signature of `torch::linalg::linalg_norm` between
releases, breaking compilation. Replaced with a numerically equivalent manual
computation in three places (gauss_newton_points, gauss_newton_rays,
gauss_newton_calib):

```cpp
// Before (broken on newer PyTorch):
delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});

// After:
delta_norm = at::sqrt((dx * dx).sum());
```

---

## 7. Tracking ↔ Mapping synchronization fixes

Eight issues were identified and fixed to improve 3DGS map quality:

| # | Issue | Fix | Files |
|---|---|---|---|
| 1 | `GlobalGaussianMap.train_gaussians` lacked explicit `grad_enabled(True)` guard | Added `torch.set_grad_enabled(True)` at start of training | `gaussian_map.py` |
| 2 | Mapping frame poses were stale tracking snapshots | Re-compute world-frame points using GN-optimised nearest-KF pose via relative transform | `gaussian_map.py`, `main.py` |
| 3 | Backend processed one KF task per loop; fell behind main thread | Batch-drain all pending tasks, one GN solve per batch, register all KF views | `main.py` |
| 4 | Isotropic Gaussian init (spheres) in GlobalGaussianMap | KNN-based anisotropic scales + surface-normal quaternions from local covariance | `gaussian_map.py` |
| 5 | Per-KF confidence normalization made thresholds inconsistent | Running global confidence max across all KFs/mapping frames | `gaussian_map.py` |
| 6 | Depth GT (camera z) vs rendered depth (scaled by Sim3 s) mismatch | Multiply `depth_gt` by Sim3 scale to match rendered depth frame | `gaussian_map.py`, `gaussian_splat.py` |
| 7 | Insufficient online training budget | Increased defaults; training steps scale with batch size | `base.yaml`, `demo_gs.yaml`, `main.py` |
| 8 | VoxelAssociation was pure-Python O(N×27×M) loop | Replaced with chunked GPU `cdist` nearest-neighbour | `gaussian_map.py` |

Main thread now enqueues `kf_sim3_data` (the nearest KF's Sim3 pose at enqueue time)
alongside each mapping frame, enabling the backend to compute the correct relative
transform and apply it with the GN-optimised KF pose.

---

## Extra dependencies

```bash
pip install gsplat        # 3DGS rasterizer (gsplat 1.5.3 used during development)
pip install plyfile        # PLY output (usually already present)
```
