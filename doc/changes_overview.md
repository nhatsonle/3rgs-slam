# MASt3R-SLAM Codebase Changes — Overview

This document summarises every modification made to the original MASt3R-SLAM codebase.
For a deep-dive on the Gaussian Splatting integration specifically, see
[gaussian_splatting_integration.md](gaussian_splatting_integration.md).

---

## Files changed

| File | Type | Purpose |
|---|---|---|
| `mast3r_slam/gaussian_splat.py` | **New** | Entire offline 3DGS pipeline |
| `config/demo_gs.yaml` | **New** | Demo config for quick PoC run |
| `config/base.yaml` | Modified | Added `gaussian_splat` config block |
| `main.py` | Modified | Calls GS training after SLAM finishes |
| `mast3r_slam/backend/src/gn_kernels.cu` | Modified | Build fix for `torch::linalg_norm` API change |
| `CLAUDE.md` | New | Developer guide / AI coding instructions |

---

## 1. `mast3r_slam/gaussian_splat.py` (new, ~530 lines)

The entire offline 3DGS module. Three public functions:

### `extract_gaussians(keyframes, use_calib, threshold=0.5)`
Converts the final SLAM state into Gaussian parameters (see
[gaussian_splatting_integration.md](gaussian_splatting_integration.md) for details).

### `train_gaussian_splat(keyframes, K, save_dir, seq_name, use_calib, cfg, threshold=None)`
Full Adam-based 3DGS training loop with adaptive density control and a combined
photometric + structural + geometric loss (L1 + D-SSIM + depth).

### Private helpers
`_psnr`, `_ssim`, `_prune_mask`, `_clone`, `_apply_mask`, `_index_data`,
`_normal_to_quat_wxyz`, `_sim3_to_w2c`, `_quat_xyzw_to_matrix`, `_save_splat`

---

## 2. `config/base.yaml` — added `gaussian_splat` block

```yaml
gaussian_splat:
  enabled: false           # off by default; enable in child configs
  c_conf_threshold: 0.5    # MASt3R confidence gate for Gaussian creation
  max_gaussians: 2000000   # hard memory cap
  n_iters: 10000
  lr_means: 1.6e-4
  lr_opacity: 5.0e-2
  lr_scales: 5.0e-3
  lr_quats: 1.0e-3
  lr_sh: 2.5e-3
  densify_from_iter: 500
  densify_until_iter: 5000
  densify_interval: 100
  opacity_reset_interval: 3000
  prune_opacity_thresh: 0.005
  grad_thresh: 2.0e-4
  # Loss weights
  lambda_ssim: 0.2       # weight of D-SSIM term; 0 = pure L1
  lambda_depth: 0.1      # weight of log-depth loss; 0 = disabled
  depth_min_conf: 0.3    # ignore depth loss where MASt3R confidence is below this
```

All existing configs are unaffected because `enabled: false` by default.

---

## 3. `config/demo_gs.yaml` (new)

Inherits `base.yaml`; enables GS with a fast demo profile:
- `subsample: 10` → ~15 keyframes from a 1200-frame video
- `max_gaussians: 500000`, `n_iters: 3000`

Run:
```bash
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml --no-viz --save-as gs_poc
```

---

## 4. `main.py` — 5-line addition

After `eval.save_keyframes(...)` at line 324:

```python
if config.get("gaussian_splat", {}).get("enabled", False):
    from mast3r_slam.gaussian_splat import train_gaussian_splat
    train_gaussian_splat(
        keyframes, K, save_dir, seq_name, config["use_calib"], config["gaussian_splat"]
    )
```

Triggers only when `save_results=True` (i.e. `--save-as` is passed) and
`gaussian_splat.enabled: true`. No impact on normal SLAM runs.

---

## 5. `mast3r_slam/backend/src/gn_kernels.cu` — build fix

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

## Extra dependency

```bash
pip install gsplat        # 3DGS rasterizer (gsplat 1.5.3 used during development)
pip install plyfile        # PLY output (usually already present)
```
