# API Reference — `mast3r_slam/gaussian_splat.py`

---

## Public functions

### `extract_gaussians(keyframes, use_calib, threshold=0.5) → dict`

Converts a fully-populated `SharedKeyframes` buffer into a flat dict of
Gaussian parameters ready for training.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `keyframes` | `SharedKeyframes` | Final SLAM keyframe buffer (post-optimisation) |
| `use_calib` | `bool` | If `True`, applies `constrain_points_to_ray` before transforming to world frame |
| `threshold` | `float` | Normalised-confidence cutoff in `(0, 1)`. Higher = fewer Gaussians. Prevents memory blowup. |

**Returns** — dict with keys:

| Key | Shape | Description |
|---|---|---|
| `means` | `(N, 3)` | World-frame Gaussian centres, `requires_grad=True` |
| `sh_dc` | `(N, 3)` | DC spherical harmonic colour coefficients |
| `log_scales` | `(N, 3)` | Log-space anisotropic scales |
| `quats` | `(N, 4)` | Unit quaternions **wxyz** convention |
| `opacity` | `(N,)` | Logit-space opacity |
| `viewmats` | `(K, 4, 4)` | World→cam matrices for each keyframe (CUDA) |
| `gt_images` | `list[(H,W,3)]` | Reference RGB images in `[0,1]` (CUDA) |
| `depth_maps` | `list[(H,W)]` | GT camera-z depth per keyframe from `X_canon[:, 2]` (CUDA) |
| `conf_maps` | `list[(H,W)]` | Normalised confidence per keyframe, used to weight depth loss (CUDA) |
| `Ks` | `(K,3,3)` or `None` | Intrinsics per keyframe; `None` when uncalibrated |

---

### `train_gaussian_splat(keyframes, K, save_dir, seq_name, use_calib, cfg, threshold=None) → None`

Full offline training loop. Calls `extract_gaussians`, runs Adam optimisation
with adaptive density control, logs PSNR/SSIM, and saves the result as a PLY.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `keyframes` | `SharedKeyframes` | Final SLAM keyframe buffer |
| `K` | `Tensor(3,3)` or `None` | Global camera intrinsics (uncalibrated: `None`) |
| `save_dir` | `Path` | Output directory |
| `seq_name` | `str` | Filename stem; output is `<save_dir>/<seq_name>_gs.ply` |
| `use_calib` | `bool` | Passed through to `extract_gaussians` |
| `cfg` | `dict` | `gaussian_splat` config block from YAML |
| `threshold` | `int` or `None` | Hard cap on Gaussian count; overrides `cfg["max_gaussians"]` when set |

---

## Private helpers

### Metric helpers

#### `_psnr(render, gt) → float`
Peak Signal-to-Noise Ratio in dB. Both inputs `(H, W, 3)` float in `[0, 1]`.

#### `_ssim(render, gt, window_size=11) → Tensor (scalar)`
Structural Similarity Index using an 11×11 Gaussian window (σ=1.5).
Returns a **differentiable tensor** so it can be used directly in the loss
(`L_D-SSIM = (1 − _ssim(...)) / 2`). Call `.item()` when logging.
Implemented with `F.conv2d` — no external metric library needed.

---

### Density control

#### `_prune_mask(data, threshold=0.005) → BoolTensor(N,)`
Returns `True` for Gaussians whose sigmoid opacity is below `threshold`.

#### `_clone(data, optimizer, mask, current_max, threshold) → None`
Clones Gaussians selected by `mask` and appends them to `data` and the
Adam state. Respects the hard cap `threshold`. Adds small position jitter so
clones don't overlap originals.

#### `_apply_mask(data, optimizer, keep) → None`
Compacts all per-Gaussian tensors and Adam state to the indices in `keep`.
Used for pruning.

#### `_index_data(data, idx) → dict`
Returns a copy of `data` with only the Gaussians at `idx`. Detaches and
re-sets `requires_grad=True` so Adam starts fresh on the subset.

---

### Geometry helpers

#### `_normal_to_quat_wxyz(normals) → Tensor(N, 4)`
Converts `(N, 3)` unit normals to wxyz quaternions that align the canonical
z-axis with the normal direction. Handles degenerate ±z cases.

#### `_sim3_to_w2c(T_WC) → Tensor(4, 4)`
Converts a `lietorch.Sim3` camera-to-world pose to a 4×4 world-to-cam matrix.
Uses `as_SE3` to strip scale (safe because world-frame points already carry scale).

#### `_quat_xyzw_to_matrix(q) → Tensor(3, 3)`
Converts an xyzw quaternion (lietorch convention) to a 3×3 rotation matrix.

---

### Output

#### `_save_splat(path, data) → None`
Writes a standard 3DGS PLY file compatible with SuperSplat and other viewers.
Field layout: `x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`.

---

## Config reference (`gaussian_splat` block in YAML)

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable GS reconstruction (requires `--save-as`) |
| `c_conf_threshold` | `0.5` | Confidence gate for Gaussian creation |
| `max_gaussians` | `2000000` | Hard cap — primary OOM guard |
| `n_iters` | `10000` | Training iterations |
| `lr_means` | `1.6e-4` | Adam LR for Gaussian centres |
| `lr_opacity` | `5.0e-2` | Adam LR for opacity |
| `lr_scales` | `5.0e-3` | Adam LR for scales |
| `lr_quats` | `1.0e-3` | Adam LR for rotations |
| `lr_sh` | `2.5e-3` | Adam LR for DC colour |
| `densify_from_iter` | `500` | Start adaptive density control after this iter |
| `densify_until_iter` | `5000` | Stop adaptive density control after this iter |
| `densify_interval` | `100` | Run density control every N iters |
| `opacity_reset_interval` | `3000` | Reset all opacities to ~0.12 every N iters |
| `prune_opacity_thresh` | `0.005` | Prune Gaussians below this opacity |
| `grad_thresh` | `2.0e-4` | Clone Gaussians above this abs-2D-gradient |
| `lambda_ssim` | `0.2` | D-SSIM loss weight; `0` = pure L1 |
| `lambda_depth` | `0.1` | Log-depth loss weight; `0` = disabled |
| `depth_min_conf` | `0.3` | Mask depth loss where MASt3R confidence is below this |
