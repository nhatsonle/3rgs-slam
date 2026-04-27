# API Reference — Gaussian Splatting Modules

Two files implement the GS integration:
- `mast3r_slam/gaussian_splat.py` — offline batch mode + `OnlineGaussianSplat`
- `mast3r_slam/gaussian_map.py` — `GlobalGaussianMap` and supporting classes

---

## `gaussian_splat.py` — Public functions

### `extract_gaussians(keyframes, use_calib, threshold=0.5) → dict`

Converts a fully-populated `SharedKeyframes` buffer into a flat dict of
Gaussian parameters ready for training.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `keyframes` | `SharedKeyframes` | Final SLAM keyframe buffer (post-optimisation) |
| `use_calib` | `bool` | If `True`, applies `constrain_points_to_ray` before transforming to world frame |
| `threshold` | `float` | Normalised-confidence cutoff in `(0, 1)`. Higher = fewer Gaussians. |

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
| `depth_maps` | `list[(H,W)]` | GT camera-z depth per keyframe from `X_canon[:, 2]` |
| `conf_maps` | `list[(H,W)]` | Normalised confidence per keyframe (weights depth loss) |
| `Ks` | `(K,3,3)` or `None` | Intrinsics per keyframe; `None` when uncalibrated |

---

### `train_gaussian_splat(keyframes, K, save_dir, seq_name, use_calib, cfg, threshold=None) → None`

Offline training loop. Calls `extract_gaussians`, runs Adam optimisation with
adaptive density control, logs PSNR/SSIM, and saves as PLY.

| Name | Type | Description |
|---|---|---|
| `keyframes` | `SharedKeyframes` | Final SLAM keyframe buffer |
| `K` | `Tensor(3,3)` or `None` | Global camera intrinsics (`None` when uncalibrated) |
| `save_dir` | `Path` | Output directory |
| `seq_name` | `str` | Filename stem; output is `<save_dir>/<seq_name>_gs.ply` |
| `use_calib` | `bool` | Passed through to `extract_gaussians` |
| `cfg` | `dict` | `gaussian_splat` config block from YAML |
| `threshold` | `int` or `None` | Hard cap on Gaussian count |

---

### `class OnlineGaussianSplat`

Incremental per-keyframe Gaussian map. Used when `online: true, use_global_map: false`.

| Method | Signature | Description |
|---|---|---|
| `add_keyframe` | `(kf, threshold=None) → bool` | Extract Gaussians from one KF; append to live map |
| `sync_poses` | `(keyframes) → None` | Overwrite viewmats from latest GN poses (non-differentiable) |
| `train_gaussians` | `(n, threshold=None) → None` | N Adam steps with L1+D-SSIM+log-depth loss; density control every 100 steps |
| `refine_poses` | `(n, threshold=None) → None` | N photometric steps on viewmats (Gaussians frozen, KF-0 pinned) |
| `add_aux_frames` | `(frames_data: list) → None` | Buffer non-KF frames for L1-only supervision |
| `save` | `(save_dir, seq_name) → None` | Write trained map as `<seq>_gs.ply` |

---

## `gaussian_map.py` — Classes

### `MappingFrameSelector`

Decides which non-keyframe frames enter the mapping pipeline.

```python
selector = MappingFrameSelector(cfg_gs)
# Inside main tracking loop (non-KF frames only):
if selector.should_map(frame, last_kf_t):
    mapping_queue.put_nowait(serialised_frame_data)
# On new KF:
selector.reset()
```

| Method | Signature | Description |
|---|---|---|
| `should_map` | `(frame, last_kf_t) → bool` | Returns `True` if frame passes all three gates (cooldown, confidence, baseline). Increments internal counter on every call. |
| `reset` | `() → None` | Reset cooldown counter; call when a new keyframe is added |

Config keys: `map_cooldown_frames`, `map_conf_thresh`, `map_baseline_thresh`.

---

### `LocalFusionBuffer`

Sliding window of mapping frames. Triggers multi-frame geometry fusion.

```python
buf = LocalFusionBuffer(window_size=7, min_frames=2)
buf.add(frame_data_dict)
if buf.ready():
    fused = buf.fuse()   # returns None if no valid points
```

| Method | Signature | Description |
|---|---|---|
| `add` | `(frame_data: dict) → None` | Append one mapping frame; oldest auto-discarded when full |
| `ready` | `() → bool` | True when buffer holds ≥ `min_frames` entries |
| `fuse` | `() → dict or None` | Concatenate all valid points across the window into one point cloud |
| `clear` | `() → None` | Empty the buffer |

**`fuse()` output keys** (all CPU tensors):

| Key | Shape | Description |
|---|---|---|
| `means` | `(M, 3)` | World-frame 3D points (all frames concatenated) |
| `rgb` | `(M, 3)` | Colours in `[0, 1]` |
| `conf` | `(M,)` | Normalised confidence |
| `depth` | `(M,)` | Camera-frame z depth |
| `viewmats` | `(W, 4, 4)` | World→cam per buffered frame |
| `gt_images` | `list[W]` | Reference RGB per frame |
| `Ks` | `(W, 3, 3)` or `None` | Intrinsics per frame |
| `H_img, W_img` | `int` | Image resolution |
| `n_frames` | `int` | Number of frames in this fused batch |

---

### `VoxelAssociation`

Spatial hash for O(1) nearest-Gaussian lookup.

```python
assoc = VoxelAssociation(voxel_size=0.05)
assoc.rebuild(existing_means)                          # O(M)
match_idx, dist = assoc.query_batch(candidate_means)  # O(27) per candidate
```

| Method | Signature | Description |
|---|---|---|
| `rebuild` | `(means: Tensor(M,3)) → None` | Build voxel grid from current Gaussian positions |
| `query_batch` | `(new_means: Tensor(N,3)) → (match_idx(N,), dist(N,))` | For each candidate, return index and distance of nearest existing Gaussian (`-1` / `inf` if none) |
| `is_empty` | `() → bool` | True before first `rebuild` or after rebuild on empty map |

---

### `class GlobalGaussianMap`

Drop-in replacement for `OnlineGaussianSplat`. Activates when
`gaussian_splat.use_global_map: true`.

```python
gs = GlobalGaussianMap(cfg_gs, use_calib=False, K=None, device="cuda:0")
```

**Constructor parameters**

| Name | Type | Description |
|---|---|---|
| `cfg` | `dict` | `gaussian_splat` block from loaded config |
| `use_calib` | `bool` | Whether calibrated mode is active |
| `K` | `Tensor(3,3)` or `None` | Global camera intrinsics |
| `device` | `str` | CUDA device string |

**Public methods**

| Method | Signature | Description |
|---|---|---|
| `add_keyframe` | `(kf, threshold=None) → bool` | Alias for `add_keyframe_view`; for API compatibility with `OnlineGaussianSplat` callers |
| `add_keyframe_view` | `(kf) → None` | Register KF as render target (viewmat + GT image + depth + conf). **Does not init Gaussians.** Triggers Step 12 cleanup every `cleanup_interval_kf` calls. |
| `add_mapping_frame` | `(frame_data: dict) → None` | Steps 3–7 pipeline: unpack → fusion buffer → fuse → candidates → associate → merge/spawn |
| `sync_poses` | `(keyframes) → None` | Overwrite all viewmats from latest GN-optimised `T_WC` (non-differentiable) |
| `capture_poses` | `(keyframes) → list[Tensor(4,4)]` | Snapshot world→cam matrices for all registered KFs; call **before** a GN solve for re-anchor |
| `reanchor` | `(keyframes, old_poses) → None` | Step 11: propagate loop-closure pose corrections to all Gaussian positions |
| `train_gaussians` | `(n_steps, threshold=None) → None` | Step 9: N Adam steps with hybrid loss; density control every 100 steps |
| `add_aux_frames` | `(frames_data: list) → None` | Buffer non-KF frames for L1-only photometric supervision |
| `cleanup` | `() → None` | Step 12: prune low-opacity + single-observation Gaussians |
| `save` | `(save_dir, seq_name) → None` | Write map as `<save_dir>/<seq_name>_gs.ply` |
| `refine_poses` | `(n_steps, threshold=None) → None` | Stub (no-op); kept for interface parity |

---

## `gaussian_splat.py` — Private helpers

### Metric helpers

#### `_psnr(render, gt) → float`
Peak Signal-to-Noise Ratio in dB. Both inputs `(H, W, 3)` float in `[0, 1]`.

#### `_ssim(render, gt, window_size=11) → Tensor (scalar)`
Structural Similarity Index, 11×11 Gaussian window (σ=1.5). Returns a
**differentiable tensor** so it can be used in the loss. Call `.item()` for
logging. Implemented with `F.conv2d` — no external metric library needed.

---

### Density control

#### `_prune_mask(data, threshold=0.005) → BoolTensor(N,)`
Returns `True` for Gaussians whose sigmoid opacity is below `threshold`.

#### `_clone(data, optimizer, mask, current_max, threshold) → None`
Clones Gaussians selected by `mask`; appends to `data` and Adam state.
Adds small position jitter so clones don't overlap originals. Respects the
hard cap `threshold`.

#### `_apply_mask(data, optimizer, keep) → None`
Compacts all per-Gaussian tensors and Adam `exp_avg / exp_avg_sq` state to
the indices in `keep`. Used for pruning.

#### `_index_data(data, idx) → dict`
Returns a copy of `data` with only the Gaussians at `idx`. Detaches and
re-sets `requires_grad=True` so Adam starts fresh on the subset.

---

### Geometry helpers

#### `_normal_to_quat_wxyz(normals) → Tensor(N, 4)`
Converts `(N, 3)` unit normals to wxyz quaternions aligning the canonical
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

## Config reference — full `gaussian_splat` block

### Base training (offline + both online modes)

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable GS reconstruction (requires `--save-as`) |
| `c_conf_threshold` | `0.5` | Confidence gate for Gaussian creation |
| `max_gaussians` | `2000000` | Hard cap — primary OOM guard |
| `n_iters` | `10000` | Training iterations (offline mode only) |
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
| `depth_min_conf` | `0.3` | Mask depth loss where confidence is below this |

### Online incremental (both `OnlineGaussianSplat` and `GlobalGaussianMap`)

| Key | Default | Description |
|---|---|---|
| `online` | `false` | Enable incremental mode |
| `online_iters_per_kf` | `50` | Adam steps per GN solve cycle |
| `densify_online` | `true` | Density control during online training |
| `pose_refine` | `false` | Photometric viewmat refinement after each KF (experimental) |
| `pose_refine_iters` | `10` | Steps of pose-only optimisation per KF |
| `lr_pose` | `1.0e-4` | Adam LR for 4×4 viewmat refinement |
| `online_finalize_iters` | `0` | Extra steps after SLAM ends; `0` = save immediately |
| `aux_ratio` | `0.3` | Fraction of training steps sampling an aux non-KF frame |
| `aux_max_frames` | `200` | Max aux frames held in buffer (oldest discarded) |
| `aux_blur_thresh` | `50.0` | Laplacian variance below this → skip (too blurry) |
| `aux_novelty_min` | `0.05` | Min translation distance to last KF (skip near-duplicates) |
| `aux_max_age` | `8` | Frames elapsed since last KF beyond which frames are skipped |
| `aux_decay` | `0.95` | Temporal weight: `weight = decay^frame_age` |

### GlobalGaussianMap only (`use_global_map: true`)

| Key | Default | Description |
|---|---|---|
| `use_global_map` | `false` | Activate `GlobalGaussianMap` instead of `OnlineGaussianSplat` |
| `map_baseline_thresh` | `0.02` | Min translation from last KF (metres) to be a mapping frame |
| `map_conf_thresh` | `0.20` | Min average MASt3R confidence |
| `map_cooldown_frames` | `3` | Min non-KF frames between consecutive mapping frames |
| `map_window_size` | `7` | Sliding fusion window depth |
| `map_min_frames` | `2` | Frames required before fusion triggers |
| `assoc_thresh` | `0.05` | 3D distance threshold: merge if closer, spawn otherwise |
| `assoc_voxel` | `0.05` | Voxel cell size for spatial hash |
| `assoc_max_check` | `5000` | Max candidate Gaussians per association pass |
| `merge_ema` | `0.30` | EMA alpha for SH colour update during merge |
| `lambda_reg` | `0.01` | Scale + opacity regulariser weight (`0` to disable) |
| `reanchor_on_loop_closure` | `true` | Propagate pose corrections to Gaussians on RELOC success |
| `cleanup_interval_kf` | `10` | Run cleanup every N keyframes |
