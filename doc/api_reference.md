# API Reference — Online + Offline 3DGS

Primary online module: `mast3r_slam/online_gaussian_splat.py`  
Legacy offline module: `mast3r_slam/gaussian_splat.py`

---

## Online public API (`online_gaussian_splat.py`)

### `make_keyframe_snapshot(kf_idx: int, frame, use_calib: bool) -> dict`

Builds a picklable keyframe payload for the online GS queue.

Returned keys:
- `type`: `"new_kf"`
- `kf_idx`, `frame_id`
- `uimg`, `X_canon`, `C`, `N`
- `T_WC_data`
- `K` (or `None`)
- `img_shape`

Notes:
- Tensors are cloned to CPU to safely cross process boundaries.
- `C` and `N` are used to recover average confidence per pixel in the GS worker.

---

### `run_online_gs(cfg, states, keyframes, save_dir, seq_name, gs_queue) -> None`

Entry point for online mapping worker process.

Event protocol on `gs_queue`:
- `{"type": "new_kf", ...}`: insert and optimize with fresh keyframe
- `{"type": "terminate"}`: sync final poses, run fine refinement, save map

Outputs:
- `<save_dir>/<seq_name>_online_gs.ply`

Behavior:
- Stage 1 (coarse online): incremental insertion + sliding-window optimization.
- Stage 2 (fine): post-SLAM refinement using final optimized poses.

---

## Online core class (`GaussianMapper`)

### `insert_keyframe(snapshot: dict, threshold: Optional[int] = None) -> None`

Initializes and appends Gaussian primitives from one keyframe snapshot.

Key controls:
- confidence gate: `c_conf_threshold`
- per-kf cap: `max_gaussians_per_kf` (or `threshold` override)
- global cap: `max_gaussians`

---

### `train_step(n_steps, window_size, random_history) -> dict`

Runs sliding-window optimization for coarse stage and returns accumulated loss
stats (`loss`, `l_rgb`, `l_depth`, `l_iso`, `n_steps`).

Loss form:
- `L = alpha_rgb * L_rgb + (1 - alpha_rgb) * L_depth + lambda_iso * L_iso`
- Optional D-SSIM blend controlled by `lambda_ssim`.

---

### `sync_poses_from_shared(keyframes) -> None`

Refreshes renderer camera matrices from latest SLAM poses in shared memory.
Used periodically during coarse stage and before fine stage.

---

### `fine_refine(n_iters: int, log_interval: int) -> None`

Runs final high-iteration refinement after SLAM completion.
Commonly uses `fine_lambda_depth = 0.0` to avoid cross-camera Sim3 depth-scale
inconsistency effects.

---

### `save(save_dir: Path, seq_name: str) -> None`

Saves current online Gaussian state as standard 3DGS PLY:
- `<save_dir>/<seq_name>_online_gs.ply`

---

## Offline API (still available)

### `extract_gaussians(keyframes, use_calib, threshold=0.5) -> dict`

Batch extraction from final keyframes.

### `train_gaussian_splat(keyframes, K, save_dir, seq_name, use_calib, cfg, threshold=None) -> None`

Legacy offline batch optimization.
Output file:
- `<save_dir>/<seq_name>_gs.ply`

---

## Shared helper functions (offline module, reused by online)

- `_psnr`, `_ssim`
- `_prune_mask`, `_clone`, `_apply_mask`
- `_normal_to_quat_wxyz`
- `_sim3_to_w2c`
- `_save_splat`

These helpers are imported by the online mapper to avoid duplicated
implementations.

---

## Config reference (`gaussian_splat`)

### Mode switches

| Key | Default | Description |
|---|---|---|
| `online_enabled` | `false` | Enable online GS worker (recommended) |
| `enabled` | `false` | Enable offline batch GS after SLAM |

### Shared safety / optimizer controls

| Key | Default | Description |
|---|---|---|
| `c_conf_threshold` | `0.5` | Confidence gate for Gaussian initialization |
| `max_gaussians` | `2000000` | Global Gaussian cap |
| `lr_means` | `1.6e-4` | LR for means |
| `lr_opacity` | `5.0e-2` | LR for opacity |
| `lr_scales` | `5.0e-3` | LR for log-scales |
| `lr_quats` | `1.0e-3` | LR for quaternions |
| `lr_sh` | `2.5e-3` | LR for SH DC color |
| `densify_from_iter` | `500` | Start densification window |
| `densify_until_iter` | `5000` | End densification window |
| `densify_interval` | `100` | Densify every N steps |
| `opacity_reset_interval` | `3000` | Opacity reset interval |
| `prune_opacity_thresh` | `0.005` | Opacity pruning threshold |
| `grad_thresh` | `2.0e-4` | Clone trigger on image-plane gradient |

### Online-specific controls

| Key | Default | Description |
|---|---|---|
| `window_size` | `5` | Recent camera count in sliding window |
| `random_history` | `2` | Random historical camera count |
| `steps_per_keyframe` | `50` | Coarse train steps when a keyframe arrives |
| `idle_train_steps` | `5` | Coarse train steps during idle cycles |
| `max_gaussians_per_kf` | `50000` | Per-keyframe insertion cap |
| `n_iters_fine` | `2000` | Fine-stage iterations |
| `fine_log_interval` | `500` | Fine-stage log interval |
| `fine_prune_thresh` | `0.01` | Pre-fine floater pruning threshold |
| `alpha_rgb` | `0.95` | RGB-vs-depth mix in paper Eq.13 |
| `lambda_iso` | `10.0` | Isotropic scale regularization |
| `lambda_ssim` | `0.0` | D-SSIM blend weight |
| `disc_z_factor` | `0.1` | Disc initialization z-axis factor |
| `max_log_scale` | `-2.0` | Upper log-scale clamp |
| `min_log_scale` | `-7.0` | Lower log-scale clamp |
| `max_scale_ratio` | `10.0` | Needle-prune ratio cap |

### Offline-specific controls

| Key | Default | Description |
|---|---|---|
| `n_iters` | `10000` | Offline training iterations |
| `lambda_depth` | `0.1` | Depth loss weight |
| `depth_min_conf` | `0.3` | Depth loss confidence mask threshold |
