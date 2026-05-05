# View-Tied Gaussian Mapping

This document describes the `view_tied` online mapper implemented in
`mast3r_slam/view_tied_gaussian_splat.py`.

The mapper is inspired by VTGaussian-SLAM, but v1 is intentionally scoped as a
mapping-only backend for the current MASt3R-SLAM pipeline:

- MASt3R-SLAM still owns tracking, keyframe selection, loop retrieval, and global pose optimization.
- The mapper consumes the same keyframe snapshots as the standard online GS worker.
- Depth is derived from MASt3R `X_canon` pointmaps and confidence, not raw RGB-D depth.
- Output includes both native view-tied state and a standard PLY for SuperSplat/evaluation.

---

## How to run

Use the VT evaluation profile:

```bash
python main.py \
  --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
  --config config/eval_vtgs.yaml
```

The profile inherits the calibrated GS evaluation settings and switches:

```yaml
gaussian_splat:
  online_enabled: true
  mapper_type: view_tied
```

The default mapper remains:

```yaml
gaussian_splat:
  mapper_type: standard
```

so existing standard online GS runs are unchanged unless `mapper_type` is set
to `view_tied`.

---

## Output files

After the run finishes, the VT worker writes:

```text
<save_dir>/<seq_name>_vtgs.pt
<save_dir>/<seq_name>_vtgs_online_gs.ply
```

Use the files as follows:

- `<seq_name>_vtgs_online_gs.ply`
  - Open this in SuperSplat.
  - Use this with `scripts/eval_gs_psnr.py`.
  - This is a materialized standard 3DGS map.

- `<seq_name>_vtgs.pt`
  - Native debug/checkpoint state for the VT mapper.
  - Stores sections, anchors, per-Gaussian params, and source keyframe ids.
  - SuperSplat cannot open this file directly.

Evaluate explicitly with:

```bash
python scripts/eval_gs_psnr.py logs/<run_dir> \
  --device cuda \
  --ply logs/<run_dir>/<seq_name>_vtgs_online_gs.ply
```

Passing `--ply` is recommended when a run directory may contain older standard
GS outputs such as `<seq_name>_online_gs.ply`.

---

## Architecture

```text
main.py
  |
  +-- MASt3R-SLAM frontend/backend
  |     - tracks frames
  |     - selects keyframes
  |     - updates SharedKeyframes
  |     - globally optimizes keyframe poses
  |
  +-- run_online_vtgs worker process
        - receives keyframe snapshots through gs_queue
        - builds view-tied sections
        - trains only current-section VT params
        - periodically syncs poses from SharedKeyframes
        - exports native .pt + materialized .ply on terminate
```

`main.py` chooses the worker from config:

- `mapper_type: standard` -> `run_online_gs`
- `mapper_type: view_tied` -> `run_online_vtgs`

Both workers use the same queue protocol:

- `{"type": "new_kf", ...}` inserts a keyframe.
- `{"type": "terminate"}` syncs final poses, runs fine refinement, and saves output.

---

## End-to-end mapping flow

The VT mapper is a second process. It does not block the SLAM backend except for
normal GPU contention.

### 1. Main process creates a snapshot

When MASt3R-SLAM adds a keyframe, `main.py` reads it back from
`SharedKeyframes` and sends:

```python
gs_queue.put(make_keyframe_snapshot(kf_idx, keyframes[kf_idx], use_calib))
```

The snapshot is CPU-cloned so it can safely cross process boundaries.

It contains:

```text
uimg       RGB image, H x W x 3
X_canon    MASt3R pointmap, H*W x 3, source camera coordinates
C, N       confidence accumulator and update count
T_WC_data  source camera pose as lietorch Sim3 data
K          intrinsics, if calibrated mode is active
img_shape  H, W
```

The VT worker does not call MASt3R inference. It only consumes this snapshot.

### 2. Worker receives `new_kf`

`run_online_vtgs(...)` waits on `gs_queue`.

On `new_kf`, it calls:

```python
mapper.insert_keyframe(event)
```

Inside `insert_keyframe`:

1. `_snapshot_to_maps(...)` converts the snapshot into mapper tensors:
   - optional calibrated ray constraint on `X_canon`
   - normalized confidence map
   - camera-z depth map from `X_canon.z`
   - world-to-camera view matrix from `T_WC`
   - per-keyframe render target RGB/depth/confidence
2. The mapper records the pose in `kf_pose_data[kf_idx]`.
3. It decides whether this keyframe starts a new section.
4. It inserts head-frame or regular-frame Gaussians.
5. It rebuilds the optimizer for the current section.

After insertion, the worker immediately runs:

```python
mapper.train_step(vt_steps_head or vt_steps_regular)
```

### 3. Idle training

When no queue event is available, the worker does not sleep uselessly. If the
map already has keyframes and SLAM is not paused, it runs:

```python
mapper.train_step(vt_steps_idle)
```

Every `vt_sync_every_idle` idle cycles, it also calls:

```python
mapper.sync_poses_from_shared(keyframes)
```

This keeps the renderer close to backend-updated poses while SLAM is still
running.

### 4. Section-local optimization

`train_step(...)` always optimizes the current section only.

For each gradient step:

1. Pick one target camera from the current section.
2. Build render context:
   - current section
   - previous section
   - best overlapping older section, if one is found
3. Materialize view-tied Gaussians into standard rasterizer tensors.
4. Render RGB, depth, and alpha with `gsplat.rasterization`.
5. Compute RGB / SSIM / depth losses.
6. Backprop only into current section:
   - `sh_dc`
   - `log_radius`
   - `opacities`
7. Clamp radius and opacity to stable bounds.

Anchors and poses are not optimized by the VT mapper.

### 5. Terminate and final map export

When the main process finishes SLAM, it waits for backend optimization, then
sends:

```python
gs_queue.put({"type": "terminate"})
```

The worker then:

1. Syncs final poses from `SharedKeyframes`.
2. Rebuilds all sections from final keyframes if
   `vt_rebuild_from_final_keyframes: true`.
3. Runs `vt_n_iters_fine` refinement.
4. Saves native VT state:

   ```text
   <seq_name>_vtgs.pt
   ```

5. Materializes all sections and exports standard PLY:

   ```text
   <seq_name>_vtgs_online_gs.ply
   ```

The `.ply` is what SuperSplat and `eval_gs_psnr.py` consume.

---

## Internal data model

The mapper owns three levels of state:

```text
VTGaussianMapper
  |
  +-- kf_pose_data[kf_idx]
  |     latest Sim3 pose data for each source keyframe
  |
  +-- sections[]
        |
        +-- target camera data
        |     kf_indices, viewmats, gt_images, depth_maps, conf_maps, Ks_list
        |
        +-- view-tied Gaussian data
              anchors, source_kfs, sh_dc, log_radius, opacities
```

The key design split is:

- `target camera data` defines which images a section is trained against.
- `view-tied Gaussian data` defines the map primitives owned by that section.

The same section can contain Gaussians inserted from multiple keyframes. Each
Gaussian stores its own `source_kf`, so materialization always knows which pose
to use.

---

## Render/materialization path

Before calling the rasterizer, the mapper converts one or more sections into
standard 3DGS tensors:

```text
for each section:
  for each unique source_kf:
    means[source_kf] = Sim3(kf_pose_data[source_kf]).act(anchors[source_kf])

scales    = log_radius expanded to xyz
quats     = identity wxyz quaternion
opacities = opacity logits
colors    = sh_dc as SH degree 0
```

This happens every render step. That is the "view-tied" part: the persistent
state is tied to keyframe views, while world-space means are derived on demand.

For export, the same materialization is run once with detached tensors and then
passed into the existing `_export_ply(...)` helper.

---

## Representation

The standard online GS mapper learns free 3D Gaussians:

```text
mean, scale xyz, rotation, opacity, color
```

The view-tied mapper instead stores each Gaussian as a point tied to a source
keyframe pixel:

```text
anchor:      X_canon pixel point in the source camera frame
source_kf:   keyframe id that owns the anchor
sh_dc:       learnable RGB SH DC color
log_radius:  learnable isotropic radius
opacity:     learnable opacity logit
```

The mapper does not optimize:

- anchor location
- world-space mean directly
- quaternion rotation
- anisotropic xyz scale

At render/export time, the mapper materializes standard GS tensors:

```text
means  = T_WC(source_kf).act(anchor)
scales = exp(log_radius).repeat(3)
quats  = identity quaternion [1, 0, 0, 0]
color  = sh_dc
```

This is why final pose sync is especially important: the exported PLY reflects
the latest SLAM pose for every source keyframe.

---

## Keyframe insertion

Each keyframe snapshot provides:

- `uimg`: RGB image in `[0,1]`
- `X_canon`: MASt3R pointmap in camera coordinates
- `C` and `N`: confidence accumulation and update count
- `T_WC_data`: camera-to-world Sim3 pose
- `K`: calibrated intrinsics if available
- `img_shape`: render size

The mapper normalizes confidence:

```text
conf_norm = (C / N) / max(C / N)
```

and keeps pixels satisfying:

```text
conf_norm > c_conf_threshold
X_canon.z > vt_min_depth
```

When calibrated mode is enabled, `X_canon` is constrained to camera rays with
the same `constrain_points_to_ray(...)` helper used by the standard mapper.

---

## Sections

The map is divided into sections controlled by:

```yaml
vt_section_size: 8
```

The first keyframe in each section is the section head.

### Head frame

The head frame inserts all valid pixels, capped by:

```yaml
vt_head_insert_cap: 120000
```

If more valid pixels are available than the cap, the mapper keeps the highest
confidence pixels.

### Regular frame

A non-head frame first renders the current section into the new camera. It then
inserts only pixels that are not already covered well:

```text
alpha < vt_silhouette_thresh
or
relative_depth_error > vt_depth_cover_tol
```

The insertion cap is:

```yaml
vt_regular_insert_cap: 60000
```

This makes regular frames complement existing section geometry instead of
duplicating every visible surface.

---

## Radius and opacity initialization

Initial opacity is confidence-derived:

```text
opacity_logit = log(conf_norm / (1 - conf_norm))
```

Initial color is SH DC:

```text
sh_dc = (rgb - 0.5) / SH_C0
```

Initial radius uses intrinsics when available:

```text
radius = z / mean(fx, fy)
```

Without calibration, it falls back to local pointmap neighbor spacing.

Radius is clamped by:

```yaml
vt_min_radius: 1.0e-5
vt_max_radius: 0.20
```

---

## Training

Only the current section is optimized. Older sections are frozen and used as
rendering context.

Learnable tensors:

```text
sh_dc
log_radius
opacities
```

Optimizer:

```yaml
vt_lr_color: 2.5e-3
vt_lr_radius: 5.0e-3
vt_lr_opacity: 5.0e-2
```

Training schedule:

```yaml
vt_steps_head: 300
vt_steps_regular: 80
vt_steps_idle: 20
```

Loss:

```text
L = vt_lambda_rgb * L_rgb
  + vt_lambda_ssim * D-SSIM
  + vt_lambda_depth * L_depth
```

Where:

- `L_rgb`: L1 RGB loss on pixels with enough confidence/render alpha.
- `D-SSIM`: optional image-structure term.
- `L_depth`: confidence-masked log-depth L1.

Default weights:

```yaml
vt_lambda_rgb: 1.0
vt_lambda_depth: 0.1
vt_lambda_ssim: 0.2
```

---

## Render context

For online training, the mapper renders:

- current section
- previous section
- one older section with the best approximate overlap, when available

Overlap is estimated by projecting a sample of candidate section Gaussians into
the current target camera and checking relative depth agreement:

```yaml
vt_overlap_depth_tol: 0.05
vt_overlap_sample_cap: 5000
```

This keeps optimization local while still providing some continuity across
section boundaries.

---

## Pose sync and finalization

During idle time, the worker periodically calls:

```text
sync_poses_from_shared(keyframes)
```

controlled by:

```yaml
vt_sync_every_idle: 10
```

On terminate:

1. Sync final backend poses.
2. Optionally rebuild all sections from final `SharedKeyframes`:

   ```yaml
   vt_rebuild_from_final_keyframes: true
   ```

3. Run fine refinement:

   ```yaml
   vt_n_iters_fine: 10000   # in eval_vtgs.yaml
   ```

4. Save native `.pt`.
5. Export materialized PLY.

Rebuild is enabled by default because the online worker may have received early
keyframe snapshots before backend pose/pointmap updates fully settled.

---

## Important differences from the standard online mapper

| Area | Standard online GS | View-tied mapper |
|---|---|---|
| Gaussian mean | Learnable free world point | Materialized from source keyframe pose + anchor |
| Scale | Learnable anisotropic xyz | Learnable isotropic radius |
| Rotation | Learnable quaternion | Identity quaternion |
| Density growth | gsplat `DefaultStrategy` densify/prune | Section insertion from uncovered pixels |
| Optimization scope | Sliding camera window over full map params | Current section params only |
| Output | Standard PLY | Native `.pt` + materialized PLY |

The VT mapper is therefore more memory-stable and more tightly coupled to the
SLAM pointmap geometry, but less expressive than full free 3DGS v1 because it
does not currently learn anisotropy or SH view-dependent color.

---

## Debug checklist

If the PLY looks sparse:

- Lower `c_conf_threshold`.
- Increase `vt_head_insert_cap` and `vt_regular_insert_cap`.
- Check that `vt_rebuild_from_final_keyframes` is enabled.

If the PLY has duplicate/fuzzy surfaces:

- Raise `vt_silhouette_thresh`.
- Lower `vt_depth_cover_tol`.
- Reduce `vt_regular_insert_cap`.

If PSNR is low but geometry looks reasonable:

- Increase `vt_n_iters_fine`.
- Try lowering `vt_lambda_depth` if depth pulls color training too strongly.
- Evaluate with explicit `--ply <seq_name>_vtgs_online_gs.ply`.

If SuperSplat does not show the map:

- Make sure you opened the `.ply`, not the `.pt`.
- Confirm the run reached the terminate stage; VT outputs are written only when
  the worker receives `{"type": "terminate"}`.

---

## Current v1 limitations

- Mapping-only: no VTGaussian tracking replacement.
- Uses MASt3R pointmaps, not raw RGB-D depth.
- Exports DC color only; no SH degree 3 view-dependent color yet.
- Native `.pt` is for debug/checkpoint inspection only; there is no resume loader yet.
- The materialized PLY is a snapshot. If poses change later, regenerate/export again.
