# Online Gaussian Splatting Integration

## Two modes

| Mode | Config | Class used | When Gaussians are created |
|---|---|---|---|
| **Offline** | `online: false` | `train_gaussian_splat()` | After SLAM finishes, from all keyframes |
| **Online / OnlineGaussianSplat** | `online: true`, `use_global_map: false` | `OnlineGaussianSplat` | Per keyframe, during SLAM |
| **Online / GlobalGaussianMap** | `online: true`, `use_global_map: true` | `GlobalGaussianMap` | From selected non-KF frames, multi-view fused |

`GlobalGaussianMap` is the recommended mode. It solves four problems with the
simpler per-keyframe approach: map blurriness, object duplication, insufficient
viewpoint coverage, and map inconsistency after loop closure.

---

## Where Online GS runs

Both online modules live entirely inside the **backend process** (`run_backend`
in `main.py`):

- The backend already has access to `SharedKeyframes` (poses + point maps).
- It runs after each `solve_GN_rays/calib()` — globally optimised poses are
  available before any Gaussian is touched.
- The backend is a `mp.Process(spawn)` child with `grad_enabled=True` by default.
- The main thread and visualisation process are completely unaffected.

---

## GlobalGaussianMap — 12-step design

### Step 1 — SLAM backbone (unchanged)

MASt3R-SLAM tracking and factor-graph optimisation run as normal. No
modifications to `tracker.py`, `global_opt.py`, or the CUDA Gauss-Newton kernel.

### Step 2 — Mapping Frame Selector (`MappingFrameSelector`)

Runs in the **main process** alongside tracking. After each non-KF frame is
tracked, three gates are checked:

| Gate | Config key | Default |
|---|---|---|
| Cooldown: min frames since last mapping frame | `map_cooldown_frames` | 3 |
| Confidence: average MASt3R confidence > threshold | `map_conf_thresh` | 0.20 |
| Baseline: translation from last KF > threshold | `map_baseline_thresh` | 0.02 m |

Frames that pass all three are serialised (CPU tensors: `uimg`, `sim3_data`,
`X_canon`, `C`, `K`) and pushed to `mapping_queue` (`mp.Queue(maxsize=300)`)
with `put_nowait` — the main thread is never blocked.

```python
# main.py — after each non-KF frame during TRACKING mode
if mapping_frame_selector.should_map(frame, last_kf_t):
    mapping_queue.put_nowait({
        "uimg": frame.uimg.cpu(),
        "sim3_data": frame.T_WC.data.detach().cpu(),
        "X_canon": frame.X_canon.detach().cpu(),
        "C": frame.C.detach().cpu(),
        "H_img": H_img, "W_img": W_img, "K": ...
    })
```

The selector resets its cooldown counter whenever a new keyframe is added.

### Step 3 — Local Fusion Buffer (`LocalFusionBuffer`)

A `deque(maxlen=window_size)` in the backend. Each incoming mapping frame is
unpacked: world-frame 3D points `T_WC.act(X_canon)`, normalised confidence,
camera-frame depth, RGB, and viewmat are stored. Once the buffer holds
`≥ map_min_frames` frames, `fuse()` concatenates all valid points across the
window into a single point cloud for candidate generation.

The sliding window automatically ages out old frames (oldest are discarded when
the deque is full), providing temporal locality without explicit bookkeeping.

### Step 4 — Gaussian Candidate Generation

From the fused point cloud:

- **Opacity** (logit) = `log(conf / (1 − conf))` where conf is normalised in `(0,1)`
- **SH DC colour** = `(rgb − 0.5) / SH_C0`
- **Scale** (isotropic) = `log(0.005 × median_scene_depth)` — prevents over-large init
- **Quaternion** = identity (`[1, 0, 0, 0]` wxyz) — relaxed during optimisation

Candidates are capped at `assoc_max_check` (default 3000 for demo, 5000 in
base) by taking the top-confidence subset, keeping association cost bounded.

### Step 5 — GlobalGaussianMap state

The live map state is a plain dict of `requires_grad=True` tensors managed by
a single Adam optimizer:

```
data["means"]       (N, 3)   world-frame centres
data["sh_dc"]       (N, 3)   DC SH colour
data["log_scales"]  (N, 3)   log-space anisotropic scales
data["quats"]       (N, 4)   unit quaternions wxyz
data["opacity"]     (N,)     logit-space opacity
data["viewmats"]    (K, 4, 4) world→cam per registered KF (not optimised)
data["gt_images"]   list[K]  reference RGB (H,W,3) per KF
data["depth_maps"]  list[K]  GT camera-z depth per KF
data["conf_maps"]   list[K]  normalised confidence per KF
```

Two auxiliary tensors are kept in sync but are not Adam parameters:
- `obs_counts (N,)` — how many times each Gaussian has been confirmed
- `absgrad_accum (N,)` — accumulated abs-2D-gradient for densification

### Step 6 — Association Engine (`VoxelAssociation`)

Before spawning any new Gaussian, every candidate is checked against the
existing map using a spatial hash:

1. `rebuild(means)` — discretise existing means by `assoc_voxel` (5 cm), build
   a Python `dict: voxel_key → [gaussian_indices]`
2. `query_batch(new_means)` — for each candidate, check the 27 surrounding voxel
   cells; return `(nearest_idx, dist)` for the closest existing Gaussian

Cost: O(M) rebuild + O(27) per candidate query ≈ O(1) per candidate.

The grid is rebuilt lazily (`_assoc_dirty` flag) only when new Gaussians are
spawned or positions change significantly (e.g. after re-anchor).

### Step 7 — Merge or Spawn

| Condition | Action |
|---|---|
| `dist < assoc_thresh` (default 5 cm) | **Merge**: confidence-weighted mean update; EMA SH colour update (`merge_ema` α) |
| No close neighbour | **Spawn**: append new Gaussian; zero-pad Adam state for new entries |

```python
# Merge: confidence-weighted position
new_mean = (old_count * old_mean + new_conf * new_mean) / (old_count + new_conf)
# EMA colour
new_sh = (1 − ema_alpha) * old_sh + ema_alpha * new_sh
obs_counts[merged] += 1
```

New Gaussians are appended via `_append_candidates` which correctly extends
both the parameter tensors and Adam's `exp_avg / exp_avg_sq` state.

### Step 8 — Density Control

Every 100 training steps during online training:

1. **Prune**: remove Gaussians with `sigmoid(opacity) < prune_opacity_thresh`
2. **Clone**: Gaussians with accumulated abs-2D-gradient / train_step > `grad_thresh`
   are cloned and jittered; count capped at `max_gaussians`

`absgrad_accum` is zeroed after each density control pass and rebuilt as
training continues.

### Step 9 — Hybrid Loss

```
L = (1 − λ_ssim)·L1 + λ_ssim·L_D-SSIM + λ_depth·L_depth + λ_reg·L_reg
```

| Term | Formula | Default weight |
|---|---|---|
| L1 | `|render − gt|.mean()` | `1 − λ_ssim = 0.8` |
| L_D-SSIM | `(1 − SSIM(render, gt)) / 2` | `λ_ssim = 0.2` |
| L_depth | `mean(conf_w · |log z_render − log z_gt|)` masked by `conf > depth_min_conf` | `λ_depth = 0.1` |
| L_reg | `0.5·mean(exp(scales)) + 0.5·mean(1 − sigmoid(opacity))` | `λ_reg = 0.01` |

`L_reg` penalises over-large Gaussians and near-zero opacity, preventing bloated
geometry and ghost Gaussians that survive pruning.

For aux frames (non-KF photometric supervision), only L1 weighted by `aux_decay^age`
is used — no depth or SSIM, keeping the computation cheap.

### Step 10 — Async Optimizer

Training runs only in the backend process, interleaved with GN solves:

```
backend per-KF cycle:
  add_keyframe_view(kf)        ← register render target (no Gaussian init)
  _drain_queue(mapping_queue)  ← fuse geometry, associate, merge/spawn
  sync_poses(keyframes)        ← copy GN-optimised viewmats (non-differentiable)
  train_gaussians(N)           ← N Adam steps
```

The main thread never touches GS tensors. The only shared state is the
`mapping_queue` (`mp.Queue`) and the `SharedKeyframes` ring buffer.

### Step 11 — Loop-Closure Re-Anchor

When the backend's RELOC solve succeeds, poses of all keyframes may shift.
Without re-anchoring, world-frame Gaussian positions would diverge from the
corrected poses.

```python
# run_backend — RELOC path
old_poses = gs_module.capture_poses(keyframes)   # snapshot before GN
success = relocalization(...)
if success:
    gs_module.reanchor(keyframes, old_poses)     # propagate corrections
```

Re-anchor math for each Gaussian point `p`, assigned to nearest old camera centre:

```
p' = R_new^T @ R_old @ p + R_new^T @ (t_old − t_new)
```

where `R, t` come from the world-to-cam matrix of the nearest keyframe.
Processing is chunked (50 k points at a time) to avoid OOM on large maps.
After re-anchor, `sync_poses` updates all viewmats and `_assoc_dirty` is set.

### Step 12 — Periodic Cleanup

Every `cleanup_interval_kf` keyframes (default 10):

1. Prune Gaussians with `sigmoid(opacity) < prune_opacity_thresh`
2. Additionally prune Gaussians with `obs_count ≤ 1` AND opacity < 0.01 —
   these were spawned but never confirmed by a second observation

After pruning, the association grid is marked dirty so it is rebuilt on the
next mapping frame.

---

## Data Flow Diagram

```
main process                          backend process
─────────────────────────────         ────────────────────────────────────────
frame i → tracker.track()
  if new KF:
    keyframes.append(frame)           solve_GN_rays/calib()
    states.queue_global_opt(idx)  →       ↓ globally optimised poses
    mapping_frame_selector.reset()    add_keyframe_view(kf)   ← render target only
  else if should_map(frame):              ↓
    mapping_queue.put_nowait(data) →  _drain_queue(mapping_queue)
                                          for each mapping frame:
                                            _unpack_mapping_frame()
                                            fusion_buffer.add()
                                            if buffer.ready():
                                              fuse() → candidates
                                              associate() → merge/spawn
                                          ↓
                                      sync_poses(keyframes)
                                          ↓
                                      train_gaussians(online_iters_per_kf)
                                          ↓ repeat per KF …

TERMINATED                            ── finalize ──
backend.join()                        drain remaining KF tasks → add_keyframe_view
                                      drain mapping_queue → associate/spawn
                                      sync_poses()
                                      train_gaussians(online_finalize_iters)
                                      gs_module.save() → <seq>_gs.ply
```

---

## Finalize behaviour

When the main process sets `Mode.TERMINATED`, the backend may have unprocessed
entries in both `global_optimizer_tasks` and `mapping_queue`. The finalize
section handles both:

1. **Drain missed KF tasks** — registers all KF views skipped when the backend
   was busy. Without this, a fast video sequence produces only 2–3 registered
   render views and training is heavily under-constrained.
2. **Drain mapping queue** — flushes remaining mapping frames so geometry from
   the tail of the sequence is fused before training.
3. **Final GN solve** — re-runs optimisation once with all factors so poses
   used for `sync_poses` are globally consistent.
4. **`train_gaussians(online_finalize_iters)`** — bulk training pass with all
   views registered (`online_finalize_iters: 2000` in `demo_gs.yaml`).

This is why the bulk of `[GMap] Mapping frame` log lines appear after "done —
waiting for backend GS finalize..." — they are flushed in the finalize drain,
not during SLAM.

---

## Running

```bash
# GlobalGaussianMap (recommended)
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml \
  --no-viz --save-as results
# Output: logs/results/IMG_2520_gs.ply  ← GS map  (separate from SLAM PLY)
#         logs/results/IMG_2520.ply     ← SLAM point-cloud reconstruction
#         logs/results/IMG_2520.txt     ← camera trajectory

# OnlineGaussianSplat (simple per-KF, no fusion)
# In demo_gs.yaml set:  use_global_map: false
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml \
  --no-viz --save-as results
```

Expected log output with GlobalGaussianMap:

```
[GMap] GlobalGaussianMap online mode enabled.
[GMap] MappingFrameSelector enabled.
[GMap] GlobalGaussianMap initialized.
[GMap] Bootstrapped 3,000 Gaussians from first mapping frame.
[GMap] Mapping frame: merged=3000 spawned=0  | total=3,000
[GMap] Mapping frame: merged=2797 spawned=203 | total=3,203
...
done — waiting for backend GS finalize...
[GMap] Finalize: registered 12 missed KF views (total 15 views).
[GMap] Finalize: flushed 22 remaining mapping frames.
[GS/GMap] Final polish: 2000 additional steps (15 views, 12,786 Gaussians)...
[GMap] Saved → logs/results/IMG_2520_gs.ply  (13,055 Gaussians, 2300 train steps, 15 KF views)
[GS] Backend finished. GS output → logs/results/IMG_2520_gs.ply  (mode: use_global_map)
```

---

## Config reference — GlobalGaussianMap keys

Added to the `gaussian_splat` block in `base.yaml`:

| Key | Default | Description |
|---|---|---|
| `use_global_map` | `false` | Set `true` to activate `GlobalGaussianMap` |
| **Mapping Frame Selector** | | |
| `map_baseline_thresh` | `0.02` | Min translation from last KF (metres) |
| `map_conf_thresh` | `0.20` | Min average MASt3R confidence |
| `map_cooldown_frames` | `3` | Min non-KF frames between mapping frames |
| **Local Fusion Buffer** | | |
| `map_window_size` | `7` | Sliding window depth (number of mapping frames) |
| `map_min_frames` | `2` | Frames required before fusion triggers |
| **Association Engine** | | |
| `assoc_thresh` | `0.05` | 3D distance threshold: merge if closer, spawn otherwise |
| `assoc_voxel` | `0.05` | Voxel cell size for spatial hash (same units as scene) |
| `assoc_max_check` | `5000` | Max candidate Gaussians per association pass |
| **Merge** | | |
| `merge_ema` | `0.30` | EMA alpha for SH colour update during merge |
| **Loss** | | |
| `lambda_reg` | `0.01` | Scale + opacity regulariser weight (0 to disable) |
| **Loop closure** | | |
| `reanchor_on_loop_closure` | `true` | Propagate pose corrections to Gaussians on RELOC |
| **Cleanup** | | |
| `cleanup_interval_kf` | `10` | Run cleanup every N keyframes |

---

## Architecture invariants preserved

- No changes to tracker, factor graph, or GN solver.
- No new shared-memory tensors — GS tensors live only in the backend process.
- `use_global_map: false` (default) falls back to `OnlineGaussianSplat` exactly
  as before.
- `online: false` runs the original offline batch training; the GlobalGaussianMap
  code is not imported.
- The `mapping_queue` uses `put_nowait` everywhere — main thread is never blocked.
