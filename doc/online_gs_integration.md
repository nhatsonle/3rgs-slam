# Online Gaussian Splatting Integration

## Goal

Deeply integrate 3DGS into the MASt3R-SLAM pipeline so Gaussians are built and
trained **incrementally during SLAM**, not just as a post-processing step.
The scene representation grows keyframe-by-keyframe alongside the factor graph,
and optionally refines camera poses photometrically after each global solve.

---

## Where Online GS Runs

Online GS lives entirely inside the **backend process** (`run_backend` in
`main.py`). This is the right place because:

- The backend already has access to `SharedKeyframes` (poses + point maps).
- It runs after each `solve_GN_rays/calib()`, when all visible poses are
  globally optimized — the best moment to extract and train.
- The backend is a `mp.Process(spawn)` child: it starts with a fresh Python
  interpreter where `torch.set_grad_enabled` defaults to `True`, so autograd
  works without any extra setup.

The main thread and visualization process are completely unaffected.

---

## Data Flow (Per Keyframe)

```
main thread                       backend process
──────────────────────────────    ──────────────────────────────────────────
frame tracked → new KF?           solve_GN_rays/calib()
  keyframes.append(frame)             ↓ poses globally optimized
  states.queue_global_opt(idx)    gs_module.add_keyframe(keyframes[idx])
                                      ↓ extract Gaussians from KF with GN pose
                                  gs_module.sync_poses(keyframes)
                                      ↓ update ALL viewmats from latest GN
                                  gs_module.train_gaussians(N)
                                      ↓ N Adam steps (RGB+D loss)
                                  [gs_module.refine_poses(M)]  # optional
                                      ↓ M photometric viewmat steps

                                  … repeat per KF …

TERMINATED                        gs_module.save(save_dir, seq_name)
backend.join()                        → writes <seq>_gs.ply
```

---

## New Code

### `mast3r_slam/gaussian_splat.py`

Three additions (existing functions unchanged):

#### `_extract_single_keyframe(kf, use_calib, threshold)` (line 526)

Extracts all Gaussian initialization data from one keyframe — means, SH-DC
colour, log-scales, wxyz quaternions, opacity logits, viewmat, GT depth/conf
maps — applying the same logic as `extract_gaussians` but for a single KF.
Returns a plain dict (no `requires_grad`), or `None` if no pixels exceed
`threshold`.

#### `_append_keyframe_data(data, optimizer, kf_data)` (line 584)

Appends one keyframe's Gaussians into the live `data` dict and Adam state.
Concatenates each parameter tensor and zero-pads the Adam first/second moment
accumulators for the new entries so training continues seamlessly.

#### `class OnlineGaussianSplat` (line 625)

| Method | Purpose |
|---|---|
| `add_keyframe(kf, threshold)` | Extract Gaussians from `kf`, merge into scene (no training) |
| `sync_poses(keyframes)` | Overwrite viewmats from latest GN-optimized `T_WC` (non-differentiable) |
| `train_gaussians(n, threshold)` | N Adam steps: combined L1 + D-SSIM + log-depth loss; density control every 100 steps |
| `refine_poses(n, threshold)` | N photometric gradient steps on per-KF viewmats (Gaussians frozen, KF-0 pinned) |
| `save(save_dir, seq_name)` | Write trained Gaussians as `<seq>_gs.ply` |

Internal helpers: `_init_optimizer`, `_append_k_render`, `_density_control`.

### `main.py`

| Change | Location |
|---|---|
| `run_backend` signature adds `gs_save_dir=None, gs_seq_name=None` | line 74 |
| Create `OnlineGaussianSplat` in backend if `online: true` | line 82–89 |
| After each `solve_GN`: `add_keyframe → sync_poses → train_gaussians → refine_poses` | line 150–158 |
| After while-loop exits: finalize + `gs_module.save()` | line 164–171 |
| Pass save paths to backend `mp.Process` | line 254–265 |
| Skip offline `train_gaussian_splat` when `online: true` | line 371–376 |

### `config/base.yaml`

New keys added to the `gaussian_splat` block:

```yaml
online: false                    # enable incremental mode
online_iters_per_kf: 50         # Adam steps after each GN solve
densify_online: true             # prune+clone every 100 training steps
pose_refine: false               # photometric viewmat refinement (experimental)
pose_refine_iters: 10            # steps of pose-only opt per KF
lr_pose: 1.0e-4                  # Adam LR for 4×4 viewmat refinement
online_finalize_iters: 0         # extra steps after SLAM ends; 0 = save immediately
```

### `config/demo_gs.yaml`

Added convenience defaults for online mode (disabled by default; flip `online:
true` to activate):

```yaml
online: false
online_iters_per_kf: 100
online_finalize_iters: 1000
```

---

## Training Details

The loss is identical to the offline mode:

```
L = (1 − λ_ssim) · L1  +  λ_ssim · D-SSIM  +  λ_depth · log-depth
```

Density control runs every 100 training steps during online training (prune
low-opacity Gaussians, clone high-gradient ones). The hard cap `max_gaussians`
is respected.

---

## Pose Refinement (`refine_poses`)

When `pose_refine: true`, after each `train_gaussians` call the module runs M
Adam steps on the per-KF 4×4 viewmats using photometric L1 loss:

- Gaussian parameters are **frozen** (detached).
- **KF-0 is pinned** as the global reference frame (never optimized).
- KF-1…K are optimized as free 4×4 matrices with a small LR (`lr_pose`).
- Refined viewmats are written back to `data["viewmats"]` after each call.
- The GN poses in `SharedKeyframes` are **not** modified — refinement is
  local to the GS module.

Because GN already provides good poses and the LR is small, the matrices stay
approximately valid SE3 without explicit orthogonality constraints.

---

## Running

**Offline mode (unchanged):**
```bash
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml \
  --no-viz --save-as gs_poc
# demo_gs.yaml: online: false  (default)
```

**Online mode:**
```bash
# In config/demo_gs.yaml, set:  online: true
python main.py --dataset IMG_2520.mp4 --config config/demo_gs.yaml \
  --no-viz --save-as gs_online
```

With `online_iters_per_kf: 100` and ~15 keyframes, the backend does ~1,500
training steps during SLAM. Add `online_finalize_iters: 1000` for a brief
final polish pass before saving.

---

## Architecture Invariants Preserved

- No changes to the SLAM backbone (tracker, factor graph, GN solver).
- No new shared-memory tensors — GS tensors live entirely in the backend
  process and are not visible to main or visualization.
- Offline mode (`online: false`) is completely unaffected.
- The `threshold` parameter is propagated through all GS functions to prevent
  Gaussians boom on long sequences.
