# Gaussian Splatting Integration in MASt3R-SLAM

This project now uses **online 3DGS mapping as the main path**.
Offline batch GS is still available for fallback/ablation.

---

## Architecture (current)

```
Frames -> MASt3R-SLAM (tracking + global optimization)
          |                               |
          | new keyframe snapshots        | optimized poses in shared memory
          v                               v
      run_online_gs worker  <---- periodic pose sync ----
          |
          +-- Stage 1: online coarse mapping (incremental)
          +-- Stage 2: post-SLAM fine refinement
          +-- Save: <seq_name>_online_gs.ply
```

Online worker is decoupled from SLAM backbone and runs in a separate process.

---

## Main integration points

### Start worker

When both conditions are met:
- `--save-as` is provided (`dataset.save_results == true`)
- `gaussian_splat.online_enabled == true`

`main.py` spawns:
- `run_online_gs(config, states, keyframes, save_dir, seq_name, gs_queue)`

### Publish keyframes

Each time a keyframe is added, main sends:
- `make_keyframe_snapshot(...)` -> `gs_queue.put({"type": "new_kf", ...})`

### Finalize

After `backend.join()`:
- `gs_queue.put({"type": "terminate"})`
- worker syncs final global-optimized poses
- worker runs fine refinement and saves final PLY

---

## Stage 1: online coarse mapping

For each `new_kf` event:
1. Decode snapshot: `X_canon`, confidence, RGB, Sim3 pose, intrinsics.
2. Build new Gaussians (disc-oriented initialization).
3. Apply insertion constraints:
   - confidence gate (`c_conf_threshold`)
   - per-kf cap (`max_gaussians_per_kf`)
   - global cap (`max_gaussians`)
4. Run `steps_per_keyframe` train steps.

During idle cycles (no new event):
- run `idle_train_steps`
- periodically call `sync_poses_from_shared(...)`

Training camera sampling:
- recent window size: `window_size`
- random history cameras: `random_history`

---

## Stage 2: fine refinement

Triggered only on terminate event, after SLAM backend has finished.

Flow:
1. Sync all camera poses from final shared keyframe poses.
2. Optional floater prune (`fine_prune_thresh`).
3. Run `n_iters_fine` full-map refinement.
4. Save final output map.

By default in demo/eval profiles, fine stage often uses `fine_lambda_depth: 0.0`
to avoid cross-camera depth inconsistency from Sim3 scale mismatch.

---

## Loss and regularization

Online objective follows MASt3R-GS style:

`L = alpha_rgb * L_rgb + (1 - alpha_rgb) * L_depth + lambda_iso * L_iso`

Where:
- `L_rgb`: RGB reconstruction (L1, optionally blended with D-SSIM via `lambda_ssim`)
- `L_depth`: confidence-masked log-depth residual
- `L_iso`: isotropy regularization for Gaussian scales

---

## Gaussian initialization and controls

Initialization sources per keyframe:
- mean: `T_WC.act(X_canon)`
- color: SH DC from RGB
- opacity: confidence -> logit
- scale: neighbor geometry in image grid
- rotation: normal-aligned quaternion

Explosion-prevention controls:
- `c_conf_threshold`
- `max_gaussians_per_kf`
- `max_gaussians`
- `max_log_scale`, `min_log_scale`
- `max_scale_ratio` (needle pruning)

These are the key controls preventing Gaussian boom.

---

## Output files

- Online output: `<save_dir>/<seq_name>_online_gs.ply` (primary)
- Offline output: `<save_dir>/<seq_name>_gs.ply` (legacy path if enabled)

Both use standard 3DGS-compatible PLY fields (`x y z`, SH DC, opacity, scales, rotation).

---

## Offline path status

Offline mapper in `mast3r_slam/gaussian_splat.py` is still supported.
It is triggered only when:
- `gaussian_splat.enabled == true`

It runs after SLAM completion and does not participate in incremental mapping.
