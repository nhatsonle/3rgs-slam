# Gaussian Splatting Integration in MASt3R-SLAM

## Design choice: offline reconstruction

The integration runs **after** SLAM terminates, not during tracking. This avoids
any impact on real-time performance and lets the GS trainer use the final,
globally-optimised poses rather than noisy incremental estimates.

```
Video frames
    │
    ▼
┌─────────────────────────────┐
│  MASt3R-SLAM  (unchanged)   │
│  tracking → factor graph    │
│  → GN optimisation          │
└──────────────┬──────────────┘
               │  SharedKeyframes (final state)
               ▼
┌─────────────────────────────┐
│  extract_gaussians()        │  ← reads SharedKeyframes
│  train_gaussian_splat()     │  ← Adam + photometric loss
│  _save_splat()              │  ← writes .ply
└─────────────────────────────┘
```

---

## What `SharedKeyframes` contains (inputs to 3DGS)

Each keyframe slot in the 512-entry ring buffer exposes:

| Attribute | Shape | Description |
|---|---|---|
| `kf.X_canon` | `(H*W, 3)` | Dense 3D point map in **camera frame**, from MASt3R decoder |
| `kf.T_WC` | `lietorch.Sim3` | Camera-to-world pose (7-DOF: translation + quaternion + scale) |
| `kf.uimg` | `(H, W, 3)` | Unnormalised RGB image in `[0, 1]` |
| `kf.get_average_conf()` | `(H*W,)` | Per-pixel MASt3R confidence score |
| `kf.K` | `(3, 3)` | Camera intrinsics (only when `use_calib=True`) |
| `kf.img_shape` | tensor `[H, W]` | Image resolution |

---

## Step 1 — `extract_gaussians()`: SLAM outputs → Gaussian parameters

### 3D means (world-frame positions)

```python
X = kf.X_canon                          # (H*W, 3)  camera frame
if use_calib:
    X = constrain_points_to_ray(...)    # project onto calibrated ray
means_world = kf.T_WC.act(X)           # (H*W, 3)  world frame
```

`T_WC.act(X)` applies the full Sim3 transform (rotation, translation, scale) to
bring camera-frame points into the common world frame. All keyframes share this
frame, so concatenating their points gives a consistent 3D scene.

### Opacity (logit space)

```python
conf_norm = (conf / conf.max()).clamp(1e-4, 1 - 1e-4)   # normalise to (0,1)
opacity_logit = log(conf_norm / (1 - conf_norm))         # inverse sigmoid
```

MASt3R's per-pixel confidence directly seeds Gaussian opacity. High-confidence
pixels (sharp, well-matched) start with high opacity; low-confidence pixels
(edges, textureless regions) start transparent and can be pruned early.

### DC colour (SH degree 0)

```python
rgb = kf.uimg.reshape(-1, 3)           # (H*W, 3)  in [0, 1]
sh_dc = (rgb - 0.5) / SH_C0           # SH_C0 = 0.28209...
```

The standard 3DGS SH DC coefficient maps `[0,1]` RGB to the zero-order spherical
harmonic space. Only degree-0 SH is used — view-independent colour is sufficient
for a PoC and keeps parameter count low.

### Scales (log space)

Scale is derived from the spatial extent of adjacent 3D points in pixel-raster order:

```python
X_grid = X.reshape(H, W, 3)
dx = X_grid[:, 1:] - X_grid[:, :-1]   # right-neighbour offset
dy = X_grid[1:,  ] - X_grid[:-1,  ]   # down-neighbour offset
sx = ||dx||,  sy = ||dy||,  sz = (sx+sy)/2
log_scales = log([sx, sy, sz])         # (H*W, 3)
```

This gives each Gaussian an anisotropic disc whose size matches the local 3D
point density — dense regions get small Gaussians, distant/sparse regions get
large ones.

### Rotations (wxyz quaternion, aligned to surface)

```python
normal = normalise(cross(dx, dy))      # surface normal from pixel grid
quats  = _normal_to_quat_wxyz(normal)  # align Gaussian z-axis with normal
```

The Gaussian disc lies in the local surface tangent plane, so it naturally
represents the surface rather than a floating sphere.

### Camera matrices for the renderer

```python
T_se3   = as_SE3(kf.T_WC)             # strips Sim3 scale; SE3 is consistent
                                        # because world points already carry scale
w2c     = [R^T | -R^T t]              # 4×4 world→cam matrix
viewmat = w2c.to(device)              # gsplat requires CUDA tensor
```

`as_SE3()` internally calls `.cpu()` on the lietorch data — the resulting matrix
must be moved back to GPU before passing to gsplat.

### GT depth and confidence maps (for depth loss)

```python
depth_gt = X[:, 2].reshape(H, W).clamp(min=1e-3)  # camera z in metres
conf_map = conf_norm.reshape(H, W)                  # normalised confidence (0–1)
```

These are stored per-keyframe alongside the RGB images and consumed by the depth
loss during training. Using the camera-frame z of `X_canon` (before `T_WC.act`)
gives the raw MASt3R depth without any Sim3 scale contamination.

### Threshold / hard cap

After extraction, all Gaussians with `conf_norm ≤ threshold` are discarded.
If the remaining count still exceeds `max_gaussians`, the lowest-opacity
Gaussians are further dropped:

```python
keep_idx = topk(sigmoid(opacity_logit), max_gaussians).indices
```

This is the primary guard against memory blowup on long sequences.

---

## Step 2 — `train_gaussian_splat()`: photometric optimisation

### Optimiser

Adam with per-parameter learning rates (matching the original 3DGS paper):

| Parameter | LR (demo) |
|---|---|
| means | 1.6 × 10⁻⁴ |
| sh_dc (colour) | 2.5 × 10⁻³ |
| log_scales | 5 × 10⁻³ |
| quats | 1 × 10⁻³ |
| opacity logit | 5 × 10⁻² |

### Training loop (per iteration)

```
sample random keyframe kf_idx
    │
    ▼
gsplat.rasterization(
    means, quats, exp(log_scales), sigmoid(opacity),
    sh_dc.unsqueeze(1),               # (N,1,3) DC SH
    viewmats=viewmat[kf_idx],
    Ks=K[kf_idx],
    render_mode="RGB+D",              # returns (1,H,W,4): RGB + camera-z depth
    sh_degree=0, packed=False, absgrad=True
)
    │
    ├─ render     (H,W,3)  ─── photometric loss ──►┐
    └─ render_dep (H,W)    ─── depth loss ──────────┤
                                                     ▼
                                              combined loss
                                              loss.backward()
                                              optimizer.step()
                                              normalise quats
```

`render_mode="RGB+D"` gives both the alpha-composited colour image and the
accumulated camera-z depth in one rasterization pass — no extra overhead vs
RGB-only rendering. `packed=False` ensures a differentiable output even when
few Gaussians project onto the image. `absgrad=True` populates
`meta["means2d"].absgrad` for density control.

**Note:** `main.py` calls `torch.set_grad_enabled(False)` globally before the
SLAM loop. The GS module re-enables gradients at the start of training:
```python
torch.set_grad_enabled(True)
```

### Loss

Combined loss with three terms:

```
L = (1 − λ_ssim) · L1  +  λ_ssim · L_D-SSIM  +  λ_depth · L_depth
```

| Term | Formula | Role |
|---|---|---|
| `L1` | `\|render − gt\|.mean()` | Per-pixel colour fidelity |
| `L_D-SSIM` | `(1 − SSIM(render, gt)) / 2` | Structural edges and local contrast |
| `L_depth` | `mean(conf · \|log z_render − log z_gt\|)` where `conf > depth_min_conf` | Geometry regularisation; prevents floaters |

Default weights: `λ_ssim = 0.2`, `λ_depth = 0.1`.

**L_D-SSIM** uses the same windowed SSIM kernel as the metric logger (11×11
Gaussian, σ = 1.5), but evaluated inside the autograd graph so gradients
flow back through the convolutions.

**L_depth** compares the rendered camera-z depth against `depth_maps[kf_idx]`
(the MASt3R `X_canon[:, 2]` stored during extraction). Log-depth L1 makes the
loss scale-invariant across near/far regions. Pixels with confidence below
`depth_min_conf` are masked out to avoid penalising noisy depth estimates in
textureless or motion-blurred areas. Set `lambda_depth: 0` to disable entirely.

### Adaptive density control

Every `densify_interval` iterations (between `densify_from_iter` and
`densify_until_iter`):

1. **Prune** Gaussians with `sigmoid(opacity) < prune_opacity_thresh` — dead /
   transparent Gaussians are removed.
2. **Clone** Gaussians with accumulated abs-2D-gradient above `grad_thresh` —
   these correspond to under-reconstructed regions that need more coverage.

Every `opacity_reset_interval` iterations, all opacities are reset to −2.0
(sigmoid ≈ 0.12) so that dead Gaussians can surface and be pruned cleanly.

### Metrics logged

Every 500 iterations:

```
[GS] iter  1500/3000 | loss 0.0376 | PSNR 23.01 dB | SSIM 0.7902 | N=491,700
```

Both PSNR and SSIM are computed against the randomly-sampled keyframe used in
that iteration (not a held-out set). Typical numbers on the demo sequence
(IMG_2520.mp4, 15 keyframes, 3000 iters, `lambda_ssim=0.2`, `lambda_depth=0.1`):

| Stage | PSNR | SSIM |
|---|---|---|
| Initialisation (iter 0) | ~10 dB | ~0.24 |
| iter 1000 | ~27 dB | ~0.96 |
| iter 2000 | ~26 dB | ~0.91 |

The D-SSIM term noticeably accelerates structural convergence (SSIM reaches
0.96 at iter 1000 vs 0.72 with L1 alone). The loss oscillates more between
opacity resets — this is expected since D-SSIM is more sensitive to phase
shifts than L1.

---

## Step 3 — output: `.ply` file

Saved to `<save_dir>/<seq_name>_gs.ply` in the standard 3DGS PLY format
(compatible with SuperSplat, gsplat viewers, and most 3DGS tools):

| PLY field | Source |
|---|---|
| `x, y, z` | `means` (world frame) |
| `f_dc_0/1/2` | `sh_dc` (DC colour) |
| `opacity` | `opacity` logit |
| `scale_0/1/2` | `log_scales` |
| `rot_0/1/2/3` | `quats` (wxyz) |

---

## Data flow diagram (detailed)

```
MASt3R decoder output (per keyframe)
────────────────────────────────────
X_canon  (H*W, 3)  ──[T_WC.act]──► means_world  (H*W, 3)
conf     (H*W,)    ──[logit]──────► opacity      (H*W,)
uimg     (H, W, 3) ──[/SH_C0]────► sh_dc        (H*W, 3)
X_canon  (H*W, 3)  ──[grid diff]──► log_scales   (H*W, 3)
X_canon  (H*W, 3)  ──[normal]─────► quats        (H*W, 4) wxyz
T_WC     Sim3      ──[as_SE3,w2c]─► viewmats     (N_kf, 4, 4)
uimg     (H, W, 3) ──────────────► gt_images     list[(H,W,3)]
X_canon  (H*W, 3)  ──[z-channel]──► depth_maps   list[(H,W)]
conf     (H*W,)    ──[reshape]────► conf_maps     list[(H,W)]
K        (3, 3)    ──────────────► Ks_render     (N_kf, 3, 3)

[threshold filter + hard cap] → N ≤ max_gaussians

Adam optimisation (n_iters), render_mode="RGB+D"
    random keyframe → rasterize (RGB+depth) →
        (1-λ_ssim)·L1 + λ_ssim·D-SSIM + λ_depth·log-depth → backward
    + adaptive density control (prune + clone)

Output: <save_dir>/<seq_name>_gs.ply
```

---

## Key design constraints

- **`threshold` parameter required on all GS functions** — prevents Gaussians
  boom on long sequences; both a confidence gate and a hard-cap path.
- **No changes to SLAM backbone** — `main.py`, `tracker.py`, `global_opt.py`,
  `frame.py`, and all MASt3R code are untouched except the 5-line trigger in
  `main.py`.
- **Offline only** — GS training starts after `backend.join()` resolves, so
  tracking and bundle adjustment are fully complete before any Gaussian
  is created.
