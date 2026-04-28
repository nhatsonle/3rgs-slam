"""
Offline Gaussian Splatting reconstruction from MASt3R-SLAM keyframes.

Called after SLAM terminates. Takes final SharedKeyframes (optimized poses +
dense point maps) and trains a 3DGS scene representation.

All public functions accept a `threshold` parameter that gates the maximum
number of Gaussians created/kept, preventing memory blowups on long sequences.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.lietorch_utils import as_SE3


# ── Constants ──────────────────────────────────────────────────────────────────
SH_C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))


# ── 1. Data extraction ─────────────────────────────────────────────────────────

def extract_gaussians(
    keyframes: SharedKeyframes,
    use_calib: bool,
    threshold: float = 0.5,
) -> dict:
    """
    Build initial Gaussian parameters from all keyframes in SharedKeyframes.

    Mirrors evaluate.save_reconstruction for data extraction. One Gaussian
    per valid pixel; validity is gated by `threshold` (normalised confidence).

    Args:
        keyframes:  Populated SharedKeyframes buffer (post-SLAM).
        use_calib:  Whether calibrated mode was used (applies constrain_points_to_ray).
        threshold:  Normalised confidence cutoff in (0, 1).  Raise to reduce
                    Gaussian count; lower to keep more but risk noisy geometry.
                    Prevents Gaussians boom on large / long sequences.

    Returns:
        Dict with keys: means, sh_dc, log_scales, quats (wxyz), opacity_logit,
        viewmats (world→cam, (N_kf,4,4)), gt_images (list of (H,W,3) tensors),
        Ks (optional (N_kf,3,3) tensor).
    """
    all_means, all_sh_dc, all_log_scales = [], [], []
    all_quats, all_opacity = [], []
    all_viewmats, all_gt_images, all_Ks = [], [], []
    all_depth_maps, all_conf_maps = [], []

    n_kf = len(keyframes)
    print(f"[GS] Extracting from {n_kf} keyframes (confidence threshold={threshold})...")

    for i in range(n_kf):
        kf = keyframes[i]
        device = kf.X_canon.device

        # ── 3D points in world frame (same logic as evaluate.save_reconstruction)
        X = kf.X_canon                                          # (H*W, 3) camera frame
        if use_calib:
            X = constrain_points_to_ray(
                kf.img_shape.flatten()[:2], X[None], kf.K
            ).squeeze(0)
        means_world = kf.T_WC.act(X)                           # (H*W, 3)

        # ── Validity mask from confidence
        conf = kf.get_average_conf().reshape(-1)                # (H*W,)
        conf_max = conf.max().clamp(min=1e-6)
        conf_norm = (conf / conf_max).clamp(1e-4, 1 - 1e-4)
        valid = conf_norm > threshold                           # (H*W,) bool

        # ── Opacity (logit space for Adam)
        opacity_logit = torch.log(conf_norm / (1.0 - conf_norm))  # (H*W,)

        # ── DC colour (SH degree 0)
        rgb = kf.uimg.reshape(-1, 3).to(device)                 # (H*W, 3) in [0,1]
        sh_dc = (rgb - 0.5) / SH_C0                             # (H*W, 3)

        # ── Scale from pixel-grid neighbour distances
        #    X_canon is in pixel-raster order so spatial neighbours are at ±1 / ±W
        H_img, W_img = kf.img_shape.flatten()[:2].tolist()
        H_img, W_img = int(H_img), int(W_img)
        X_grid = X.reshape(H_img, W_img, 3)

        dx = F.pad(X_grid[:, 1:] - X_grid[:, :-1], (0, 0, 0, 1))    # (H,W,3)
        dy = F.pad(X_grid[1:] - X_grid[:-1],       (0, 0, 0, 0, 0, 1))  # (H,W,3)

        sx = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp(min=1e-6)   # (H,W,1)
        sy = torch.linalg.norm(dy, dim=-1, keepdim=True).clamp(min=1e-6)
        sz = (sx + sy) / 2.0
        log_scales = torch.log(
            torch.cat([sx, sy, sz], dim=-1)
        ).reshape(-1, 3)                                         # (H*W, 3)

        # ── Rotation: align Gaussian disc to surface normal (wxyz convention)
        normal = F.normalize(
            torch.cross(dx.reshape(-1, 3), dy.reshape(-1, 3), dim=-1), dim=-1
        )                                                        # (H*W, 3)
        quats = _normal_to_quat_wxyz(normal)                    # (H*W, 4)

        # ── Camera pose for renderer (world→cam 4×4)
        viewmat = _sim3_to_w2c(kf.T_WC)                        # (4, 4)

        # ── Per-pixel GT depth — must match rendered depth frame.
        # Rendered depth = s*z_cam (Sim3 scale baked into world points,
        # stripped from viewmat).  Multiply camera z by scale to match.
        sim3_data = kf.T_WC.data.reshape(-1)
        sim3_scale = float(sim3_data[-1].exp())   # lietorch Sim3 stores log(s)
        depth_gt = (X[:, 2] * sim3_scale).reshape(H_img, W_img).clamp(min=1e-3)
        conf_map = conf_norm.reshape(H_img, W_img)                  # (H, W) in (0,1)

        all_means.append(means_world[valid])
        all_sh_dc.append(sh_dc[valid])
        all_log_scales.append(log_scales[valid])
        all_quats.append(quats[valid])
        all_opacity.append(opacity_logit[valid])
        all_viewmats.append(viewmat.to(device))
        all_gt_images.append(kf.uimg.to(device))                # (H, W, 3)
        all_depth_maps.append(depth_gt.to(device))              # (H, W)
        all_conf_maps.append(conf_map.to(device))               # (H, W)
        if use_calib and kf.K is not None:
            all_Ks.append(kf.K)

    total = sum(t.shape[0] for t in all_means)
    print(f"[GS] Initialised {total:,} Gaussians across {n_kf} keyframes.")

    result = {
        "means":      torch.cat(all_means,      dim=0).requires_grad_(True),
        "sh_dc":      torch.cat(all_sh_dc,      dim=0).requires_grad_(True),
        "log_scales": torch.cat(all_log_scales, dim=0).requires_grad_(True),
        "quats":      torch.cat(all_quats,      dim=0).requires_grad_(True),
        "opacity":    torch.cat(all_opacity,    dim=0).requires_grad_(True),
        "viewmats":   torch.stack(all_viewmats),                # (N_kf, 4, 4)
        "gt_images":  all_gt_images,
        "depth_maps": all_depth_maps,                           # list[(H, W)]
        "conf_maps":  all_conf_maps,                            # list[(H, W)]
        "Ks":         torch.stack(all_Ks) if all_Ks else None,
    }
    return result


# ── 2. Training loop ───────────────────────────────────────────────────────────

def train_gaussian_splat(
    keyframes: SharedKeyframes,
    K,
    save_dir: Path,
    seq_name: str,
    use_calib: bool,
    cfg: dict,
    threshold: float = None,
) -> None:
    """
    Full offline 3DGS training from MASt3R-SLAM keyframes.

    Args:
        keyframes:  Final SharedKeyframes buffer.
        K:          Camera intrinsic tensor (3,3) or None (uncalibrated).
        save_dir:   Directory for output PLY.
        seq_name:   Sequence name used as filename stem.
        use_calib:  Whether calibrated mode was used.
        cfg:        gaussian_splat config dict (from config/base.yaml).
        threshold:  Hard cap on max Gaussians after initialisation; overrides
                    cfg['max_gaussians'] when set.  Guards against memory OOM
                    on very long sequences.  None means use cfg['max_gaussians'].
    """
    from gsplat import rasterization

    conf_thresh = cfg.get("c_conf_threshold", 0.5)
    max_gaussians = threshold if threshold is not None else cfg.get("max_gaussians", 2_000_000)

    data = extract_gaussians(keyframes, use_calib, threshold=conf_thresh)

    # Hard cap: keep highest-confidence Gaussians if count exceeds max_gaussians
    n_init = data["means"].shape[0]
    if n_init > max_gaussians:
        print(f"[GS] Capping {n_init:,} → {max_gaussians:,} Gaussians (threshold={max_gaussians})")
        keep_idx = torch.topk(
            torch.sigmoid(data["opacity"].detach()), max_gaussians
        ).indices
        data = _index_data(data, keep_idx)

    n_kf = data["viewmats"].shape[0]
    device = data["means"].device
    H_img, W_img = data["gt_images"][0].shape[:2]

    # ── Camera intrinsics for renderer
    if use_calib and data["Ks"] is not None:
        # (N_kf, 3, 3) — one K per keyframe
        Ks_render = data["Ks"].to(device)
        if Ks_render.dim() == 2:
            Ks_render = Ks_render.unsqueeze(0).expand(n_kf, -1, -1).contiguous()
    else:
        # Pinhole fallback for uncalibrated runs
        fx = fy = float(max(H_img, W_img))
        cx, cy = W_img / 2.0, H_img / 2.0
        K_fb = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=device,
        )
        Ks_render = K_fb.unsqueeze(0).expand(n_kf, -1, -1).contiguous()

    # ── Optimiser (per-parameter learning rates from config)
    params = [
        {"params": [data["means"]],      "lr": cfg.get("lr_means",   1.6e-4)},
        {"params": [data["sh_dc"]],      "lr": cfg.get("lr_sh",      2.5e-3)},
        {"params": [data["log_scales"]], "lr": cfg.get("lr_scales",  5e-3)},
        {"params": [data["quats"]],      "lr": cfg.get("lr_quats",   1e-3)},
        {"params": [data["opacity"]],    "lr": cfg.get("lr_opacity", 5e-2)},
    ]
    optimizer = torch.optim.Adam(params, eps=1e-15)

    n_iters            = cfg.get("n_iters",                10_000)
    densify_from       = cfg.get("densify_from_iter",         500)
    densify_until      = cfg.get("densify_until_iter",       5_000)
    densify_interval   = cfg.get("densify_interval",           100)
    opacity_reset_int  = cfg.get("opacity_reset_interval",   3_000)
    prune_thresh       = cfg.get("prune_opacity_thresh",      0.005)
    grad_thresh        = cfg.get("grad_thresh",               2e-4)
    lambda_ssim        = cfg.get("lambda_ssim",               0.2)
    lambda_depth       = cfg.get("lambda_depth",              0.1)
    depth_min_conf     = cfg.get("depth_min_conf",            0.3)

    print(f"[GS] Training {data['means'].shape[0]:,} Gaussians for {n_iters} iters ...")
    print(f"[GS] Image size: {H_img}×{W_img}  |  Keyframes: {n_kf}")

    absgrad_accum = torch.zeros(data["means"].shape[0], device=device)

    # main.py disables grad globally for SLAM; re-enable for GS optimisation
    torch.set_grad_enabled(True)

    for it in range(n_iters):
        # ── Sample a random keyframe
        kf_idx = torch.randint(0, n_kf, (1,)).item()
        viewmat = data["viewmats"][kf_idx].unsqueeze(0)         # (1, 4, 4)
        gt      = data["gt_images"][kf_idx]                     # (H, W, 3)
        K_i     = Ks_render[kf_idx].unsqueeze(0)               # (1, 3, 3)

        # ── Rasterise RGB + camera-z depth in one pass
        # render_mode="RGB+D": output shape (1, H, W, 4); last channel = depth (m)
        # packed=False: ensures grad_fn even when few Gaussians project onto image.
        render_colors, render_alphas, meta = rasterization(
            means       = data["means"],
            quats       = data["quats"],                        # wxyz
            scales      = torch.exp(data["log_scales"]),
            opacities   = torch.sigmoid(data["opacity"]),       # (N,)
            colors      = data["sh_dc"].unsqueeze(1),           # (N, 1, 3) DC SH
            viewmats    = viewmat,
            Ks          = K_i,
            width       = W_img,
            height      = H_img,
            sh_degree   = 0,
            render_mode = "RGB+D",
            absgrad     = True,
            packed      = False,
        )
        render      = render_colors[0, :, :, :3].clamp(0.0, 1.0)  # (H, W, 3)
        render_dep  = render_colors[0, :, :, 3]                    # (H, W) camera z

        # Guard: skip frames where no Gaussians were visible
        if render.grad_fn is None:
            continue

        # ── Combined loss: (1-λ)*L1 + λ*D-SSIM + λ_depth*L_depth
        l_l1     = (render - gt).abs().mean()
        l_d_ssim = (1.0 - _ssim(render, gt)) / 2.0
        loss     = (1.0 - lambda_ssim) * l_l1 + lambda_ssim * l_d_ssim

        if lambda_depth > 0.0:
            gt_dep   = data["depth_maps"][kf_idx]               # (H, W) MASt3R depth
            conf_w   = data["conf_maps"][kf_idx]                # (H, W) confidence
            valid    = (gt_dep > 1e-3) & (render_dep > 1e-3) & (conf_w > depth_min_conf)
            if valid.any():
                l_depth = (conf_w[valid] * (
                    render_dep[valid].log() - gt_dep[valid].log()
                ).abs()).mean()
                loss = loss + lambda_depth * l_depth

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate abs 2D gradient for densification
        if "means2d" in meta and meta["means2d"].absgrad is not None:
            absgrad_accum += meta["means2d"].absgrad.squeeze(0).norm(dim=-1).detach()

        # Gradient clipping: prevents single large step exploding means
        torch.nn.utils.clip_grad_norm_(
            [data["means"], data["log_scales"], data["quats"], data["opacity"]],
            max_norm=1.0,
        )
        optimizer.step()

        # ── Keep quats normalised
        with torch.no_grad():
            data["quats"].data = F.normalize(data["quats"].data, dim=-1)

        # ── Opacity reset: flatten all to ~0.12 periodically so dead Gaussians surface
        if it > 0 and it % opacity_reset_int == 0:
            with torch.no_grad():
                data["opacity"].data.fill_(-2.0)               # sigmoid(-2) ≈ 0.12
            absgrad_accum.zero_()

        # ── Adaptive density control
        if densify_from <= it < densify_until and it % densify_interval == 0:
            # Prune low-opacity first (reduces count before potential split)
            prune_mask = _prune_mask(data, threshold=prune_thresh)
            if prune_mask.any():
                _apply_mask(data, optimizer, ~prune_mask)
                absgrad_accum = absgrad_accum[~prune_mask]

            # Clone Gaussians with high 2D gradient (under-reconstructed regions)
            if absgrad_accum.shape[0] == data["means"].shape[0]:
                clone_mask = (absgrad_accum / max(it, 1)) > grad_thresh
                if clone_mask.any():
                    _clone(data, optimizer, clone_mask, max_gaussians, threshold=max_gaussians)
                    extra = clone_mask.sum().item()
                    absgrad_accum = torch.cat([
                        absgrad_accum, torch.zeros(extra, device=device)
                    ])
            absgrad_accum.zero_()

        if it % 500 == 0:
            with torch.no_grad():
                psnr_val = _psnr(render.detach(), gt)
                ssim_val = _ssim(render.detach(), gt).item()
            n_gs = data["means"].shape[0]
            print(
                f"[GS] iter {it:5d}/{n_iters} | loss {loss.item():.4f} | "
                f"PSNR {psnr_val:.2f} dB | SSIM {ssim_val:.4f} | N={n_gs:,}"
            )

    out_path = save_dir / f"{seq_name}_gs.ply"
    _save_splat(out_path, data)
    print(f"[GS] Saved → {out_path}")


# ── 3. Helpers ─────────────────────────────────────────────────────────────────

def _psnr(render: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB. Inputs: (H,W,3) float in [0,1]."""
    mse = (render - gt).pow(2).mean().clamp(min=1e-10)
    return -10.0 * math.log10(mse.item())


def _ssim(render: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> float:
    """Structural Similarity Index (mean over spatial map). Inputs: (H,W,3)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Build 1-D Gaussian kernel, expand to 2-D separable conv
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=render.device)
    coords -= window_size // 2
    kernel_1d = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]          # (W, W)
    kernel = kernel_2d.expand(3, 1, window_size, window_size)    # (3, 1, W, W)

    pad = window_size // 2

    def _filt(x):
        # x: (H, W, 3) → (1, 3, H, W)
        x_ = x.permute(2, 0, 1).unsqueeze(0)
        return F.conv2d(x_, kernel, padding=pad, groups=3).squeeze(0).permute(1, 2, 0)

    mu_x   = _filt(render)
    mu_y   = _filt(gt)
    mu_xx  = _filt(render * render)
    mu_yy  = _filt(gt    * gt)
    mu_xy  = _filt(render * gt)

    sig_x  = mu_xx - mu_x * mu_x
    sig_y  = mu_yy - mu_y * mu_y
    sig_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sig_x + sig_y + C2)
    return (num / den.clamp(min=1e-10)).mean()


def _prune_mask(data: dict, threshold: float = 0.005) -> torch.Tensor:
    """Return bool mask of Gaussians to prune (True = prune).

    Args:
        threshold: Opacity below which a Gaussian is considered dead.
                   Lower values keep more Gaussians; raise to prune harder.
    """
    return torch.sigmoid(data["opacity"]).detach() < threshold


def _clone(data: dict, optimizer, mask: torch.Tensor, current_max: int, threshold: int) -> None:
    """Clone Gaussians indicated by mask; respects hard cap `threshold`."""
    n_existing = data["means"].shape[0]
    n_clone = mask.sum().item()
    allowed = max(0, threshold - n_existing)
    if allowed == 0:
        return
    n_clone = min(n_clone, allowed)
    clone_idx = mask.nonzero(as_tuple=True)[0][:n_clone]

    param_keys = ["means", "sh_dc", "log_scales", "quats", "opacity"]
    with torch.no_grad():
        for key in param_keys:
            cloned = data[key][clone_idx].detach().clone().requires_grad_(True)
            # Small position jitter so clones don't sit on top of originals
            if key == "means":
                jitter = torch.randn_like(cloned) * 0.001
                cloned = (cloned + jitter).detach().requires_grad_(True)
            new_tensor = torch.cat([data[key], cloned], dim=0).requires_grad_(True)
            old_p = data[key]
            data[key] = new_tensor
            # Extend Adam state
            state = optimizer.state.pop(old_p, {})
            new_state = {}
            if "exp_avg" in state:
                pad = torch.zeros(n_clone, *state["exp_avg"].shape[1:],
                                  device=state["exp_avg"].device)
                new_state["exp_avg"]    = torch.cat([state["exp_avg"],    pad], dim=0)
                new_state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], pad], dim=0)
                new_state["step"] = state["step"]
            optimizer.state[new_tensor] = new_state
            for group in optimizer.param_groups:
                if group["params"][0] is old_p:
                    group["params"][0] = new_tensor
                    break


def _apply_mask(data: dict, optimizer, keep: torch.Tensor) -> None:
    """Compact all per-Gaussian tensors and Adam state to `keep` indices."""
    param_keys = ["means", "sh_dc", "log_scales", "quats", "opacity"]
    for key in param_keys:
        old_p = data[key]
        new_p = old_p[keep].detach().requires_grad_(True)
        data[key] = new_p
        state = optimizer.state.pop(old_p, {})
        new_state = {}
        if "exp_avg" in state:
            new_state["exp_avg"]    = state["exp_avg"][keep]
            new_state["exp_avg_sq"] = state["exp_avg_sq"][keep]
            new_state["step"]       = state["step"]
        optimizer.state[new_p] = new_state
        for group in optimizer.param_groups:
            if group["params"][0] is old_p:
                group["params"][0] = new_p
                break


def _index_data(data: dict, idx: torch.Tensor) -> dict:
    """Return a new data dict with only the Gaussians at `idx`."""
    param_keys = ["means", "sh_dc", "log_scales", "quats", "opacity"]
    out = dict(data)
    for key in param_keys:
        out[key] = data[key][idx].detach().requires_grad_(True)
    return out


def _normal_to_quat_wxyz(normals: torch.Tensor) -> torch.Tensor:
    """Convert surface normals to wxyz quaternions (aligns z-axis with normal).

    Args:
        normals: (N, 3) unit vectors.

    Returns:
        (N, 3) wxyz unit quaternions.
    """
    z = torch.zeros_like(normals)
    z[:, 2] = 1.0                                               # canonical z-axis

    cross = torch.cross(z, normals, dim=-1)                     # rotation axis
    dot   = (z * normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    # cos and sin of half-angle
    cos_h = ((1.0 + dot) / 2.0).clamp(min=0.0).sqrt()          # (N, 1)
    sin_h = torch.linalg.norm(cross, dim=-1, keepdim=True) / 2.0

    axis_norm = F.normalize(cross, dim=-1)                      # (N, 3)
    xyz = axis_norm * sin_h                                     # (N, 3)

    # Handle degenerate case: normal ≈ ±z
    # When normal ≈ +z: identity quaternion [1,0,0,0]
    # When normal ≈ -z: 180° rotation around x-axis [0,1,0,0]
    near_pos_z = dot.squeeze(-1) > 0.9999
    near_neg_z = dot.squeeze(-1) < -0.9999

    w = cos_h                                                   # (N, 1)
    quats = torch.cat([w, xyz], dim=-1)                         # (N, 4) wxyz

    quats[near_pos_z] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=normals.device)
    quats[near_neg_z] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device)

    return F.normalize(quats, dim=-1)


def _sim3_to_w2c(T_WC) -> torch.Tensor:
    """Convert a lietorch.Sim3 camera-to-world pose to a 4×4 world-to-cam matrix.

    as_SE3 strips scale; the world-frame points (T_WC.act(X_canon)) already
    have scale baked in, so the stripped SE3 is geometrically consistent.
    """
    T_se3 = as_SE3(T_WC)
    data = T_se3.data.reshape(-1)                               # (7,) t(3) q(4) xyzw
    t = data[:3]
    q_xyzw = data[3:7]

    # lietorch stores xyzw; convert to rotation matrix
    R = _quat_xyzw_to_matrix(q_xyzw)                           # (3, 3) camera→world

    # w2c: R^T and -R^T @ t
    w2c = torch.eye(4, device=t.device, dtype=t.dtype)
    w2c[:3, :3] = R.T
    w2c[:3,  3] = -(R.T @ t)
    return w2c


def _quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion → 3×3 rotation matrix."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(3, 3)


def _extract_single_keyframe(kf, use_calib: bool, threshold: float = 0.5):
    """Extract Gaussian init data from one keyframe.

    Returns a plain dict (no requires_grad) or None if no pixels pass threshold.
    """
    device = kf.X_canon.device
    X = kf.X_canon
    if use_calib and kf.K is not None:
        X = constrain_points_to_ray(
            kf.img_shape.flatten()[:2], X[None], kf.K
        ).squeeze(0)
    means_world = kf.T_WC.act(X)

    conf = kf.get_average_conf().reshape(-1)
    conf_max = conf.max().clamp(min=1e-6)
    conf_norm = (conf / conf_max).clamp(1e-4, 1 - 1e-4)
    valid = conf_norm > threshold
    if not valid.any():
        return None

    opacity_logit = torch.log(conf_norm / (1.0 - conf_norm))
    rgb = kf.uimg.reshape(-1, 3).to(device)
    sh_dc = (rgb - 0.5) / SH_C0

    H_img, W_img = kf.img_shape.flatten()[:2].tolist()
    H_img, W_img = int(H_img), int(W_img)
    X_grid = X.reshape(H_img, W_img, 3)
    dx = F.pad(X_grid[:, 1:] - X_grid[:, :-1], (0, 0, 0, 1))
    dy = F.pad(X_grid[1:] - X_grid[:-1], (0, 0, 0, 0, 0, 1))
    sx = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp(min=1e-6)
    sy = torch.linalg.norm(dy, dim=-1, keepdim=True).clamp(min=1e-6)
    sz = (sx + sy) / 2.0
    log_scales = torch.log(torch.cat([sx, sy, sz], dim=-1)).reshape(-1, 3)
    normal = F.normalize(
        torch.cross(dx.reshape(-1, 3), dy.reshape(-1, 3), dim=-1), dim=-1
    )
    quats = _normal_to_quat_wxyz(normal)

    sim3_data = kf.T_WC.data.reshape(-1)
    sim3_scale = float(sim3_data[-1].exp())
    depth_gt = (X[:, 2] * sim3_scale).reshape(H_img, W_img).clamp(min=1e-3)
    conf_map = conf_norm.reshape(H_img, W_img)
    viewmat = _sim3_to_w2c(kf.T_WC).to(device)

    return {
        "means":      means_world[valid].detach(),
        "sh_dc":      sh_dc[valid].detach(),
        "log_scales": log_scales[valid].detach(),
        "quats":      quats[valid].detach(),
        "opacity":    opacity_logit[valid].detach(),
        "viewmat":    viewmat,
        "gt_image":   kf.uimg.to(device),
        "depth_map":  depth_gt,
        "conf_map":   conf_map,
        "K":          kf.K if (use_calib and kf.K is not None) else None,
        "H_img":      H_img,
        "W_img":      W_img,
    }


def _append_keyframe_data(data: dict, optimizer, kf_data: dict) -> int:
    """Append one keyframe's Gaussians into the live data dict and Adam state.

    Returns the number of new Gaussians added.
    """
    param_keys = ["means", "sh_dc", "log_scales", "quats", "opacity"]
    n_new = kf_data["means"].shape[0]

    for key in param_keys:
        old_p = data[key]
        new_p = torch.cat(
            [old_p.detach(), kf_data[key].to(old_p.device)], dim=0
        ).requires_grad_(True)
        state = optimizer.state.pop(old_p, {})
        new_state = {}
        if "exp_avg" in state:
            pad = torch.zeros(n_new, *state["exp_avg"].shape[1:],
                              device=state["exp_avg"].device)
            new_state["exp_avg"]    = torch.cat([state["exp_avg"], pad])
            new_state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], pad])
            new_state["step"]       = state["step"]
        optimizer.state[new_p] = new_state
        for group in optimizer.param_groups:
            if group["params"][0] is old_p:
                group["params"][0] = new_p
                break
        data[key] = new_p

    new_vm = kf_data["viewmat"].unsqueeze(0).to(data["viewmats"].device)
    data["viewmats"] = torch.cat([data["viewmats"], new_vm], dim=0)
    data["gt_images"].append(kf_data["gt_image"])
    data["depth_maps"].append(kf_data["depth_map"])
    data["conf_maps"].append(kf_data["conf_map"])
    if data["Ks"] is not None and kf_data["K"] is not None:
        data["Ks"] = torch.cat([data["Ks"], kf_data["K"].unsqueeze(0)], dim=0)

    return n_new


# ── 4. Online Gaussian Splatting ───────────────────────────────────────────────

class OnlineGaussianSplat:
    """Incrementally builds and trains a 3DGS scene alongside MASt3R-SLAM.

    Called from the backend process after each GN optimization solve.
    All training happens in the backend; results are saved to disk at termination.

    All public methods accept a `threshold` guard (inherited from cfg) to prevent
    Gaussians boom as new keyframes are added on long sequences.
    """

    def __init__(self, cfg: dict, use_calib: bool, K, device: str):
        self.cfg          = cfg
        self.use_calib    = use_calib
        self.K_global     = K
        self.device       = device

        self.data         = None   # Gaussian parameters + per-KF rendering data
        self.optimizer    = None
        self.n_kf         = 0
        self.train_step   = 0
        self.absgrad_accum = None

        self.H_img     = None
        self.W_img     = None
        self.Ks_render = None      # (K, 3, 3)

        self.conf_thresh    = cfg.get("c_conf_threshold",    0.5)
        self.max_gaussians  = cfg.get("max_gaussians",  2_000_000)
        self.lambda_ssim    = cfg.get("lambda_ssim",         0.2)
        self.lambda_depth   = cfg.get("lambda_depth",        0.1)
        self.depth_min_conf = cfg.get("depth_min_conf",      0.3)
        self.prune_thresh   = cfg.get("prune_opacity_thresh", 0.005)
        self.grad_thresh    = cfg.get("grad_thresh",         2e-4)
        self.densify_online = cfg.get("densify_online",      True)

        # Aux (non-keyframe) photometric supervision
        self.aux_frames     = []   # list of {gt_image, viewmat, weight}
        self.aux_ratio      = cfg.get("aux_ratio",      0.3)
        self.aux_max_frames = cfg.get("aux_max_frames", 200)
        self.heartbeat_sec  = cfg.get("heartbeat_sec", 30)
        self._last_heartbeat_ts = time.time()

    # ── Public API ──────────────────────────────────────────────────────────

    def add_keyframe(self, kf, threshold: float = None) -> bool:
        """Extract Gaussians from kf and append to the live scene.

        Does NOT run training — call train_gaussians() after sync_poses().
        threshold: override conf_thresh; prevents Gaussians boom on this KF.
        """
        thr = threshold if threshold is not None else self.conf_thresh
        kf_data = _extract_single_keyframe(kf, self.use_calib, thr)
        if kf_data is None:
            print(f"[GS Online] KF {kf.frame_id}: no valid Gaussians above threshold.")
            return False

        if self.H_img is None:
            self.H_img = kf_data["H_img"]
            self.W_img = kf_data["W_img"]

        n_new = kf_data["means"].shape[0]

        if self.data is None:
            self.data = {
                "means":      kf_data["means"].requires_grad_(True),
                "sh_dc":      kf_data["sh_dc"].requires_grad_(True),
                "log_scales": kf_data["log_scales"].requires_grad_(True),
                "quats":      kf_data["quats"].requires_grad_(True),
                "opacity":    kf_data["opacity"].requires_grad_(True),
                "viewmats":   kf_data["viewmat"].unsqueeze(0),
                "gt_images":  [kf_data["gt_image"]],
                "depth_maps": [kf_data["depth_map"]],
                "conf_maps":  [kf_data["conf_map"]],
                "Ks":         (kf_data["K"].unsqueeze(0)
                               if kf_data["K"] is not None else None),
            }
            self._init_optimizer()
            self.absgrad_accum = torch.zeros(n_new, device=self.device)
        else:
            n_before = self.data["means"].shape[0]
            _append_keyframe_data(self.data, self.optimizer, kf_data)
            n_after = self.data["means"].shape[0]
            extra = n_after - n_before
            self.absgrad_accum = torch.cat([
                self.absgrad_accum,
                torch.zeros(extra, device=self.device),
            ])

        self._append_k_render(kf_data["K"])
        self.n_kf += 1
        n_total = self.data["means"].shape[0]
        print(
            f"[GS Online] KF {self.n_kf} added | +{n_new:,} Gaussians "
            f"| total {n_total:,}"
        )
        return True

    def sync_poses(self, keyframes) -> None:
        """Copy latest GN-optimized poses into viewmats (non-differentiable)."""
        if self.data is None or self.n_kf == 0:
            return
        new_vms = []
        for i in range(self.n_kf):
            kf = keyframes[i]
            new_vms.append(_sim3_to_w2c(kf.T_WC).to(self.device))
        self.data["viewmats"] = torch.stack(new_vms)

    def train_gaussians(self, n_steps: int, threshold: float = None) -> None:
        """N Adam steps on Gaussian parameters with combined photometric loss.

        Poses are fixed (use sync_poses before calling).
        threshold: max-Gaussians guard; prevents density control from blooming.
        """
        if self.data is None or self.n_kf == 0 or n_steps == 0:
            return
        max_g = threshold if threshold is not None else self.max_gaussians

        start_step = self.train_step
        start_ts = time.time()
        print(
            f"[GS Online] train start | req_steps={n_steps} | step={start_step} | "
            f"N={self.data['means'].shape[0]:,} | KFs={self.n_kf}",
            flush=True,
        )

        from gsplat import rasterization

        for _ in range(n_steps):
            now = time.time()
            if now - self._last_heartbeat_ts >= self.heartbeat_sec:
                n_gs = self.data["means"].shape[0]
                print(
                    f"[GS Online] heartbeat | train_step={self.train_step} | "
                    f"n_steps_req={n_steps} | N={n_gs:,} | KFs={self.n_kf} | "
                    f"aux={len(self.aux_frames)}"
                )
                self._last_heartbeat_ts = now

            # Sample either a KF (full loss) or an aux non-KF (L1 only)
            use_aux = (
                self.aux_ratio > 0.0
                and len(self.aux_frames) > 0
                and torch.rand(1).item() < self.aux_ratio
            )
            if use_aux:
                aux_idx      = torch.randint(0, len(self.aux_frames), (1,)).item()
                aux          = self.aux_frames[aux_idx]
                viewmat      = aux["viewmat"].unsqueeze(0)
                gt           = aux["gt_image"]
                K_i          = self.Ks_render[0].unsqueeze(0)  # same camera
                frame_weight = aux["weight"]
                kf_idx       = None
            else:
                kf_idx       = torch.randint(0, self.n_kf, (1,)).item()
                viewmat      = self.data["viewmats"][kf_idx].unsqueeze(0)
                gt           = self.data["gt_images"][kf_idx]
                K_i          = self.Ks_render[kf_idx].unsqueeze(0)
                frame_weight = 1.0

            render_colors, _, meta = rasterization(
                means      = self.data["means"],
                quats      = self.data["quats"],
                scales     = torch.exp(self.data["log_scales"]),
                opacities  = torch.sigmoid(self.data["opacity"]),
                colors     = self.data["sh_dc"].unsqueeze(1),
                viewmats   = viewmat,
                Ks         = K_i,
                width      = self.W_img,
                height     = self.H_img,
                sh_degree  = 0,
                render_mode = "RGB+D",
                absgrad    = True,
                packed     = False,
            )
            render     = render_colors[0, :, :, :3].clamp(0.0, 1.0)
            render_dep = render_colors[0, :, :, 3]

            if render.grad_fn is None:
                continue

            l_l1 = (render - gt).abs().mean()
            if use_aux:
                # Aux frames: L1 only (locally-tracked pose — skip depth/SSIM)
                loss = l_l1 * frame_weight
            else:
                l_d_ssim = (1.0 - _ssim(render, gt)) / 2.0
                loss     = (1.0 - self.lambda_ssim) * l_l1 + self.lambda_ssim * l_d_ssim

                if self.lambda_depth > 0.0:
                    gt_dep = self.data["depth_maps"][kf_idx]
                    conf_w = self.data["conf_maps"][kf_idx]
                    valid  = (gt_dep > 1e-3) & (render_dep > 1e-3) & (conf_w > self.depth_min_conf)
                    if valid.any():
                        l_depth = (
                            conf_w[valid]
                            * (render_dep[valid].log() - gt_dep[valid].log()).abs()
                        ).mean()
                        loss = loss + self.lambda_depth * l_depth

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if "means2d" in meta and meta["means2d"].absgrad is not None:
                g = meta["means2d"].absgrad.squeeze(0).norm(dim=-1).detach()
                if g.shape[0] == self.absgrad_accum.shape[0]:
                    self.absgrad_accum += g

            torch.nn.utils.clip_grad_norm_(
                [self.data["means"], self.data["log_scales"],
                 self.data["quats"], self.data["opacity"]],
                max_norm=1.0,
            )
            self.optimizer.step()

            with torch.no_grad():
                self.data["quats"].data = F.normalize(self.data["quats"].data, dim=-1)

            self.train_step += 1

            if self.densify_online and self.train_step % 100 == 0:
                self._density_control(max_g)

        end_ts = time.time()
        done_steps = self.train_step - start_step
        print(
            f"[GS Online] train done  | req_steps={n_steps} | done_steps={done_steps} | "
            f"step={self.train_step} | elapsed={end_ts - start_ts:.1f}s | "
            f"N={self.data['means'].shape[0]:,}",
            flush=True,
        )

    def refine_poses(self, n_steps: int, threshold: float = None) -> None:
        """N photometric gradient steps on per-KF viewmats (Gaussians frozen).

        KF-0 is pinned as the global reference frame.
        threshold: unused here (kept for API consistency).
        """
        if self.data is None or self.n_kf < 2 or n_steps == 0:
            return

        from gsplat import rasterization

        # Differentiable free poses (all except the pinned first KF)
        vm_fixed  = self.data["viewmats"][0].unsqueeze(0).detach()
        vm_free   = self.data["viewmats"][1:].detach().clone().requires_grad_(True)
        pose_opt  = torch.optim.Adam([vm_free], lr=self.cfg.get("lr_pose", 1e-4))

        # Frozen Gaussian inputs
        means_d     = self.data["means"].detach()
        quats_d     = self.data["quats"].detach()
        scales_d    = torch.exp(self.data["log_scales"].detach())
        opacities_d = torch.sigmoid(self.data["opacity"].detach())
        colors_d    = self.data["sh_dc"].detach().unsqueeze(1)

        for _ in range(n_steps):
            kf_idx = torch.randint(1, self.n_kf, (1,)).item()
            vm     = vm_free[kf_idx - 1].unsqueeze(0)
            gt     = self.data["gt_images"][kf_idx]
            K_i    = self.Ks_render[kf_idx].unsqueeze(0)

            render_colors, _, _ = rasterization(
                means=means_d, quats=quats_d,
                scales=scales_d, opacities=opacities_d, colors=colors_d,
                viewmats=vm, Ks=K_i,
                width=self.W_img, height=self.H_img,
                sh_degree=0, packed=False,
            )
            render = render_colors[0, :, :, :3].clamp(0.0, 1.0)
            if render.grad_fn is None:
                continue
            loss = (render - gt).abs().mean()
            pose_opt.zero_grad()
            loss.backward()
            pose_opt.step()

        with torch.no_grad():
            self.data["viewmats"] = torch.cat(
                [vm_fixed, vm_free.detach()], dim=0
            )

    def save(self, save_dir, seq_name: str) -> None:
        """Save the incrementally trained Gaussians as a PLY file."""
        if self.data is None:
            print("[GS Online] Nothing to save.")
            return
        out_path = Path(save_dir) / f"{seq_name}_gs.ply"
        _save_splat(out_path, self.data)
        print(
            f"[GS Online] Saved → {out_path}  "
            f"({self.data['means'].shape[0]:,} Gaussians, "
            f"{self.train_step} total steps across {self.n_kf} keyframes)"
        )

    def add_aux_frames(self, frames_data: list) -> None:
        """Buffer non-KF frames for photometric-only supervision.

        Called from backend after draining the nonkf_queue.  Frames are
        rendered against existing Gaussians during train_gaussians() with
        L1 loss only (no depth, no SSIM) and weighted by temporal decay.

        frames_data: list of dicts with keys
            'uimg'      – (H,W,3) float32 CPU tensor  [0,1]
            'sim3_data' – raw lietorch.Sim3 data CPU tensor
            'weight'    – float in (0, 1]
        """
        if self.data is None or self.H_img is None:
            return
        import lietorch as _lietorch
        for fd in frames_data:
            T_WC    = _lietorch.Sim3(fd["sim3_data"].to(self.device))
            viewmat = _sim3_to_w2c(T_WC).to(self.device)
            gt      = fd["uimg"].to(self.device)
            self.aux_frames.append({
                "gt_image": gt,
                "viewmat":  viewmat,
                "weight":   fd["weight"],
            })
        # Keep only the most recent frames to bound memory
        if len(self.aux_frames) > self.aux_max_frames:
            self.aux_frames = self.aux_frames[-self.aux_max_frames:]
        if frames_data:
            print(
                f"[GS Online] +{len(frames_data)} aux frames | "
                f"buffer {len(self.aux_frames)}/{self.aux_max_frames}"
            )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _init_optimizer(self):
        params = [
            {"params": [self.data["means"]],      "lr": self.cfg.get("lr_means",   1.6e-4)},
            {"params": [self.data["sh_dc"]],      "lr": self.cfg.get("lr_sh",      2.5e-3)},
            {"params": [self.data["log_scales"]], "lr": self.cfg.get("lr_scales",  5e-3)},
            {"params": [self.data["quats"]],      "lr": self.cfg.get("lr_quats",   1e-3)},
            {"params": [self.data["opacity"]],    "lr": self.cfg.get("lr_opacity", 5e-2)},
        ]
        self.optimizer = torch.optim.Adam(params, eps=1e-15)

    def _append_k_render(self, K_kf):
        if self.use_calib and K_kf is not None:
            K = K_kf.to(self.device).unsqueeze(0)
        else:
            fx = fy = float(max(self.H_img, self.W_img))
            cx, cy  = self.W_img / 2.0, self.H_img / 2.0
            K = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0)
        self.Ks_render = (K if self.Ks_render is None
                          else torch.cat([self.Ks_render, K], dim=0))

    def _density_control(self, max_gaussians: int):
        prune_mask = _prune_mask(self.data, threshold=self.prune_thresh)
        if prune_mask.any():
            _apply_mask(self.data, self.optimizer, ~prune_mask)
            self.absgrad_accum = self.absgrad_accum[~prune_mask]

        n = self.data["means"].shape[0]
        if self.absgrad_accum.shape[0] == n and n < max_gaussians:
            clone_mask = (self.absgrad_accum / max(self.train_step, 1)) > self.grad_thresh
            if clone_mask.any():
                _clone(self.data, self.optimizer, clone_mask,
                       max_gaussians, max_gaussians)
                allowed = min(clone_mask.sum().item(), max_gaussians - n)
                self.absgrad_accum = torch.cat([
                    self.absgrad_accum,
                    torch.zeros(allowed, device=self.device),
                ])
        self.absgrad_accum.zero_()


def _save_splat(path: Path, data: dict) -> None:
    """Save Gaussians as a standard 3DGS PLY (compatible with SuperSplat viewer).

    3DGS PLY convention: rot fields are wxyz.
    """
    from plyfile import PlyData, PlyElement

    means      = data["means"].detach().cpu().float().numpy()
    sh_dc      = data["sh_dc"].detach().cpu().float().numpy()
    log_scales = data["log_scales"].detach().cpu().float().numpy()
    quats      = data["quats"].detach().cpu().float().numpy()   # wxyz
    opacity    = data["opacity"].detach().cpu().float().numpy() # logit

    N = means.shape[0]
    dtype = [
        ("x",     "f4"), ("y",     "f4"), ("z",     "f4"),
        ("f_dc_0","f4"), ("f_dc_1","f4"), ("f_dc_2","f4"),
        ("opacity","f4"),
        ("scale_0","f4"), ("scale_1","f4"), ("scale_2","f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3","f4"),
    ]
    arr = np.empty(N, dtype=dtype)
    arr["x"],       arr["y"],       arr["z"]       = means.T
    arr["f_dc_0"],  arr["f_dc_1"],  arr["f_dc_2"]  = sh_dc.T
    arr["opacity"]                                 = opacity
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = log_scales.T
    # rot_0=w, rot_1=x, rot_2=y, rot_3=z  (wxyz → standard 3DGS PLY layout)
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats.T  # already wxyz

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"[GS] PLY written: {path}  ({N:,} Gaussians)")
