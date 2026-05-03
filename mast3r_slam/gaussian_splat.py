"""
Offline Gaussian Splatting reconstruction from MASt3R-SLAM keyframes.

Called after SLAM terminates. Takes final SharedKeyframes (optimized poses +
dense point maps) and trains a 3DGS scene representation.

All public functions accept a `threshold` parameter that gates the maximum
number of Gaussians created/kept, preventing memory blowups on long sequences.
"""

import math
from pathlib import Path

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
        Dict with keys: means, sh_dc, scales (log), quats (wxyz), opacities (logit),
        viewmats (world→cam, (N_kf,4,4)), gt_images (list of (H,W,3) tensors),
        Ks (optional (N_kf,3,3) tensor).
    """
    all_means, all_sh_dc, all_scales = [], [], []
    all_quats, all_opacities = [], []
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

        # ── Per-pixel GT depth (camera z) and confidence — used for depth loss
        depth_gt = X[:, 2].reshape(H_img, W_img).clamp(min=1e-3)   # (H, W)
        conf_map = conf_norm.reshape(H_img, W_img)                  # (H, W) in (0,1)

        all_means.append(means_world[valid])
        all_sh_dc.append(sh_dc[valid])
        all_scales.append(log_scales[valid])
        all_quats.append(quats[valid])
        all_opacities.append(opacity_logit[valid])
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
        "scales":     torch.cat(all_scales,     dim=0).requires_grad_(True),
        "quats":      torch.cat(all_quats,      dim=0).requires_grad_(True),
        "opacities":  torch.cat(all_opacities,  dim=0).requires_grad_(True),
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
    from gsplat import rasterization, DefaultStrategy

    conf_thresh = cfg.get("c_conf_threshold", 0.5)
    max_gaussians = threshold if threshold is not None else cfg.get("max_gaussians", 2_000_000)

    data = extract_gaussians(keyframes, use_calib, threshold=conf_thresh)

    # Hard cap: keep highest-confidence Gaussians if count exceeds max_gaussians
    n_init = data["means"].shape[0]
    if n_init > max_gaussians:
        print(f"[GS] Capping {n_init:,} → {max_gaussians:,} Gaussians (threshold={max_gaussians})")
        keep_idx = torch.topk(
            torch.sigmoid(data["opacities"].detach()), max_gaussians
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

    # ── Separate params dict (Gaussian tensors only).
    # strategy functions (check_sanity, step_pre/post_backward) iterate over ALL
    # keys in params via _update_param_with_optimizer — passing the full data dict
    # (which also contains gt_images lists, depth_maps lists, Ks/None) would crash
    # or silently corrupt non-Gaussian tensors. Use a dedicated 5-key sub-dict.
    _PARAM_KEYS_BASE = ["means", "sh_dc", "scales", "quats", "opacities"]
    params = {k: data[k] for k in _PARAM_KEYS_BASE}

    # ── Per-parameter SelectiveAdam optimisers
    optimizers = _make_splat_optimizers(params, cfg)

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

    # ── DefaultStrategy for adaptive density control
    strategy = DefaultStrategy(
        prune_opa         = prune_thresh,
        grow_grad2d       = grad_thresh,
        refine_start_iter = densify_from,
        refine_stop_iter  = densify_until,
        refine_every      = densify_interval,
        reset_every       = opacity_reset_int,
        absgrad           = True,
        verbose           = False,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state()

    print(f"[GS] Training {params['means'].shape[0]:,} Gaussians for {n_iters} iters ...")
    print(f"[GS] Image size: {H_img}×{W_img}  |  Keyframes: {n_kf}")

    # main.py disables grad globally for SLAM; re-enable for GS optimisation
    torch.set_grad_enabled(True)

    for it in range(n_iters):
        # ── Sample a random keyframe
        kf_idx = torch.randint(0, n_kf, (1,)).item()
        viewmat = data["viewmats"][kf_idx].unsqueeze(0)         # (1, 4, 4)
        gt      = data["gt_images"][kf_idx]                     # (H, W, 3)
        K_i     = Ks_render[kf_idx].unsqueeze(0)               # (1, 3, 3)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        # ── Rasterise RGB + camera-z depth in one pass
        render_colors, render_alphas, meta = rasterization(
            means       = params["means"],
            quats       = params["quats"],                      # wxyz
            scales      = torch.exp(params["scales"]),
            opacities   = torch.sigmoid(params["opacities"]),   # (N,)
            colors      = params["sh_dc"].unsqueeze(1),         # (N, 1, 3) DC SH
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

        # Required by DefaultStrategy._update_state
        meta["n_cameras"] = 1

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

        strategy.step_pre_backward(params, optimizers, strategy_state, it, meta)
        loss.backward()

        # Gradient clipping: prevents single large step exploding means
        torch.nn.utils.clip_grad_norm_(
            [params["means"], params["scales"], params["quats"], params["opacities"]],
            max_norm=1.0,
        )

        n_before = params["means"].shape[0]
        strategy.step_post_backward(params, optimizers, strategy_state, it, meta, packed=False)
        n_after = params["means"].shape[0]

        # Visibility-aware optimizer step; extend to new size if densification changed count
        if n_after == n_before:
            visible = _visibility_from_meta(meta)
        else:
            visible = torch.ones(n_after, dtype=torch.bool, device=device)
        for opt in optimizers.values():
            opt.step(visibility=visible)

        # ── Keep quats normalised
        with torch.no_grad():
            params["quats"].data = F.normalize(params["quats"].data, dim=-1)

        # ── Manual opacity reset (DefaultStrategy's built-in is broken due to operator precedence)
        if it > 0 and it % opacity_reset_int == 0:
            with torch.no_grad():
                params["opacities"].data.fill_(-2.0)             # sigmoid(-2) ≈ 0.12
            if strategy_state.get("grad2d") is not None:
                strategy_state["grad2d"].zero_()
                strategy_state["count"].zero_()

        # ── Global Gaussian count cap
        if params["means"].shape[0] > max_gaussians:
            with torch.no_grad():
                keep_idx = torch.topk(
                    torch.sigmoid(params["opacities"].detach()), max_gaussians
                ).indices
                bool_keep = torch.zeros(params["means"].shape[0], dtype=torch.bool, device=device)
                bool_keep[keep_idx] = True
            _apply_mask_params(params, optimizers, bool_keep, strategy_state)

        if it % 500 == 0:
            with torch.no_grad():
                psnr_val = _psnr(render.detach(), gt)
                ssim_val = _ssim(render.detach(), gt).item()
            n_gs = params["means"].shape[0]
            print(
                f"[GS] iter {it:5d}/{n_iters} | loss {loss.item():.4f} | "
                f"PSNR {psnr_val:.2f} dB | SSIM {ssim_val:.4f} | N={n_gs:,}"
            )

    out_path = save_dir / f"{seq_name}_gs.ply"
    _export_ply(out_path, params)
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


def _gs_param_keys(data: dict) -> list:
    """Return ordered list of per-Gaussian parameter keys present in data.

    Always starts with the fixed base keys; appends optional keys (e.g. sh_rest
    added at fine-refinement stage) if they exist in data.
    """
    base = ["means", "sh_dc", "scales", "quats", "opacities"]
    extra = [k for k in ("sh_rest",) if k in data and data[k] is not None]
    return base + extra


def _index_data(data: dict, idx: torch.Tensor) -> dict:
    """Return a new data dict with only the Gaussians at `idx`."""
    param_keys = ["means", "sh_dc", "scales", "quats", "opacities"]
    out = dict(data)
    for key in param_keys:
        out[key] = data[key][idx].detach().requires_grad_(True)
    return out


def _make_splat_optimizers(params: dict, cfg: dict) -> dict:
    """Create one SelectiveAdam per Gaussian parameter."""
    from gsplat import SelectiveAdam
    lrs = {
        "means":     cfg.get("lr_means",   1.6e-4),
        "sh_dc":     cfg.get("lr_sh",      2.5e-3),
        "scales":    cfg.get("lr_scales",  5e-3),
        "quats":     cfg.get("lr_quats",   1e-3),
        "opacities": cfg.get("lr_opacity", 5e-2),
    }
    return {
        key: SelectiveAdam(
            [{"params": [params[key]], "lr": lr}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for key, lr in lrs.items()
    }


def _apply_mask_params(
    params: dict,
    optimizers: dict,
    keep: torch.Tensor,
    strategy_state: dict = None,
) -> None:
    """Compact all per-Gaussian tensors and per-param optimizer state to `keep` indices."""
    for key in _gs_param_keys(params):
        old_p = params[key]
        new_p = old_p[keep].detach().requires_grad_(True)
        params[key] = new_p
        if key not in optimizers:
            continue
        opt = optimizers[key]
        old_state = opt.state.pop(old_p, {})
        new_state = {}
        if "exp_avg" in old_state:
            new_state = {
                "exp_avg":    old_state["exp_avg"][keep],
                "exp_avg_sq": old_state["exp_avg_sq"][keep],
                "step":       old_state["step"],
            }
        opt.state[new_p] = new_state
        opt.param_groups[0]["params"] = [new_p]
    if strategy_state is not None:
        for k in ("grad2d", "count"):
            if strategy_state.get(k) is not None:
                strategy_state[k] = strategy_state[k][keep]


def _visibility_from_meta(meta: dict) -> torch.Tensor:
    """Extract per-Gaussian visibility (N,) bool from rasterization meta.

    gsplat radii shape is (..., C, N, 2) — the last dim holds x/y pixel radii.
    The standard gsplat idiom is (radii > 0).all(dim=-1) to reduce that dim first.
    SelectiveAdam CUDA kernel requires a 1-D Bool tensor.
    """
    radii = meta["radii"]                    # (..., C, N, 2) unpacked or (nnz, 2) packed
    visible = (radii > 0).all(dim=-1)        # (..., C, N) — reduces x/y radii dim
    while visible.dim() > 1:
        visible = visible.any(dim=0)         # reduce batch/camera dims → (N,)
    return visible                           # bool (N,)


def _export_ply(path: Path, params: dict) -> None:
    """Save Gaussians to PLY using gsplat.export_splats (fixes f_rest channel ordering)."""
    from gsplat import export_splats
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = params["means"].shape[0]
    sh_rest = params.get("sh_rest")
    shN = sh_rest.detach().cpu() if sh_rest is not None else torch.zeros(N, 0, 3)
    export_splats(
        means     = params["means"].detach().cpu(),
        scales    = torch.exp(params["scales"]).detach().cpu(),       # actual, not log
        quats     = params["quats"].detach().cpu(),
        opacities = torch.sigmoid(params["opacities"]).detach().cpu(),  # actual, not logit
        sh0       = params["sh_dc"].detach().cpu().unsqueeze(1),      # (N, 1, 3)
        shN       = shN,                                               # (N, K, 3)
        format    = "ply",
        save_to   = str(path),
    )
    print(f"[GS] PLY written: {path}  ({N:,} Gaussians)")


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
