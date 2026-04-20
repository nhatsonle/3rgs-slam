"""
Offline Gaussian Splatting reconstruction from MASt3R-SLAM keyframes.

Called after SLAM terminates. Takes final SharedKeyframes (optimized poses +
dense point maps) and trains a 3DGS scene representation.

All public functions accept a `threshold` parameter that gates the maximum
number of Gaussians created/kept, preventing memory blowups on long sequences.
"""

import math
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

        all_means.append(means_world[valid])
        all_sh_dc.append(sh_dc[valid])
        all_log_scales.append(log_scales[valid])
        all_quats.append(quats[valid])
        all_opacity.append(opacity_logit[valid])
        all_viewmats.append(viewmat.to(device))
        all_gt_images.append(kf.uimg.to(device))                # (H, W, 3)
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

        # ── Rasterise (absgrad=True so we can accumulate 2D grad for densify)
        # packed=False: ensures render_colors always has grad_fn even if
        # no Gaussians project onto the image (packed=True returns bare zeros).
        render_colors, render_alphas, meta = rasterization(
            means     = data["means"],
            quats     = data["quats"],                          # wxyz
            scales    = torch.exp(data["log_scales"]),
            opacities = torch.sigmoid(data["opacity"]),         # (N,)
            colors    = data["sh_dc"].unsqueeze(1),             # (N, 1, 3) DC SH
            viewmats  = viewmat,
            Ks        = K_i,
            width     = W_img,
            height    = H_img,
            sh_degree = 0,
            absgrad   = True,
            packed    = False,
        )
        render = render_colors[0].clamp(0.0, 1.0)              # (H, W, 3)

        # Guard: skip frames where no Gaussians were visible
        if render.grad_fn is None:
            continue

        # ── L1 photometric loss
        loss = torch.abs(render - gt).mean()

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
            n_gs = data["means"].shape[0]
            print(f"[GS] iter {it:5d}/{n_iters} | loss {loss.item():.4f} | N={n_gs:,}")

    out_path = save_dir / f"{seq_name}_gs.ply"
    _save_splat(out_path, data)
    print(f"[GS] Saved → {out_path}")


# ── 3. Helpers ─────────────────────────────────────────────────────────────────

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
