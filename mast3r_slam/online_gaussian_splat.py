"""
Online Gaussian Splatting mapping following the MASt3R-GS paper.

Paper: "MASt3R-GS: Bridging 3D Reconstruction Priors with Gaussian Splatting
for Real-Time Dense SLAM" (Qingze Wang, 2025)

Architecture:
  - MASt3R-SLAM is the front-end: tracking, keyframe selection, factor graph,
    global pose optimization.  3DGS is a decoupled mapping back-end.
  - The Tracking-Mapping Interface handles:
      1. Depth map generation  — project MASt3R X_canon to image space, filter by
                                  confidence (paper τ_c), produce per-pixel depth.
      2. Pose synchronization  — after factor graph GN, refresh mapping viewmats
                                  from SharedKeyframes.

Two-stage mapping (paper §III-C):
  Stage 1 (coarse online): per-keyframe Gaussian insertion + sliding-window Adam.
    Optimization window = W most-recent keyframes + R random historical keyframes.
    Poses are always fixed to SLAM output (no GS-side pose refinement).
  Stage 2 (fine refinement): after SLAM terminates, final globally-optimised poses
    are synced and high-iteration color refinement runs over all cameras.

Loss (paper Eq.13):
  L_total = alpha_rgb · L_rgb + (1 − alpha_rgb) · L_depth + lambda_iso · L_iso
  where:
    L_rgb   = L1 pixel error with RGB boundary mask
    L_depth = confidence-weighted L1 on log-depth  (Eq.11)
    L_iso   = scale isotropy regularisation          (Eq.12)
  Paper defaults: alpha_rgb = 0.95, lambda_iso = 10, depth conf threshold τ_c = 3.0.
  In this repo confidence is normalised to [0,1]; depth_min_conf maps to τ_c.

All public functions accept a `threshold` parameter to gate/cap Gaussian count.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from mast3r_slam.frame import Mode, SharedKeyframes
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.lietorch_utils import as_SE3

from mast3r_slam.gaussian_splat import (
    SH_C0,
    _apply_mask_params,
    _export_ply,
    _make_splat_optimizers,
    _normal_to_quat_wxyz,
    _psnr,
    _sim3_to_w2c,
    _ssim,
    _visibility_from_meta,
)

PARAM_KEYS: List[str] = ["means", "sh_dc", "scales", "quats", "opacities"]


# ── Tracking-Mapping Interface: keyframe snapshot ──────────────────────────────

def make_keyframe_snapshot(kf_idx: int, frame, use_calib: bool) -> dict:
    """
    Build a picklable dict payload from a live Frame for the GS event queue.

    All tensors are cloned to CPU so the snapshot can cross process boundaries
    without holding references to shared CUDA memory.

    Args:
        kf_idx:    Index into SharedKeyframes for this frame.
        frame:     Frame object freshly written to SharedKeyframes.
        use_calib: Whether calibrated mode (K is available).

    Returns:
        Dict suitable for putting into a multiprocessing manager Queue.
    """
    return {
        "type":      "new_kf",
        "kf_idx":    kf_idx,
        "frame_id":  frame.frame_id,
        "uimg":      frame.uimg.cpu().detach().clone(),              # (H, W, 3)
        "X_canon":   frame.X_canon.reshape(-1, 3).cpu().detach().clone(),  # (H*W, 3)
        "C":         frame.C.reshape(-1, 1).cpu().detach().clone(),  # (H*W, 1) cumulative
        "N":         int(frame.N),
        "T_WC_data": frame.T_WC.data.cpu().detach().clone(),
        "K": (
            frame.K.cpu().detach().clone()
            if (use_calib and frame.K is not None)
            else None
        ),
        "img_shape": (
            int(frame.img_shape.flatten()[0]),
            int(frame.img_shape.flatten()[1]),
        ),
    }


# ── Incremental Gaussian Mapper ────────────────────────────────────────────────

class GaussianMapper:
    """
    Incremental 3D Gaussian mapping back-end.

    Gaussians are inserted one keyframe at a time and optimised photometrically
    within a sliding window.  Poses are always consumed from the SLAM front-end;
    the mapper never optimises camera parameters.
    """

    def __init__(self, cfg: dict, use_calib: bool, device: str = "cuda"):
        self.cfg       = cfg
        self.use_calib = use_calib
        self.device    = device

        # Per-camera state — grows as keyframes arrive
        self.kf_indices: List[int]                     = []
        self.viewmats:   List[torch.Tensor]            = []  # (4,4) w2c, cpu
        self.gt_images:  List[torch.Tensor]            = []  # (H,W,3) cpu [0,1]
        self.depth_maps: List[torch.Tensor]            = []  # (H,W) cpu, cam-z in m
        self.conf_maps:  List[torch.Tensor]            = []  # (H,W) cpu, normalised
        self.Ks_list:    List[Optional[torch.Tensor]]  = []  # (3,3) cpu or None

        # Gaussian parameters (None until first keyframe)
        self.data: Dict[str, Optional[torch.Tensor]] = {k: None for k in PARAM_KEYS}
        self.optimizers:    Optional[Dict[str, object]] = None
        self.strategy:      Optional[object]            = None
        self.strategy_state: Optional[dict]             = None

        self.n_kf:        int            = 0
        self.total_steps: int            = 0
        self.img_shape:   Optional[tuple] = None  # (H, W)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_optimizers(self) -> dict:
        return _make_splat_optimizers(self.data, self.cfg)

    def _estimate_scene_scale(self) -> float:
        """Robust scene-scale estimate for gsplat DefaultStrategy thresholds."""
        means = self.data.get("means")
        if means is None or means.numel() == 0:
            return 1.0
        with torch.no_grad():
            r = torch.linalg.norm(means.detach(), dim=-1)
            q95 = torch.quantile(r, 0.95).item()
        return max(float(q95), 1e-3)

    def _get_K(self, cam_idx: int) -> torch.Tensor:
        """Return (1, 3, 3) intrinsic tensor on self.device for cam_idx."""
        """Return (1, 3, 3) intrinsic tensor on self.device for cam_idx."""
        k = self.Ks_list[cam_idx]
        if k is not None:
            return k.to(self.device).unsqueeze(0)
        H, W = self.img_shape
        fx = fy = float(max(H, W))
        K_fb = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        return K_fb.unsqueeze(0)

    def _extend_gaussians(self, new_gs: Dict[str, torch.Tensor]) -> None:
        """
        Append new Gaussian primitives to the existing state dict and
        surgically extend per-param optimizer state to match.
        """
        n_new = new_gs["means"].shape[0]
        for key in PARAM_KEYS:
            old_p = self.data[key]
            new_p = torch.cat(
                [old_p, new_gs[key].to(self.device)], dim=0
            ).detach().requires_grad_(True)
            self.data[key] = new_p

            if key not in self.optimizers:
                continue
            opt = self.optimizers[key]
            old_state = opt.state.pop(old_p, {})
            new_state: dict = {}
            if "exp_avg" in old_state:
                pad = torch.zeros(n_new, *old_state["exp_avg"].shape[1:], device=self.device)
                new_state["exp_avg"]    = torch.cat([old_state["exp_avg"],    pad])
                new_state["exp_avg_sq"] = torch.cat([old_state["exp_avg_sq"], pad])
                new_state["step"]       = old_state["step"]
            opt.state[new_p] = new_state
            opt.param_groups[0]["params"] = [new_p]

        # Extend strategy running state (grad2d, count) with zeros for new Gaussians
        if self.strategy_state is not None:
            for k in ("grad2d", "count"):
                v = self.strategy_state.get(k)
                if v is not None:
                    pad = torch.zeros(n_new, *v.shape[1:], device=v.device)
                    self.strategy_state[k] = torch.cat([v, pad])

    # ── Tracking-Mapping Interface: depth generation ──────────────────────────

    def _extract_from_snapshot(
        self, snap: dict
    ):
        """
        Depth map generation + Gaussian initialisation from a keyframe snapshot.

        Projects MASt3R X_canon to image space, applies confidence filtering,
        and builds initial Gaussian parameters per paper §III-B-2a.

        Returns:
            new_gs       — dict of raw (unmasked) Gaussian tensors on self.device
            viewmat_cpu  — (4,4) world→cam matrix on CPU
            gt_img_cpu   — (H, W, 3) ground-truth RGB on CPU
            depth_cpu    — (H, W) MASt3R depth (camera Z) on CPU
            conf_cpu     — (H, W) normalised confidence map on CPU
        """
        import lietorch

        device  = self.device
        uimg    = snap["uimg"].to(device)                    # (H, W, 3)
        X_canon = snap["X_canon"].to(device)                 # (H*W, 3) cam frame
        C_raw   = snap["C"].to(device).reshape(-1)           # (H*W,) cumulative conf
        N       = max(snap["N"], 1)
        conf    = (C_raw / N).clamp(min=1e-6)                # average per-pixel conf

        H_img, W_img = snap["img_shape"]

        # Optionally constrain to calibrated ray (same as save_reconstruction)
        if self.use_calib and snap["K"] is not None:
            K_t       = snap["K"].to(device)
            img_shape = torch.tensor([H_img, W_img], device=device, dtype=torch.int)
            X_canon   = constrain_points_to_ray(
                img_shape.flatten()[:2], X_canon[None], K_t
            ).squeeze(0)

        # Normalise confidence and gate valid pixels
        conf_max  = conf.max().clamp(min=1e-6)
        conf_norm = (conf / conf_max).clamp(1e-4, 1 - 1e-4)          # (H*W,) in (0,1)
        threshold = self.cfg.get("c_conf_threshold", 0.5)
        valid     = conf_norm > threshold                              # (H*W,) bool

        # Pose: Sim3 world→world-frame transform
        T_WC = lietorch.Sim3(snap["T_WC_data"].to(device))

        # Extract Sim3 scale factor.
        # lietorch Sim3 data layout: [t(3), q(4), s(1)] — scale stored DIRECTLY (not log).
        # Verified: Sim3.Identity gives data[7]=1.0; act(X) with s=2 gives 2*X.
        # T_WC.act(X) = s * R @ X + t, so rendered depth through the SE3-stripped
        # viewmat gives s * X_z, NOT X_z.
        s_scale = snap["T_WC_data"].to(device).reshape(-1)[7]          # direct scale (≥0)

        # 3-D Gaussian means in world frame
        means_world = T_WC.act(X_canon)                               # (H*W, 3)

        # Opacity logit seeded from normalised confidence
        opacity_logit = torch.log(conf_norm / (1.0 - conf_norm))      # (H*W,)

        # DC colour (SH degree 0)
        sh_dc = (uimg.reshape(-1, 3) - 0.5) / SH_C0                  # (H*W, 3)

        # Anisotropic scale from pixel-grid neighbour offsets.
        # dx/dy are in camera-frame units; multiply by s_scale so Gaussian radii
        # are consistent with means_world (which are in world-frame = s-scaled units).
        X_grid = X_canon.reshape(H_img, W_img, 3)
        dx     = F.pad(X_grid[:, 1:] - X_grid[:, :-1], (0, 0, 0, 1))
        dy     = F.pad(X_grid[1:]    - X_grid[:-1],    (0, 0, 0, 0, 0, 1))
        sx     = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp(min=1e-6) * s_scale
        sy     = torch.linalg.norm(dy, dim=-1, keepdim=True).clamp(min=1e-6) * s_scale

        # z-axis (surface normal direction) = fraction of in-plane size → disc shape.
        # Computed from UNMODIFIED sx/sy so disc_z_factor relationship is preserved
        # regardless of any per-pixel scale boost applied later.
        disc_z_factor = self.cfg.get("disc_z_factor", 0.1)
        sz = (sx + sy) / 2.0 * disc_z_factor
        log_scales = torch.log(torch.cat([sx, sy, sz], dim=-1)).reshape(-1, 3)

        # Global scale bounds: clamp all axes.
        max_log_scale = self.cfg.get("max_log_scale", -2.0)
        min_log_scale = self.cfg.get("min_log_scale", -7.0)
        log_scales[:, :2] = log_scales[:, :2].clamp(min=min_log_scale, max=max_log_scale)
        # Cap z-axis too; allowing unlimited z creates giant splats ("bubbles").
        log_scales[:, 2]  = log_scales[:, 2].clamp(min=min_log_scale, max=max_log_scale)

        # Confidence-modulated scale boost applied IN LOG-SPACE, AFTER the global cap.
        # Low-confidence pixels get a relaxed per-pixel cap so the boost is not erased.
        # Previous design applied boost in linear space BEFORE clamping → neutralised
        # for any pixel where log(sx*boost) > max_log_scale, i.e. most pixels in practice.
        # Design: conf=1 → log_boost=0 (no change); conf=0 → log_boost=log(scale_conf_boost).
        scale_conf_boost = self.cfg.get("scale_conf_boost", 1.0)
        if scale_conf_boost > 1.0:
            scale_conf_gamma = self.cfg.get("scale_conf_gamma", 2.0)
            conf_flat = conf_norm.reshape(-1, 1)                            # (H*W, 1)
            log_boost = math.log(scale_conf_boost) * (1.0 - conf_flat).pow(scale_conf_gamma)
            relaxed_cap = max_log_scale + math.log(scale_conf_boost)        # absolute ceiling
            log_scales[:, :2] = (log_scales[:, :2] + log_boost).clamp(
                min=min_log_scale, max=relaxed_cap
            )

        # Rotation: align Gaussian disc to local surface normal
        normal = F.normalize(
            torch.cross(dx.reshape(-1, 3), dy.reshape(-1, 3), dim=-1), dim=-1
        )
        quats = _normal_to_quat_wxyz(normal)                          # (H*W, 4) wxyz

        # Camera pose for renderer
        viewmat  = _sim3_to_w2c(T_WC)                                 # (4, 4)

        # Depth GT: camera-Z scaled by s_scale to match the rendered depth.
        # _sim3_to_w2c strips scale → rendered_depth = s * X_z, not X_z.
        # Multiplying GT by s ensures: depth_loss = |log(s*X_z) - log(s*X_z)| = 0 at init.
        depth_gt = (X_canon[:, 2] * s_scale).reshape(H_img, W_img).clamp(min=1e-3)
        conf_map = conf_norm.reshape(H_img, W_img)

        new_gs = {
            "means":     means_world[valid].detach(),
            "sh_dc":     sh_dc[valid].detach(),
            "scales":    log_scales[valid].detach(),
            "quats":     quats[valid].detach(),
            "opacities": opacity_logit[valid].detach(),
        }
        return new_gs, viewmat.cpu(), uimg.cpu(), depth_gt.cpu(), conf_map.cpu()

    # ── Public API ────────────────────────────────────────────────────────────

    def insert_keyframe(self, snap: dict, threshold: Optional[int] = None) -> None:
        """
        Incorporate a new keyframe snapshot into the mapper.

        Extracts Gaussian primitives from the keyframe, enforces per-keyframe and
        global caps, and extends the optimiser state.

        Args:
            snap:      Keyframe snapshot dict (from make_keyframe_snapshot).
            threshold: Hard cap on newly inserted Gaussians for this keyframe.
                       Overrides config max_gaussians_per_kf when set.
        """
        from gsplat import DefaultStrategy

        new_gs, viewmat, gt_img, depth_gt, conf_map = self._extract_from_snapshot(snap)

        # Per-keyframe Gaussian cap
        max_per_kf = threshold if threshold is not None else self.cfg.get(
            "max_gaussians_per_kf", 50_000
        )
        n = new_gs["means"].shape[0]
        if n > max_per_kf:
            keep = torch.topk(
                torch.sigmoid(new_gs["opacities"]), max_per_kf
            ).indices
            for k in PARAM_KEYS:
                new_gs[k] = new_gs[k][keep]

        # Register camera data
        self.kf_indices.append(snap["kf_idx"])
        self.viewmats.append(viewmat)
        self.gt_images.append(gt_img)
        self.depth_maps.append(depth_gt)
        self.conf_maps.append(conf_map)
        self.Ks_list.append(
            snap["K"].cpu().detach().clone()
            if (self.use_calib and snap["K"] is not None)
            else None
        )
        if self.img_shape is None:
            self.img_shape = snap["img_shape"]

        cfg = self.cfg
        if self.data["means"] is None:
            # Bootstrap: first keyframe initialises state, strategy, and optimizers
            for k in PARAM_KEYS:
                self.data[k] = new_gs[k].to(self.device).requires_grad_(True)
            self.optimizers = self._make_optimizers()

            prune_thresh = cfg.get("prune_opacity_thresh", 0.005)
            grad_thresh  = cfg.get("grad_thresh",          2e-4)
            self.strategy = DefaultStrategy(
                prune_opa         = prune_thresh,
                grow_grad2d       = grad_thresh,
                refine_start_iter = cfg.get("densify_from_iter",      500),
                refine_stop_iter  = cfg.get("densify_until_iter",    5_000),
                refine_every      = cfg.get("densify_interval",         100),
                reset_every       = cfg.get("opacity_reset_interval", 3_000),
                absgrad           = True,
                verbose           = False,
            )
            self.strategy.check_sanity(self.data, self.optimizers)
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=self._estimate_scene_scale()
            )
        else:
            self._extend_gaussians(new_gs)

        # Global hard cap on total Gaussian count
        max_total = cfg.get("max_gaussians", 2_000_000)
        n_total   = self.data["means"].shape[0]
        if n_total > max_total:
            keep = torch.topk(
                torch.sigmoid(self.data["opacities"].detach()), max_total
            ).indices
            bool_keep = torch.zeros(n_total, dtype=torch.bool, device=self.device)
            bool_keep[keep] = True
            _apply_mask_params(self.data, self.optimizers, bool_keep, self.strategy_state)

        self.n_kf += 1

    def train_step(
        self,
        n_steps:        int  = 1,
        window_size:    int  = 5,
        random_history: int  = 2,
        use_all_cams:   bool = False,
        threshold:      Optional[int] = None,
    ) -> None:
        """
        Run n_steps of photometric Adam optimisation.

        Camera pool follows paper §III-C-a: most-recent W keyframes plus R
        random historical keyframes.  Poses are never optimised here.

        Args:
            n_steps:        Number of Adam gradient steps to run.
            window_size:    Number of recent cameras in the sliding window (W).
            random_history: Additional historical cameras sampled each step (R).
            use_all_cams:   If True, ignore window and sample uniformly from all cameras
                            (used during fine refinement).
            threshold:      Overrides max_gaussians config for densification cap.
        """
        if self.n_kf == 0 or self.data["means"] is None:
            return {"loss": 0.0, "l_rgb": 0.0, "l_depth": 0.0, "l_iso": 0.0, "n_steps": 0}

        from gsplat import rasterization

        cfg = self.cfg

        # Running accumulators for loss logging
        acc = {"loss": 0.0, "l_rgb": 0.0, "l_depth": 0.0, "l_iso": 0.0, "n_steps": 0}

        # Loss weights (paper §III-C-1)
        alpha_rgb      = cfg.get("alpha_rgb",              0.95)
        lambda_depth   = cfg.get("lambda_depth",           0.1)
        depth_min_conf = cfg.get("depth_min_conf",         0.3)
        lambda_iso     = cfg.get("lambda_iso",             10.0)

        # Scale clamping bounds (applied every step to prevent needle/blurry artifacts)
        max_log_scale  = cfg.get("max_log_scale",         -2.0)
        min_log_scale  = cfg.get("min_log_scale",         -7.0)
        # Prune Gaussians where max_scale/min_scale > this ratio (needle pruning)
        max_scale_ratio = cfg.get("max_scale_ratio",      10.0)

        # Density control parameters
        opacity_rst   = cfg.get("opacity_reset_interval", 3_000)
        prune_thresh  = cfg.get("prune_opacity_thresh",   0.005)
        max_total     = threshold if threshold is not None else cfg.get("max_gaussians", 2_000_000)

        H, W = self.img_shape

        # Build camera pool for this step
        if use_all_cams:
            cam_pool = list(range(self.n_kf))
        else:
            recent     = list(range(max(0, self.n_kf - window_size), self.n_kf))
            hist_pool  = list(range(max(0, self.n_kf - window_size)))
            n_hist     = min(random_history, len(hist_pool))
            if hist_pool and n_hist > 0:
                perm       = torch.randperm(len(hist_pool))[:n_hist].tolist()
                historical = [hist_pool[i] for i in perm]
            else:
                historical = []
            cam_pool = recent + historical

        for _ in range(n_steps):
            cam_idx = cam_pool[torch.randint(0, len(cam_pool), (1,)).item()]

            viewmat = self.viewmats[cam_idx].to(self.device).unsqueeze(0)  # (1, 4, 4)
            gt      = self.gt_images[cam_idx].to(self.device)              # (H, W, 3)
            K_i     = self._get_K(cam_idx)                                 # (1, 3, 3)

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # SH degree: use sh_rest if available (added at fine-stage start)
            sh_rest = self.data.get("sh_rest")
            if sh_rest is not None:
                sh_colors  = torch.cat([self.data["sh_dc"].unsqueeze(1), sh_rest], dim=1)
                sh_deg_use = int(round((sh_colors.shape[1] ** 0.5) - 1))  # 1→1, 4→1, 16→3
            else:
                sh_colors  = self.data["sh_dc"].unsqueeze(1)               # (N, 1, 3)
                sh_deg_use = 0

            render_colors, _, meta = rasterization(
                means       = self.data["means"],
                quats       = self.data["quats"],
                scales      = torch.exp(self.data["scales"]),
                opacities   = torch.sigmoid(self.data["opacities"]),
                colors      = sh_colors,
                viewmats    = viewmat,
                Ks          = K_i,
                width       = W,
                height      = H,
                sh_degree   = sh_deg_use,
                render_mode = "RGB+D",
                absgrad     = True,
                packed      = False,
            )
            render     = render_colors[0, :, :, :3].clamp(0.0, 1.0)       # (H, W, 3)
            render_dep = render_colors[0, :, :, 3]                         # (H, W)

            if render.grad_fn is None:
                continue

            # Required by DefaultStrategy._update_state
            meta["n_cameras"] = 1

            # L_rgb: L1 + optional D-SSIM (paper Eq.10 is pure L1; SSIM improves
            # perceptual quality when lambda_ssim > 0 in config)
            l_rgb     = (render - gt).abs().mean()
            lambda_ssim = cfg.get("lambda_ssim", 0.0)
            if lambda_ssim > 0.0:
                l_d_ssim = (1.0 - _ssim(render, gt)) / 2.0
                l_rgb    = (1.0 - lambda_ssim) * l_rgb + lambda_ssim * l_d_ssim

            # L_depth: confidence-weighted log-depth L1 (paper Eq.11)
            l_depth = torch.tensor(0.0, device=self.device)
            if lambda_depth > 0.0:
                gt_dep = self.depth_maps[cam_idx].to(self.device)
                conf_w = self.conf_maps[cam_idx].to(self.device)
                valid  = (gt_dep > 1e-3) & (render_dep > 1e-3) & (conf_w > depth_min_conf)
                if valid.any():
                    l_depth = (
                        conf_w[valid] * (
                            render_dep[valid].log() - gt_dep[valid].log()
                        ).abs()
                    ).mean()

            # L_iso: isotropic scale regularisation (paper Eq.12)
            scales  = torch.exp(self.data["scales"])                       # (N, 3)
            s_bar   = scales.mean(dim=-1, keepdim=True)
            l_iso   = ((scales - s_bar) ** 2).sum(dim=-1).mean()

            # Combined loss (paper Eq.13)
            loss = alpha_rgb * l_rgb + (1.0 - alpha_rgb) * l_depth + lambda_iso * l_iso

            # Accumulate for caller logging
            acc["loss"]    += loss.item()
            acc["l_rgb"]   += l_rgb.item()
            acc["l_depth"] += l_depth.item() if isinstance(l_depth, torch.Tensor) else l_depth
            acc["l_iso"]   += l_iso.item()
            acc["n_steps"] += 1

            self.strategy.step_pre_backward(
                self.data, self.optimizers, self.strategy_state, self.total_steps, meta
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    self.data["means"],
                    self.data["scales"],
                    self.data["quats"],
                    self.data["opacities"],
                ],
                max_norm=1.0,
            )

            n_before = self.data["means"].shape[0]
            self.strategy.step_post_backward(
                self.data, self.optimizers, self.strategy_state,
                self.total_steps, meta, packed=False,
            )
            n_after = self.data["means"].shape[0]

            # Visibility-aware optimizer step
            if n_after == n_before:
                visible = _visibility_from_meta(meta)
            else:
                visible = torch.ones(n_after, dtype=torch.bool, device=self.device)
            for opt in self.optimizers.values():
                opt.step(visibility=visible)

            with torch.no_grad():
                self.data["quats"].data = F.normalize(self.data["quats"].data, dim=-1)
                ls = self.data["scales"].data
                ls.clamp_(min=min_log_scale)
                max_s    = ls.max(dim=-1, keepdim=True).values            # (N,1)
                overflow = (max_s - max_log_scale).clamp(min=0.0)         # (N,1) ≥0
                ls      -= overflow
                ls.clamp_(min=min_log_scale)

            if self.total_steps % 100 == 0:
                with torch.no_grad():
                    s_lin = torch.exp(self.data["scales"].detach())
                    p50 = torch.quantile(s_lin, 0.50, dim=0).tolist()
                    p95 = torch.quantile(s_lin, 0.95, dim=0).tolist()
                    print(
                        "[OnlineGS][debug] scale p50="
                        f"({p50[0]:.4f},{p50[1]:.4f},{p50[2]:.4f}) "
                        "p95="
                        f"({p95[0]:.4f},{p95[1]:.4f},{p95[2]:.4f})"
                    )

            self.total_steps += 1
            s = self.total_steps

            # Manual opacity reset (DefaultStrategy built-in is broken)
            if s > 0 and s % opacity_rst == 0:
                with torch.no_grad():
                    self.data["opacities"].data.fill_(-2.0)
                if self.strategy_state.get("grad2d") is not None:
                    self.strategy_state["grad2d"].zero_()
                    self.strategy_state["count"].zero_()

            # Needle pruning (DefaultStrategy doesn't handle anisotropy ratio)
            with torch.no_grad():
                cur_scales = torch.exp(self.data["scales"].detach())       # (N, 3)
                sorted_s = cur_scales.sort(dim=-1, descending=True).values
                needle_ratio = sorted_s[:, 0] / sorted_s[:, 1].clamp(min=1e-6)
                needle_mask = needle_ratio > max_scale_ratio
            if needle_mask.any():
                keep = ~needle_mask
                _apply_mask_params(self.data, self.optimizers, keep, self.strategy_state)
                # rebuild visibility for new count
                visible = torch.ones(self.data["means"].shape[0], dtype=torch.bool, device=self.device)

            # Global cap
            if self.data["means"].shape[0] > max_total:
                keep_idx = torch.topk(
                    torch.sigmoid(self.data["opacities"].detach()), max_total
                ).indices
                bool_keep = torch.zeros(self.data["means"].shape[0], dtype=torch.bool, device=self.device)
                bool_keep[keep_idx] = True
                _apply_mask_params(self.data, self.optimizers, bool_keep, self.strategy_state)

        return acc

    # ── Tracking-Mapping Interface: pose synchronisation ─────────────────────

    def sync_poses_from_shared(
        self,
        keyframes: SharedKeyframes,
        threshold: Optional[int] = None,
    ) -> None:
        """
        Refresh viewmats for all registered keyframes from SharedKeyframes.

        Called after the backend factor graph Gauss-Newton pass to propagate
        globally-optimised poses into the GS mapping camera list.

        Args:
            keyframes: SharedKeyframes proxy (readable from worker process).
            threshold: Unused; kept for API consistency with other public methods.
        """
        for i, kf_idx in enumerate(self.kf_indices):
            try:
                kf = keyframes[kf_idx]
                self.viewmats[i] = _sim3_to_w2c(kf.T_WC).cpu()
            except Exception:
                pass  # Index out of range or lock timeout — skip silently

    # ── Stage 2: Fine colour refinement ──────────────────────────────────────

    def run_fine_refinement(
        self,
        n_iters:   int = 2000,
        threshold: Optional[int] = None,
    ) -> None:
        """
        High-iteration global colour refinement using final SLAM poses.

        Paper §III-C-b: after factor graph global optimisation is complete,
        run colour refinement over ALL keyframes to improve visual fidelity.

        Three key differences from coarse stage:
          1. Step counter is reset so densification window covers fine stage properly.
          2. Opacity resets are disabled — they destroy well-learnt opacity values.
          3. An initial aggressive prune removes coarse-stage floaters first.

        Args:
            n_iters:   Number of Adam steps (paper uses high iteration count).
            threshold: Overrides max_gaussians for density control during refinement.
        """
        if self.n_kf == 0 or self.data["means"] is None:
            return

        cfg          = self.cfg
        log_interval = cfg.get("fine_log_interval", 500)

        # ── 1. Initial prune: remove low-opacity floaters before fine stage ──
        fine_prune_thresh = cfg.get("fine_prune_thresh", 0.01)
        with torch.no_grad():
            prune_mask = torch.sigmoid(self.data["opacities"].detach()) < fine_prune_thresh
        if prune_mask.any():
            _apply_mask_params(self.data, self.optimizers, ~prune_mask, self.strategy_state)
            print(
                f"[OnlineGS] Initial prune removed {prune_mask.sum().item():,} floaters → "
                f"{self.data['means'].shape[0]:,} remaining"
            )

        # ── 2. Reset step counter → density control window covers fine iters ─
        self.total_steps = 0

        # ── 3. Promote to higher SH degree for view-dependent color ──────────
        fine_sh_degree = cfg.get("fine_sh_degree", 3)
        if fine_sh_degree > 0 and self.data.get("sh_rest") is None:
            n_gs    = self.data["means"].shape[0]
            n_rest  = (fine_sh_degree + 1) ** 2 - 1           # 3 for deg1, 15 for deg3
            sh_rest = torch.zeros(n_gs, n_rest, 3, device=self.device, requires_grad=True)
            self.data["sh_rest"] = sh_rest
            lr_sh_rest = cfg.get("lr_sh_rest", cfg.get("lr_sh", 2.5e-3) / 20.0)
            # Add sh_rest to per-param optimizers dict
            from gsplat import SelectiveAdam
            self.optimizers["sh_rest"] = SelectiveAdam(
                [{"params": [sh_rest], "lr": lr_sh_rest}],
                eps=1e-15,
                betas=(0.9, 0.999),
            )
            print(f"[OnlineGS] Promoted to SH degree {fine_sh_degree} "
                  f"(+{n_rest}×3 coeffs per Gaussian, lr={lr_sh_rest:.2e})")

        # ── 4. Disable opacity resets and depth loss for fine stage ──────────
        saved_opacity_reset = cfg.get("opacity_reset_interval", 3000)
        cfg["opacity_reset_interval"] = n_iters * 100  # effectively ∞
        saved_lambda_depth  = cfg.get("lambda_depth", 0.1)
        cfg["lambda_depth"]  = cfg.get("fine_lambda_depth", 0.0)  # pure photometric

        print(
            f"[OnlineGS] Fine refinement: {n_iters} iters | "
            f"{self.n_kf} cameras | {self.data['means'].shape[0]:,} Gaussians"
        )

        from gsplat import rasterization

        try:
            for it in range(n_iters):
                self.train_step(n_steps=1, use_all_cams=True, threshold=threshold)

                if it % log_interval == 0:
                    # Log average PSNR over all cameras for stable monitoring
                    psnr_sum = 0.0
                    with torch.no_grad():
                        H, W = self.img_shape
                        _sh_rest = self.data.get("sh_rest")
                        if _sh_rest is not None:
                            _sh_col = torch.cat([self.data["sh_dc"].unsqueeze(1), _sh_rest], dim=1)
                            _sh_deg = int(round(_sh_col.shape[1] ** 0.5) - 1)
                        else:
                            _sh_col = self.data["sh_dc"].unsqueeze(1)
                            _sh_deg = 0
                        # Sample at most 20 cameras to keep logging fast
                        max_log_cams = cfg.get("fine_psnr_cams", 20)
                        step = max(1, self.n_kf // max_log_cams)
                        eval_indices = list(range(0, self.n_kf, step))
                        for cam_idx in eval_indices:
                            viewmat = self.viewmats[cam_idx].to(self.device).unsqueeze(0)
                            K_i     = self._get_K(cam_idx)
                            gt      = self.gt_images[cam_idx].to(self.device)
                            out, _, _ = rasterization(
                                means     = self.data["means"],
                                quats     = self.data["quats"],
                                scales    = torch.exp(self.data["scales"]),
                                opacities = torch.sigmoid(self.data["opacities"]),
                                colors    = _sh_col,
                                viewmats  = viewmat,
                                Ks        = K_i,
                                width     = W,
                                height    = H,
                                sh_degree = _sh_deg,
                                render_mode = "RGB+D",
                                packed    = False,
                            )
                            render   = out[0, :, :, :3].clamp(0.0, 1.0)
                            psnr_sum += _psnr(render, gt)
                    avg_psnr = psnr_sum / max(len(eval_indices), 1)
                    print(
                        f"[OnlineGS] Fine {it:5d}/{n_iters} | "
                        f"avg PSNR {avg_psnr:.2f} dB | N={self.data['means'].shape[0]:,}"
                    )
        finally:
            cfg["opacity_reset_interval"] = saved_opacity_reset
            cfg["lambda_depth"]            = saved_lambda_depth

    def save(self, save_dir: Path, seq_name: str) -> None:
        """Save current Gaussian map as a standard 3DGS PLY file."""
        out_path = Path(save_dir) / f"{seq_name}_online_gs.ply"
        _export_ply(out_path, self.data)


# ── Worker process entry point ────────────────────────────────────────────────

def run_online_gs(
    cfg:       dict,
    states,                  # SharedStates multiprocessing proxy
    keyframes: SharedKeyframes,
    save_dir:  str,
    seq_name:  str,
    gs_queue,                # manager.Queue carrying events
) -> None:
    """
    Online GS mapping worker process.

    Consumes events from gs_queue:
      {'type': 'new_kf',   ...snapshot fields...}  — new keyframe from SLAM tracking
      {'type': 'terminate'}                         — SLAM done; run fine refinement

    Idle cycles between events run additional training steps and periodic
    pose synchronisation against SharedKeyframes.
    """
    from mast3r_slam.config import set_global_config

    set_global_config(cfg)
    torch.set_grad_enabled(True)

    device    = "cuda:0"
    gs_cfg    = cfg.get("gaussian_splat", {})
    use_calib = cfg.get("use_calib", False)

    mapper    = GaussianMapper(gs_cfg, use_calib=use_calib, device=device)

    idle_count   = 0
    SYNC_EVERY   = 10   # sync poses every this many idle cycles
    log_interval = gs_cfg.get("coarse_log_interval", 100)

    # Running loss accumulators for coarse stage logging
    _loss_acc: dict = {"loss": 0.0, "l_rgb": 0.0, "l_depth": 0.0, "l_iso": 0.0, "n_steps": 0}

    def _merge_acc(dst: dict, src: dict) -> None:
        dst["loss"]    += src["loss"]
        dst["l_rgb"]   += src["l_rgb"]
        dst["l_depth"] += src["l_depth"]
        dst["l_iso"]   += src["l_iso"]
        dst["n_steps"] += src["n_steps"]

    def _maybe_log_loss(force: bool = False) -> None:
        n = _loss_acc["n_steps"]
        if n == 0:
            return
        if force or (mapper.total_steps % log_interval < gs_cfg.get("idle_train_steps", 5) + 1):
            n_gs = mapper.data["means"].shape[0] if mapper.data["means"] is not None else 0
            print(
                f"[OnlineGS] step={mapper.total_steps:6d} | N={n_gs:,} | "
                f"loss={_loss_acc['loss']/n:.4f} "
                f"(rgb={_loss_acc['l_rgb']/n:.4f} "
                f"dep={_loss_acc['l_depth']/n:.4f} "
                f"iso={_loss_acc['l_iso']/n:.4f})"
            )
            for k in _loss_acc:
                _loss_acc[k] = 0.0

    print("[OnlineGS] Mapping worker started.")

    while True:
        # ── Try to receive an event ──────────────────────────────────────────
        try:
            event = gs_queue.get(timeout=0.05)
        except Exception:
            # No event available — do idle training steps
            if mapper.n_kf > 0 and not states.is_paused():
                acc = mapper.train_step(
                    n_steps        = gs_cfg.get("idle_train_steps", 5),
                    window_size    = gs_cfg.get("window_size",       5),
                    random_history = gs_cfg.get("random_history",    2),
                )
                _merge_acc(_loss_acc, acc)
                _maybe_log_loss()
                idle_count += 1
                if idle_count % SYNC_EVERY == 0:
                    # Periodically pull fresh poses from the SLAM backend
                    mapper.sync_poses_from_shared(keyframes)
            continue

        # ── Dispatch event ───────────────────────────────────────────────────
        etype = event.get("type", "")

        if etype == "new_kf":
            mapper.insert_keyframe(event)
            steps = gs_cfg.get("steps_per_keyframe", 50)
            acc = mapper.train_step(
                n_steps        = steps,
                window_size    = gs_cfg.get("window_size",    5),
                random_history = gs_cfg.get("random_history", 2),
            )
            _merge_acc(_loss_acc, acc)
            n_gs = mapper.data["means"].shape[0] if mapper.data["means"] is not None else 0
            n    = max(_loss_acc["n_steps"], 1)
            print(
                f"[OnlineGS] kf={mapper.n_kf:3d} | step={mapper.total_steps:6d} | N={n_gs:,} | "
                f"loss={_loss_acc['loss']/n:.4f} "
                f"(rgb={_loss_acc['l_rgb']/n:.4f} "
                f"dep={_loss_acc['l_depth']/n:.4f} "
                f"iso={_loss_acc['l_iso']/n:.4f})"
            )
            for k in _loss_acc:
                _loss_acc[k] = 0.0

        elif etype == "terminate":
            print("[OnlineGS] Terminate received — syncing final poses ...")
            if mapper.n_kf > 0:
                # Sync globally optimised poses from the now-finished backend
                mapper.sync_poses_from_shared(keyframes)
                n_fine = gs_cfg.get("n_iters_fine", 2000)
                mapper.run_fine_refinement(n_iters=n_fine)
                mapper.save(Path(save_dir), seq_name)
            break

    print("[OnlineGS] Mapping worker done.")
