"""
GlobalGaussianMap — improved 3DGS backend for MASt3R-SLAM.

Implements the 12-step integration plan:
  Step 2  — MappingFrameSelector:  non-KF frames selected by baseline/conf/cooldown
  Step 3  — LocalFusionBuffer:     sliding-window multi-frame geometry fusion
  Step 4  — Gaussian init from fused geometry (conf-weighted, isotropic init)
  Step 5  — GlobalGaussianMap:     state-tracked Gaussians with observation counts
  Step 6  — VoxelAssociation:      fast spatial association to prevent duplication
  Step 7  — Merge rules:           conf-weighted mean, EMA SH, obs-count update
  Step 8  — Density control:       prune by opacity+obs; clone by 2D gradient
  Step 9  — Hybrid loss:           λ_photo*L_photo + λ_depth*L_depth + λ_reg*L_reg
  Step 10 — Async optimizer:       runs in backend only, never in tracking thread
  Step 11 — Loop-closure re-anchor: propagates per-KF pose corrections to Gaussians
  Step 12 — Periodic cleanup:      prune dead Gaussians, compact map, reset grid
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.gaussian_splat import (
    SH_C0,
    _apply_mask,
    _clone,
    _normal_to_quat_wxyz,
    _prune_mask,
    _save_splat,
    _sim3_to_w2c,
    _ssim,
)


# ── Step 2: Mapping Frame Selector ─────────────────────────────────────────────

class MappingFrameSelector:
    """
    Decides which non-keyframe frames enter the mapping pipeline.

    A frame becomes a mapping_frame when ALL of:
      - cooldown elapsed since last mapping frame
      - average confidence > conf_thresh
      - translation baseline from last KF > baseline_thresh
    """

    def __init__(self, cfg: dict):
        self.baseline_thresh = cfg.get("map_baseline_thresh", 0.02)
        self.conf_thresh     = cfg.get("map_conf_thresh",     0.20)
        self.cooldown        = cfg.get("map_cooldown_frames", 3)
        self._since_map      = 0

    def should_map(self, frame, last_kf_t) -> bool:
        """Return True if frame should enter the mapping pipeline."""
        self._since_map += 1
        if self._since_map < self.cooldown:
            return False
        if frame.C is None or frame.X_canon is None:
            return False
        avg_conf = float(frame.get_average_conf().mean())
        if avg_conf < self.conf_thresh:
            return False
        if last_kf_t is not None:
            t = frame.T_WC.matrix()[0, :3, 3].detach().cpu()
            if float((t - last_kf_t).norm()) < self.baseline_thresh:
                return False
        self._since_map = 0
        return True

    def reset(self):
        self._since_map = 0


# ── Step 3: Local Fusion Buffer ────────────────────────────────────────────────

class LocalFusionBuffer:
    """
    Sliding window of mapping frames.

    Each slot stores serialised frame data (world-frame 3-D points + RGB +
    confidence).  When ready(), fuse() merges all valid points across the window
    into a single point cloud for Gaussian candidate generation.

    Fusion strategy: collect all valid world-frame points from every frame in
    the window.  The Association Engine (Step 6) deduplicates across windows.
    """

    def __init__(self, window_size: int = 7, min_frames: int = 2):
        self.window_size = window_size
        self.min_frames  = min_frames
        self.buffer: deque = deque(maxlen=window_size)

    def add(self, frame_data: dict) -> None:
        self.buffer.append(frame_data)

    def __len__(self) -> int:
        return len(self.buffer)

    def ready(self) -> bool:
        return len(self.buffer) >= self.min_frames

    def fuse(self) -> Optional[dict]:
        """
        Merge buffered frames into a multi-view point cloud.

        Expected frame_data keys (all CPU tensors):
          means_world  (N, 3)  world-frame 3-D points
          rgb          (N, 3)  in [0, 1]
          conf_norm    (N,)    normalised confidence in [0, 1]
          depth_cam    (N,)    camera-frame z depth
          viewmat      (4, 4)  world-to-cam 4×4
          gt_image     (H, W, 3)
          K            (3, 3) or None
          H_img, W_img int

        Returns None when no valid points remain after filtering.
        """
        if not self.ready():
            return None

        all_means, all_rgb, all_conf, all_depth = [], [], [], []
        all_viewmats, all_gt_images, all_Ks     = [], [], []
        H_img = W_img = None

        for fd in self.buffer:
            valid = fd["conf_norm"] > 0.05
            if not valid.any():
                continue
            all_means.append(fd["means_world"][valid])
            all_rgb.append(fd["rgb"][valid])
            all_conf.append(fd["conf_norm"][valid])
            all_depth.append(fd["depth_cam"][valid])
            all_viewmats.append(fd["viewmat"])
            all_gt_images.append(fd["gt_image"])
            if fd.get("K") is not None:
                all_Ks.append(fd["K"])
            if H_img is None:
                H_img = fd["H_img"]
                W_img = fd["W_img"]

        if not all_means:
            return None

        return {
            "means":     torch.cat(all_means,  dim=0),
            "rgb":       torch.cat(all_rgb,    dim=0),
            "conf":      torch.cat(all_conf,   dim=0),
            "depth":     torch.cat(all_depth,  dim=0),
            "viewmats":  torch.stack(all_viewmats),
            "gt_images": all_gt_images,
            "Ks":        torch.stack(all_Ks) if all_Ks else None,
            "H_img":     H_img,
            "W_img":     W_img,
            "n_frames":  len(self.buffer),
        }

    def clear(self) -> None:
        self.buffer.clear()


# ── Step 6: Voxel-Based Association Engine ─────────────────────────────────────

class VoxelAssociation:
    """
    Prevents object duplication by checking spatial proximity before spawn.

    Uses chunked brute-force cdist on GPU for speed.  Falls back to the
    voxel-hash path only when the existing map is very large (>50k).
    """

    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size
        self._means: Optional[torch.Tensor] = None

    def rebuild(self, means: torch.Tensor) -> None:
        """Store current Gaussian means (detached, on original device)."""
        self._means = means.detach()

    def query_batch(
        self,
        new_means: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each candidate, return (nearest_idx, distance).

        Uses chunked GPU cdist for fast vectorised nearest-neighbour lookup.

        Returns:
            match_idx (N,) int64  — index into existing Gaussians, or -1
            dist      (N,) float  — 3-D distance to nearest, or inf
        """
        N = new_means.shape[0]
        match_idx = torch.full((N,), -1,           dtype=torch.long)
        dist      = torch.full((N,), float("inf"), dtype=torch.float32)

        if self._means is None or self._means.shape[0] == 0:
            return match_idx, dist

        device = self._means.device
        existing = self._means.float()
        queries  = new_means.to(device).float()

        # Chunked cdist to avoid OOM on very large maps
        chunk = 5_000
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            d = torch.cdist(queries[start:end], existing)   # (chunk, M)
            min_d, min_i = d.min(dim=1)
            match_idx[start:end] = min_i.cpu()
            dist[start:end]      = min_d.cpu()

        return match_idx, dist

    def is_empty(self) -> bool:
        return self._means is None or self._means.shape[0] == 0


# ── Step 4: Gaussian Candidate Generation ─────────────────────────────────────

def _fused_to_gaussian_candidates(fused: dict) -> Optional[dict]:
    """
    Convert LocalFusionBuffer output → Gaussian candidates.

    Uses KNN-based anisotropic scale and surface-normal quaternions for
    better initial geometry (disc-shaped Gaussians aligned to surfaces
    instead of isotropic spheres).

    All candidates on the same device as fused["means"].
    Returns None when there are no valid points.
    """
    means = fused["means"]
    rgb   = fused["rgb"]
    conf  = fused["conf"]
    depth = fused["depth"]
    N     = means.shape[0]
    if N == 0:
        return None

    device = means.device

    # Opacity (logit) from confidence
    conf_c        = conf.clamp(1e-4, 1 - 1e-4)
    opacity_logit = torch.log(conf_c / (1.0 - conf_c))

    # SH DC from RGB
    sh_dc = (rgb.to(device) - 0.5) / SH_C0

    # Anisotropic scale from KNN distances + surface normal quaternions.
    # For each point, find K nearest neighbours and use the distances to
    # set sx, sy (tangent-plane extents) and sz (thin disc along normal).
    K_nn = min(4, N - 1)
    if K_nn >= 2:
        log_scales, quats = _knn_scales_and_normals(means, K_nn, device)
    else:
        median_depth = float(depth.clamp(min=1e-3).median())
        init_scale = max(median_depth * 0.005, 1e-4)
        log_scales = torch.full((N, 3), math.log(init_scale), device=device)
        quats = torch.zeros(N, 4, device=device)
        quats[:, 0] = 1.0

    return {
        "means":      means.to(device),
        "sh_dc":      sh_dc.to(device),
        "log_scales": log_scales,
        "quats":      quats,
        "opacity":    opacity_logit.to(device),
        "conf":       conf.to(device),
    }


def _knn_scales_and_normals(
    means: torch.Tensor, K: int, device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute anisotropic scales and surface-normal quaternions from KNN.

    For each point, the K nearest neighbours define a local covariance.
    Eigendecomposition gives the tangent plane (2 large eigenvalues) and
    normal direction (smallest eigenvalue).

    Returns:
        log_scales (N, 3): log of per-axis scales
        quats (N, 4): wxyz quaternions aligning Gaussian to local surface
    """
    N = means.shape[0]

    # Chunked KNN via brute-force cdist to avoid OOM on large point clouds
    chunk = 10_000
    all_dists = []
    all_indices = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        dists = torch.cdist(means[start:end], means)   # (chunk, N)
        # Exclude self (distance 0) by setting to inf
        dists[:, start:end].fill_diagonal_(float("inf"))
        topk = dists.topk(K, dim=1, largest=False)
        all_dists.append(topk.values)
        all_indices.append(topk.indices)

    knn_dists = torch.cat(all_dists, dim=0)     # (N, K)
    knn_idx   = torch.cat(all_indices, dim=0)    # (N, K)

    # Tangent-plane scale: mean of K neighbour distances
    mean_dist = knn_dists.mean(dim=1).clamp(min=1e-6)   # (N,)
    # Thin disc: sz = mean_dist * 0.1 (10% of tangent extent)
    sx = mean_dist
    sy = mean_dist
    sz = mean_dist * 0.1
    log_scales = torch.log(
        torch.stack([sx, sy, sz], dim=-1).clamp(min=1e-7)
    ).to(device)

    # Surface normal from local covariance of KNN offsets
    neighbours = means[knn_idx]                  # (N, K, 3)
    offsets = neighbours - means.unsqueeze(1)    # (N, K, 3)
    # Covariance: (N, 3, 3) = offsets^T @ offsets / K
    cov = torch.bmm(offsets.transpose(1, 2), offsets) / K

    # Eigen decomposition — smallest eigenvector is the normal
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # ascending order
        normals = eigenvectors[:, :, 0]  # smallest eigenvalue → normal
        normals = F.normalize(normals, dim=-1)
    except Exception:
        normals = torch.zeros(N, 3, device=device)
        normals[:, 2] = 1.0

    quats = _normal_to_quat_wxyz(normals.to(device))
    return log_scales, quats


# ── Step 7: Merge helpers ──────────────────────────────────────────────────────

def _merge_candidates(
    data: dict,
    optimizer,
    candidates: dict,
    match_idx: torch.Tensor,
    dist: torch.Tensor,
    assoc_thresh: float,
    ema_alpha: float,
    obs_counts: torch.Tensor,
) -> Tuple[int, int, torch.Tensor]:
    """
    Apply merge-or-spawn logic for one batch of candidates.

    Merge  (dist < assoc_thresh): confidence-weighted mean update, EMA SH.
    Spawn  (no close neighbour):  append new Gaussian parameters.

    Returns (n_merged, n_spawned, updated_obs_counts).
    """
    merge_mask = (match_idx >= 0) & (dist.to(match_idx.device) < assoc_thresh)
    spawn_mask = ~merge_mask

    n_merged  = int(merge_mask.sum().item())
    n_spawned = int(spawn_mask.sum().item())

    device = data["means"].device

    # ── Merge ──
    if n_merged > 0:
        m_idx  = match_idx[merge_mask].to(device)
        new_m  = candidates["means"][merge_mask].to(device)
        new_sh = candidates["sh_dc"][merge_mask].to(device)
        new_c  = candidates["conf"][merge_mask].to(device).unsqueeze(-1)
        old_c  = obs_counts[m_idx].float().unsqueeze(-1).clamp(min=1.0)
        total  = old_c + new_c

        with torch.no_grad():
            # Confidence-weighted mean
            data["means"].data[m_idx] = (
                (old_c * data["means"].data[m_idx] + new_c * new_m) / total
            )
            # EMA SH
            data["sh_dc"].data[m_idx] = (
                (1.0 - ema_alpha) * data["sh_dc"].data[m_idx]
                + ema_alpha * new_sh
            )
            obs_counts[m_idx] += 1

    # ── Spawn ──
    if n_spawned > 0:
        new_cands = {
            k: candidates[k][spawn_mask].to(device)
            for k in ["means", "sh_dc", "log_scales", "quats", "opacity"]
        }
        _append_candidates(data, optimizer, new_cands)
        obs_counts = torch.cat([
            obs_counts,
            torch.ones(n_spawned, dtype=obs_counts.dtype, device=device),
        ])

    return n_merged, n_spawned, obs_counts


def _append_candidates(data: dict, optimizer, candidates: dict) -> int:
    """Append Gaussian candidates into live data dict; extend Adam state."""
    param_keys = ["means", "sh_dc", "log_scales", "quats", "opacity"]
    n_new = candidates["means"].shape[0]

    for key in param_keys:
        old_p = data[key]
        new_p = torch.cat(
            [old_p.detach(), candidates[key]], dim=0
        ).requires_grad_(True)
        state = optimizer.state.pop(old_p, {})
        new_state = {}
        if "exp_avg" in state:
            pad = torch.zeros(
                n_new, *state["exp_avg"].shape[1:],
                device=state["exp_avg"].device,
            )
            new_state["exp_avg"]    = torch.cat([state["exp_avg"],    pad])
            new_state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], pad])
            new_state["step"]       = state["step"]
        optimizer.state[new_p] = new_state
        for group in optimizer.param_groups:
            if group["params"][0] is old_p:
                group["params"][0] = new_p
                break
        data[key] = new_p

    return n_new


# ── Steps 5 + 8–12: GlobalGaussianMap ─────────────────────────────────────────

class GlobalGaussianMap:
    """
    Incremental 3DGS map that replaces OnlineGaussianSplat.

    Key differences from OnlineGaussianSplat:
    - Gaussian init from *mapping frames* (not keyframes); Steps 2–4
    - Local sliding-window fusion before init; Step 3
    - VoxelAssociation prevents object duplication; Steps 6–7
    - Hybrid loss with regularisation; Step 9
    - Loop-closure re-anchor propagates pose corrections; Step 11
    - Periodic full cleanup; Step 12

    Public interface is a superset of OnlineGaussianSplat so it can be
    used as a drop-in replacement in run_backend().
    """

    def __init__(self, cfg: dict, use_calib: bool, K, device: str):
        self.cfg        = cfg
        self.use_calib  = use_calib
        self.K_global   = K
        self.device     = device

        # ── Gaussian parameters (populated lazily on first mapping frame)
        self.data:   Optional[dict] = None
        self.optimizer              = None
        self.obs_counts: Optional[torch.Tensor] = None
        self.absgrad_accum: Optional[torch.Tensor] = None

        # ── Rendering metadata (populated as KFs arrive)
        self.n_kf      = 0
        self.train_step = 0
        self.H_img: Optional[int] = None
        self.W_img: Optional[int] = None
        self.Ks_render: Optional[torch.Tensor] = None

        # ── Config knobs
        self.conf_thresh    = cfg.get("c_conf_threshold",    0.5)
        self.max_gaussians  = cfg.get("max_gaussians",  2_000_000)
        self.lambda_ssim    = cfg.get("lambda_ssim",         0.2)
        self.lambda_depth   = cfg.get("lambda_depth",        0.1)
        self.lambda_reg     = cfg.get("lambda_reg",          0.01)
        self.depth_min_conf = cfg.get("depth_min_conf",      0.3)
        self.prune_thresh   = cfg.get("prune_opacity_thresh", 0.005)
        self.grad_thresh    = cfg.get("grad_thresh",         2e-4)
        self.densify_online = cfg.get("densify_online",      True)
        self.aux_ratio      = cfg.get("aux_ratio",           0.3)
        self.aux_max_frames = cfg.get("aux_max_frames",      200)
        self.assoc_thresh   = cfg.get("assoc_thresh",        0.05)
        self.merge_ema      = cfg.get("merge_ema",           0.3)
        self.max_assoc_check = cfg.get("assoc_max_check",   5000)
        self.cleanup_every  = cfg.get("cleanup_interval_kf", 10)
        self._kf_since_cleanup = 0

        # ── Step 3: Local Fusion Buffer
        self.fusion_buffer = LocalFusionBuffer(
            window_size=cfg.get("map_window_size", 7),
            min_frames =cfg.get("map_min_frames",  2),
        )

        # ── Step 6: Voxel Association
        self.association  = VoxelAssociation(
            voxel_size=cfg.get("assoc_voxel", 0.05)
        )
        self._assoc_dirty = False

        # ── Aux frames for photometric supervision
        self.aux_frames: list = []

        # Running global confidence max for cross-KF consistent normalization
        self._global_conf_max: float = 0.0
        self.heartbeat_sec = cfg.get("heartbeat_sec", 30)
        self._last_heartbeat_ts = time.time()

        print("[GMap] GlobalGaussianMap initialized.")

    # ── Rendering-view registration ───────────────────────────────────────────

    def add_keyframe_view(self, kf) -> None:
        """
        Register a keyframe as a render target for photometric supervision.

        Step 2: KFs do NOT initialise Gaussians — only mapping frames do.
        """
        device  = self.device
        H_img, W_img = [int(x) for x in kf.img_shape.flatten()[:2].tolist()]

        if self.H_img is None:
            self.H_img, self.W_img = H_img, W_img

        viewmat  = _sim3_to_w2c(kf.T_WC).to(device)
        gt_image = kf.uimg.to(device)

        # Geometry supervision maps
        X = kf.X_canon
        if self.use_calib and kf.K is not None:
            X = constrain_points_to_ray(
                kf.img_shape.flatten()[:2], X[None], kf.K
            ).squeeze(0)

        # GT depth must be in the same frame as rendered depth.  Rendered depth
        # comes from world-frame Gaussians (= s*R@X_cam + t) projected through
        # scale-free w2c, yielding s*X_cam_z.  Multiply GT by Sim3 scale to match.
        sim3_data = kf.T_WC.data.reshape(-1)
        sim3_scale = float(sim3_data[-1].exp())     # lietorch Sim3 stores log(s)
        depth_map = (X[:, 2] * sim3_scale).reshape(H_img, W_img).clamp(min=1e-3).to(device)
        conf_raw  = kf.get_average_conf().reshape(H_img, W_img).to(device)

        # Global confidence normalization: track running max across all KFs
        # so confidence thresholds are consistent (a blurry frame with low max
        # confidence doesn't get artificially boosted to the same range as a
        # sharp frame).
        local_max = float(conf_raw.max().clamp(min=1e-6))
        self._global_conf_max = max(self._global_conf_max, local_max)
        conf_norm = (conf_raw / self._global_conf_max).clamp(1e-4, 1 - 1e-4)

        if self.data is None:
            # Create data shell; Gaussian params populated on first mapping frame
            self.data = {
                "means":      torch.zeros(0, 3, device=device),
                "sh_dc":      torch.zeros(0, 3, device=device),
                "log_scales": torch.zeros(0, 3, device=device),
                "quats":      torch.zeros(0, 4, device=device),
                "opacity":    torch.zeros(0,    device=device),
                "viewmats":   viewmat.unsqueeze(0),
                "gt_images":  [gt_image],
                "depth_maps": [depth_map],
                "conf_maps":  [conf_norm],
                "Ks":         None,
            }
            self.obs_counts    = torch.zeros(0, dtype=torch.int32, device=device)
            self.absgrad_accum = torch.zeros(0, device=device)
        else:
            self.data["viewmats"] = torch.cat(
                [self.data["viewmats"], viewmat.unsqueeze(0)], dim=0
            )
            self.data["gt_images"].append(gt_image)
            self.data["depth_maps"].append(depth_map)
            self.data["conf_maps"].append(conf_norm)

        # Append camera intrinsics for renderer
        if self.use_calib and kf.K is not None:
            K_kf = kf.K.to(device).unsqueeze(0)
        else:
            fx = fy = float(max(H_img, W_img))
            cx, cy = W_img / 2.0, H_img / 2.0
            K_kf = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
        self.Ks_render = (
            K_kf if self.Ks_render is None
            else torch.cat([self.Ks_render, K_kf], dim=0)
        )

        self.n_kf += 1
        self._kf_since_cleanup += 1

        # Step 12: periodic cleanup
        if self._kf_since_cleanup >= self.cleanup_every:
            self.cleanup()
            self._kf_since_cleanup = 0

    # Alias kept for drop-in compatibility with OnlineGaussianSplat callers
    def add_keyframe(self, kf, threshold=None) -> bool:
        self.add_keyframe_view(kf)
        return True

    # ── Mapping frame intake ──────────────────────────────────────────────────

    def add_mapping_frame(self, frame_data: dict, keyframes=None) -> None:
        """
        Add one mapping frame to the local fusion buffer.

        When the buffer is ready, fuse → generate candidates → associate →
        merge or spawn.  Steps 3–7.

        frame_data keys (CPU tensors, serialised by main thread):
          uimg       (H, W, 3) float32 [0,1]
          sim3_data  (1, 8) raw lietorch.Sim3 data
          X_canon    (H*W, 3) float32 camera-frame 3-D points
          C          (H*W, 1) float32 confidence
          H_img, W_img  int
          K          (3, 3) float32 or None

        keyframes: SharedKeyframes reference — when provided, the nearest KF's
            GN-optimised pose is used instead of the stale enqueued pose.
        """
        if self.data is None:
            return  # wait until first KF view registers the data shell

        fd = self._unpack_mapping_frame(frame_data, keyframes=keyframes)
        if fd is None:
            return
        self.fusion_buffer.add(fd)

        if not self.fusion_buffer.ready():
            return

        fused = self.fusion_buffer.fuse()
        if fused is None:
            return

        # ── Step 4: Generate Gaussian candidates from fused geometry ──
        # Move to device for candidate generation
        fused_dev = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in fused.items()
        }
        candidates = _fused_to_gaussian_candidates(fused_dev)
        if candidates is None:
            return

        # Cap candidate batch for association performance
        N = candidates["means"].shape[0]
        if N > self.max_assoc_check:
            top_idx   = torch.topk(candidates["conf"], self.max_assoc_check).indices
            candidates = {k: candidates[k][top_idx] for k in candidates}
            N = self.max_assoc_check

        # ── Init or associate ──
        if self.data["means"].shape[0] == 0:
            # First mapping frame: bootstrap Gaussian params
            self._bootstrap_from_candidates(candidates)
            return

        # Lazily init optimizer (may already exist)
        if self.optimizer is None:
            self._init_optimizer()

        # Step 6: rebuild voxel grid when dirty
        if self._assoc_dirty or self.association.is_empty():
            self.association.rebuild(self.data["means"].detach())
            self._assoc_dirty = False

        match_idx, dist = self.association.query_batch(
            candidates["means"].cpu()
        )

        # Step 7: merge or spawn
        n_m, n_s, self.obs_counts = _merge_candidates(
            self.data, self.optimizer, candidates,
            match_idx, dist,
            assoc_thresh=self.assoc_thresh,
            ema_alpha=self.merge_ema,
            obs_counts=self.obs_counts,
        )

        if n_s > 0:
            self.absgrad_accum = torch.cat([
                self.absgrad_accum,
                torch.zeros(n_s, device=self.device),
            ])
            self._assoc_dirty = True

        total = self.data["means"].shape[0]
        print(
            f"[GMap] Mapping frame: merged={n_m} spawned={n_s} | total={total:,}"
        )

    # ── Pose synchronisation ──────────────────────────────────────────────────

    def sync_poses(self, keyframes) -> None:
        """Copy latest GN-optimised poses into viewmats (non-differentiable)."""
        if self.data is None or self.n_kf == 0:
            return
        vms = [_sim3_to_w2c(keyframes[i].T_WC).to(self.device)
               for i in range(self.n_kf)]
        self.data["viewmats"] = torch.stack(vms)

    def capture_poses(self, keyframes) -> List[torch.Tensor]:
        """Snapshot current KF world-to-cam matrices (call *before* GN solve)."""
        return [
            _sim3_to_w2c(keyframes[i].T_WC).detach().clone().to(self.device)
            for i in range(len(keyframes))
        ]

    # ── Step 11: Loop-closure re-anchor ───────────────────────────────────────

    def reanchor(
        self,
        keyframes,
        old_poses: List[torch.Tensor],
    ) -> None:
        """
        Propagate loop-closure pose corrections to all Gaussians.

        For each Gaussian, the nearest keyframe camera-centre is found using
        the *old* poses.  The corrected position is:

            p' = R_new^T R_old p + R_new^T (t_old − t_new)

        where R, t come from the corresponding world-to-cam matrices.
        """
        if self.data is None or self.data["means"].shape[0] == 0:
            return

        n_kf = min(len(old_poses), len(keyframes))
        if n_kf == 0:
            return

        new_poses = [
            _sim3_to_w2c(keyframes[i].T_WC).to(self.device)
            for i in range(n_kf)
        ]

        # Camera centres in world frame under OLD poses: c = -R^T t
        old_centers = torch.stack([
            -(p[:3, :3].T @ p[:3, 3]) for p in old_poses[:n_kf]
        ]).to(self.device)                               # (K, 3)

        means   = self.data["means"].detach()            # (N, 3)
        N       = means.shape[0]
        chunk   = 50_000
        corrected = torch.empty_like(means)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            pts = means[start:end]                       # (B, 3)

            # Nearest old camera-centre for each point
            diffs  = pts.unsqueeze(1) - old_centers.unsqueeze(0)  # (B, K, 3)
            nn_idx = diffs.norm(dim=-1).argmin(dim=-1)            # (B,)

            corrected_chunk = torch.empty_like(pts)
            for ki in range(n_kf):
                mask = nn_idx == ki
                if not mask.any():
                    continue
                T_old  = old_poses[ki]
                T_new  = new_poses[ki]
                R_old, t_old = T_old[:3, :3], T_old[:3, 3]
                R_new, t_new = T_new[:3, :3], T_new[:3, 3]
                R_delta = R_new.T @ R_old
                t_delta = R_new.T @ (t_old - t_new)
                corrected_chunk[mask] = (
                    pts[mask] @ R_delta.T + t_delta.unsqueeze(0)
                )
            corrected[start:end] = corrected_chunk

        with torch.no_grad():
            self.data["means"].data.copy_(corrected)

        self._assoc_dirty = True
        self.sync_poses(keyframes)
        print(
            f"[GMap] Re-anchored {N:,} Gaussians across {n_kf} keyframes."
        )

    # ── Step 9: Training ──────────────────────────────────────────────────────

    def train_gaussians(self, n_steps: int, threshold: float = None) -> None:
        """
        Async optimisation with hybrid photometric + depth + regularisation loss.

        Runs only in the backend process (Step 10).
        Poses are fixed — call sync_poses() first.

        Loss:
            total = (1−λs)·L1 + λs·D-SSIM          (photometric)
                  + λd · L_depth                    (geometry from MASt3R)
                  + λr · L_reg                      (scale + opacity regulariser)
        """
        if self.data is None or self.n_kf == 0 or n_steps == 0:
            return
        if self.data["means"].shape[0] == 0:
            return
        if self.optimizer is None:
            self._init_optimizer()

        start_step = self.train_step
        start_ts = time.time()
        print(
            f"[GMap] train start | req_steps={n_steps} | step={start_step} | "
            f"N={self.data['means'].shape[0]:,} | KFs={self.n_kf}",
            flush=True,
        )

        torch.set_grad_enabled(True)
        from gsplat import rasterization

        max_g = threshold if threshold is not None else self.max_gaussians

        for _ in range(n_steps):
            now = time.time()
            if now - self._last_heartbeat_ts >= self.heartbeat_sec:
                n_gs = self.data["means"].shape[0]
                print(
                    f"[GMap] heartbeat | train_step={self.train_step} | "
                    f"n_steps_req={n_steps} | N={n_gs:,} | KFs={self.n_kf} | "
                    f"aux={len(self.aux_frames)}"
                )
                self._last_heartbeat_ts = now

            use_aux = (
                self.aux_ratio > 0.0
                and len(self.aux_frames) > 0
                and torch.rand(1).item() < self.aux_ratio
            )
            if use_aux:
                aux_i  = torch.randint(0, len(self.aux_frames), (1,)).item()
                aux    = self.aux_frames[aux_i]
                vm     = aux["viewmat"].unsqueeze(0)
                gt     = aux["gt_image"]
                K_i    = self.Ks_render[0].unsqueeze(0)
                w      = aux["weight"]
                kf_idx = None
            else:
                kf_idx = torch.randint(0, self.n_kf, (1,)).item()
                vm     = self.data["viewmats"][kf_idx].unsqueeze(0)
                gt     = self.data["gt_images"][kf_idx]
                K_i    = self.Ks_render[kf_idx].unsqueeze(0)
                w      = 1.0

            render_colors, _, meta = rasterization(
                means      = self.data["means"],
                quats      = self.data["quats"],
                scales     = torch.exp(self.data["log_scales"]),
                opacities  = torch.sigmoid(self.data["opacity"]),
                colors     = self.data["sh_dc"].unsqueeze(1),
                viewmats   = vm,
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

            # Photometric
            l_l1 = (render - gt).abs().mean()

            if use_aux:
                loss = l_l1 * w
            else:
                l_ssim  = (1.0 - _ssim(render, gt)) / 2.0
                l_photo = (
                    (1.0 - self.lambda_ssim) * l_l1
                    + self.lambda_ssim * l_ssim
                )

                # Geometry: log-depth consistency (MASt3R supervision)
                l_depth = torch.tensor(0.0, device=self.device)
                if self.lambda_depth > 0.0:
                    gt_dep = self.data["depth_maps"][kf_idx]
                    conf_w = self.data["conf_maps"][kf_idx]
                    valid  = (
                        (gt_dep > 1e-3) & (render_dep > 1e-3)
                        & (conf_w > self.depth_min_conf)
                    )
                    if valid.any():
                        l_depth = (
                            conf_w[valid]
                            * (render_dep[valid].log() - gt_dep[valid].log()).abs()
                        ).mean()

                # Regularisation: penalise over-large scales + dead opacity
                l_reg = torch.tensor(0.0, device=self.device)
                if self.lambda_reg > 0.0:
                    l_reg = (
                        0.5 * torch.exp(self.data["log_scales"]).mean()
                        + 0.5 * (1.0 - torch.sigmoid(self.data["opacity"])).mean()
                    )

                loss = (
                    l_photo
                    + self.lambda_depth * l_depth
                    + self.lambda_reg   * l_reg
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Accumulate abs 2-D gradient for Step 8 densification
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

            if self.train_step % 500 == 0:
                with torch.no_grad():
                    mse = (render - gt).pow(2).mean().clamp(min=1e-10)
                    psnr = -10.0 * math.log10(float(mse))
                n_gs = self.data["means"].shape[0]
                print(
                    f"[GMap] step {self.train_step:5d} | loss {float(loss):.4f} | "
                    f"PSNR {psnr:.2f} dB | N={n_gs:,} | KFs={self.n_kf}"
                )

        end_ts = time.time()
        done_steps = self.train_step - start_step
        print(
            f"[GMap] train done  | req_steps={n_steps} | done_steps={done_steps} | "
            f"step={self.train_step} | elapsed={end_ts - start_ts:.1f}s | "
            f"N={self.data['means'].shape[0]:,}",
            flush=True,
        )

    # ── Aux frames ────────────────────────────────────────────────────────────

    def add_aux_frames(self, frames_data: list) -> None:
        """Buffer non-KF / non-mapping frames for L1-only supervision."""
        if self.data is None or self.H_img is None:
            return
        import lietorch as _lt
        for fd in frames_data:
            T_WC    = _lt.Sim3(fd["sim3_data"].to(self.device))
            viewmat = _sim3_to_w2c(T_WC).to(self.device)
            gt      = fd["uimg"].to(self.device)
            self.aux_frames.append({
                "gt_image": gt,
                "viewmat":  viewmat,
                "weight":   fd["weight"],
            })
        if len(self.aux_frames) > self.aux_max_frames:
            self.aux_frames = self.aux_frames[-self.aux_max_frames:]
        if frames_data:
            print(
                f"[GMap] +{len(frames_data)} aux frames | "
                f"buffer {len(self.aux_frames)}/{self.aux_max_frames}"
            )

    # ── Step 12: Periodic cleanup ─────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Prune dead Gaussians; compact tensors; mark association grid dirty.

        Triggered automatically every cfg['cleanup_interval_kf'] keyframes.
        """
        if self.data is None or self.data["means"].shape[0] == 0:
            return
        if self.optimizer is None:
            return

        n_before = self.data["means"].shape[0]

        # Prune: low opacity
        prune = _prune_mask(self.data, threshold=self.prune_thresh)

        # Prune: single observation + near-zero opacity (never confirmed)
        if (
            self.obs_counts is not None
            and self.obs_counts.shape[0] == n_before
        ):
            unconfirmed = self.obs_counts <= 1
            low_op      = torch.sigmoid(self.data["opacity"]).detach() < 0.01
            prune       = prune | (unconfirmed & low_op)

        if prune.any():
            keep = ~prune
            _apply_mask(self.data, self.optimizer, keep)
            if self.absgrad_accum.shape[0] == n_before:
                self.absgrad_accum = self.absgrad_accum[keep]
            if (
                self.obs_counts is not None
                and self.obs_counts.shape[0] == n_before
            ):
                self.obs_counts = self.obs_counts[keep]
            self._assoc_dirty = True

        n_after = self.data["means"].shape[0]
        if n_after != n_before:
            print(
                f"[GMap] Cleanup: pruned {n_before - n_after:,} | "
                f"{n_after:,} Gaussians remain."
            )

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, save_dir, seq_name: str) -> None:
        if self.data is None or self.data["means"].shape[0] == 0:
            print("[GMap] Nothing to save.")
            return
        out_path = Path(save_dir) / f"{seq_name}_gs.ply"
        _save_splat(out_path, self.data)
        print(
            f"[GMap] Saved → {out_path}  "
            f"({self.data['means'].shape[0]:,} Gaussians, "
            f"{self.train_step} train steps, {self.n_kf} KF views)"
        )

    # Stub to keep OnlineGaussianSplat interface parity
    def refine_poses(self, n_steps: int, threshold: float = None) -> None:
        pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _unpack_mapping_frame(
        self, fd: dict, keyframes=None,
    ) -> Optional[dict]:
        """
        Convert serialised mapping-frame dict (from main-thread queue) into
        the format expected by LocalFusionBuffer.

        When keyframes is provided, the nearest KF's GN-optimised pose is used
        to compute relative camera-to-KF transform, then world-frame points are
        obtained via the optimised KF pose.  This avoids stale tracking poses
        placing geometry at wrong world positions.
        """
        import lietorch as _lt

        X_canon = fd["X_canon"].to(self.device)     # (H*W, 3)
        C       = fd["C"].to(self.device)            # (H*W, 1)
        uimg    = fd["uimg"].to(self.device)         # (H, W, 3)
        H_img   = fd["H_img"]
        W_img   = fd["W_img"]
        K       = fd.get("K")
        if K is not None:
            K = K.to(self.device)

        T_WC_enqueued = _lt.Sim3(fd["sim3_data"].to(self.device))

        # Optionally constrain to calibrated rays
        X = X_canon
        if self.use_calib and K is not None:
            img_shape = torch.tensor([[H_img, W_img]], device=self.device)
            X = constrain_points_to_ray(img_shape.flatten()[:2], X[None], K).squeeze(0)

        # Use GN-optimised nearest-KF pose to correct world-frame positions.
        #
        # The mapping frame was tracked against the latest KF, so both share the
        # same tracking epoch.  We compute the relative transform T_kf_mf between
        # the old KF pose and the MF pose, then reapply it with the optimised KF
        # pose:  c2w_mf_corrected = c2w_kf_opt @ T_kf_from_mf
        #
        # T_kf_from_mf takes MF camera-frame points into KF camera frame:
        #   T_kf_from_mf = w2c_kf_enq @ c2w_mf_enq
        if keyframes is not None and self.n_kf > 0 and "kf_sim3_data" in fd:
            nearest_kf_idx = self.n_kf - 1
            T_WC_kf_opt = keyframes[nearest_kf_idx].T_WC

            # Build 4x4 matrices
            w2c_mf_enq = _sim3_to_w2c(T_WC_enqueued).to(self.device)
            c2w_mf_enq = torch.eye(4, device=self.device, dtype=w2c_mf_enq.dtype)
            c2w_mf_enq[:3, :3] = w2c_mf_enq[:3, :3].T
            c2w_mf_enq[:3, 3]  = -(w2c_mf_enq[:3, :3].T @ w2c_mf_enq[:3, 3])

            T_WC_kf_enq = _lt.Sim3(fd["kf_sim3_data"].to(self.device))
            w2c_kf_enq = _sim3_to_w2c(T_WC_kf_enq).to(self.device)

            w2c_kf_opt = _sim3_to_w2c(T_WC_kf_opt).to(self.device)
            c2w_kf_opt = torch.eye(4, device=self.device, dtype=w2c_kf_opt.dtype)
            c2w_kf_opt[:3, :3] = w2c_kf_opt[:3, :3].T
            c2w_kf_opt[:3, 3]  = -(w2c_kf_opt[:3, :3].T @ w2c_kf_opt[:3, 3])

            # Relative: MF-cam → KF-cam (using enqueue-epoch poses)
            T_kf_from_mf = w2c_kf_enq @ c2w_mf_enq

            # Corrected: MF-cam → world (using optimised KF pose)
            c2w_corrected = c2w_kf_opt @ T_kf_from_mf

            R = c2w_corrected[:3, :3]
            t = c2w_corrected[:3, 3]
            means_world = (X @ R.T) + t.unsqueeze(0)

            viewmat = torch.eye(4, device=self.device, dtype=R.dtype)
            viewmat[:3, :3] = R.T
            viewmat[:3, 3]  = -(R.T @ t)
        else:
            means_world = T_WC_enqueued.act(X)
            viewmat = _sim3_to_w2c(T_WC_enqueued).to(self.device)

        # Confidence normalisation — use global running max for consistency
        conf = C.reshape(-1)                         # (H*W,)
        local_max = float(conf.max().clamp(min=1e-6))
        self._global_conf_max = max(self._global_conf_max, local_max)
        conf_norm = (conf / self._global_conf_max).clamp(1e-4, 1 - 1e-4)

        depth_cam = X[:, 2].clamp(min=1e-3)          # (H*W,)
        rgb       = uimg.reshape(-1, 3)               # (H*W, 3)

        valid = conf_norm > 0.05
        if not valid.any():
            return None

        return {
            "means_world": means_world[valid].detach().cpu(),
            "rgb":         rgb[valid].detach().cpu(),
            "conf_norm":   conf_norm[valid].detach().cpu(),
            "depth_cam":   depth_cam[valid].detach().cpu(),
            "viewmat":     viewmat.detach().cpu(),
            "gt_image":    uimg.detach(),
            "K":           K.cpu() if K is not None else None,
            "H_img":       H_img,
            "W_img":       W_img,
        }

    def _bootstrap_from_candidates(self, candidates: dict) -> None:
        """Initialise Gaussian parameters from first candidate batch."""
        device = self.device
        for key in ["means", "sh_dc", "log_scales", "quats", "opacity"]:
            self.data[key] = candidates[key].to(device).requires_grad_(True)
        n_new = candidates["means"].shape[0]
        self.obs_counts    = torch.ones(n_new, dtype=torch.int32, device=device)
        self.absgrad_accum = torch.zeros(n_new, device=device)
        self._init_optimizer()
        self._assoc_dirty = True
        print(f"[GMap] Bootstrapped {n_new:,} Gaussians from first mapping frame.")

    def _init_optimizer(self) -> None:
        params = [
            {"params": [self.data["means"]],      "lr": self.cfg.get("lr_means",   1.6e-4)},
            {"params": [self.data["sh_dc"]],      "lr": self.cfg.get("lr_sh",      2.5e-3)},
            {"params": [self.data["log_scales"]], "lr": self.cfg.get("lr_scales",  5e-3)},
            {"params": [self.data["quats"]],      "lr": self.cfg.get("lr_quats",   1e-3)},
            {"params": [self.data["opacity"]],    "lr": self.cfg.get("lr_opacity", 5e-2)},
        ]
        self.optimizer = torch.optim.Adam(params, eps=1e-15)

    def _density_control(self, max_gaussians: int) -> None:
        """Step 8: prune low-opacity Gaussians; clone high-gradient ones."""
        n = self.data["means"].shape[0]

        # Prune
        prune = _prune_mask(self.data, threshold=self.prune_thresh)
        if prune.any():
            keep = ~prune
            _apply_mask(self.data, self.optimizer, keep)
            if self.absgrad_accum.shape[0] == n:
                self.absgrad_accum = self.absgrad_accum[keep]
            if self.obs_counts is not None and self.obs_counts.shape[0] == n:
                self.obs_counts = self.obs_counts[keep]
            n = self.data["means"].shape[0]
            self._assoc_dirty = True

        # Clone under-reconstructed regions
        if self.absgrad_accum.shape[0] == n and n < max_gaussians:
            clone_mask = (
                (self.absgrad_accum / max(self.train_step, 1)) > self.grad_thresh
            )
            if clone_mask.any():
                _clone(self.data, self.optimizer, clone_mask,
                       max_gaussians, max_gaussians)
                allowed = min(int(clone_mask.sum()), max_gaussians - n)
                if allowed > 0:
                    self.absgrad_accum = torch.cat([
                        self.absgrad_accum,
                        torch.zeros(allowed, device=self.device),
                    ])
                    if (
                        self.obs_counts is not None
                        and self.obs_counts.shape[0] == n
                    ):
                        self.obs_counts = torch.cat([
                            self.obs_counts,
                            torch.ones(allowed, dtype=torch.int32,
                                       device=self.device),
                        ])
                    self._assoc_dirty = True

        self.absgrad_accum.zero_()
