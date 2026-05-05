"""
View-tied Gaussian mapping back-end.

This module implements a conservative v1 of VTGaussian-style mapping for the
existing MASt3R-SLAM pipeline:
  - tracking/backend poses remain owned by MASt3R-SLAM;
  - Gaussians are tied to MASt3R keyframe pointmap pixels;
  - only color, isotropic radius, and opacity are optimized;
  - older sections are frozen, while the current section remains learnable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import lietorch
import torch
import torch.nn.functional as F

from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.gaussian_splat import (
    SH_C0,
    _export_ply,
    _psnr,
    _sim3_to_w2c,
    _ssim,
)
from mast3r_slam.online_gaussian_splat import make_keyframe_snapshot


@dataclass
class VTSection:
    """A local group of view-tied Gaussians and their target cameras."""

    section_id: int
    kf_indices: List[int] = field(default_factory=list)
    viewmats: List[torch.Tensor] = field(default_factory=list)       # CPU, w2c
    gt_images: List[torch.Tensor] = field(default_factory=list)      # CPU, H,W,3
    depth_maps: List[torch.Tensor] = field(default_factory=list)     # CPU, H,W
    conf_maps: List[torch.Tensor] = field(default_factory=list)      # CPU, H,W
    Ks_list: List[Optional[torch.Tensor]] = field(default_factory=list)

    anchors: Optional[torch.Tensor] = None                           # device, N,3 camera points
    source_kfs: Optional[torch.Tensor] = None                        # device, N long
    sh_dc: Optional[torch.Tensor] = None                             # device, N,3 learnable
    log_radius: Optional[torch.Tensor] = None                        # device, N,1 learnable
    opacities: Optional[torch.Tensor] = None                         # device, N learnable logits

    def n_gaussians(self) -> int:
        return 0 if self.anchors is None else int(self.anchors.shape[0])


class VTGaussianMapper:
    """Incremental view-tied Gaussian mapper using MASt3R keyframe pointmaps."""

    def __init__(self, cfg: dict, use_calib: bool, device: str = "cuda:0"):
        self.cfg = cfg
        self.use_calib = use_calib
        self.device = device

        self.sections: List[VTSection] = []
        self.kf_pose_data: Dict[int, torch.Tensor] = {}
        self.img_shape: Optional[tuple[int, int]] = None
        self.n_kf = 0
        self.total_steps = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------ state

    def _reset(self) -> None:
        self.sections.clear()
        self.kf_pose_data.clear()
        self.img_shape = None
        self.n_kf = 0
        self.total_steps = 0
        self.optimizer = None

    @property
    def current_section(self) -> Optional[VTSection]:
        return self.sections[-1] if self.sections else None

    def _start_section(self) -> VTSection:
        sec = VTSection(section_id=len(self.sections))
        self.sections.append(sec)
        self.optimizer = None
        print(f"[VTGS] Started section {sec.section_id}")
        return sec

    def _rebuild_optimizer(self) -> None:
        sec = self.current_section
        if sec is None or sec.n_gaussians() == 0:
            self.optimizer = None
            return
        self.optimizer = torch.optim.Adam(
            [
                {"params": [sec.sh_dc], "lr": self.cfg.get("vt_lr_color", self.cfg.get("lr_sh", 2.5e-3))},
                {"params": [sec.log_radius], "lr": self.cfg.get("vt_lr_radius", self.cfg.get("lr_scales", 5.0e-3))},
                {"params": [sec.opacities], "lr": self.cfg.get("vt_lr_opacity", self.cfg.get("lr_opacity", 5.0e-2))},
            ],
            betas=(0.9, 0.999),
            eps=1e-15,
        )

    # ------------------------------------------------------------- extraction

    def _snapshot_to_maps(self, snap: dict) -> dict:
        device = self.device
        uimg = snap["uimg"].to(device)
        X = snap["X_canon"].to(device)
        C_raw = snap["C"].to(device).reshape(-1)
        N_updates = max(int(snap["N"]), 1)
        conf = (C_raw / N_updates).clamp(min=1e-6)
        H, W = snap["img_shape"]

        if self.use_calib and snap["K"] is None:
            raise ValueError("VTGS calibrated mode requires keyframe intrinsics K.")
        if self.use_calib and snap["K"] is not None:
            K_t = snap["K"].to(device)
            img_shape = torch.tensor([H, W], device=device, dtype=torch.int)
            X = constrain_points_to_ray(img_shape, X[None], K_t).squeeze(0)

        conf_max = conf.max().clamp(min=1e-6)
        conf_norm = (conf / conf_max).clamp(1e-4, 1.0 - 1e-4)

        T_data = snap["T_WC_data"].to(device)
        s_scale = T_data.reshape(-1)[7].clamp(min=1e-6)
        T_WC = lietorch.Sim3(T_data)
        viewmat = _sim3_to_w2c(T_WC)

        depth = (X[:, 2] * s_scale).reshape(H, W).clamp(min=1e-3)
        conf_map = conf_norm.reshape(H, W)

        if self.img_shape is None:
            self.img_shape = (H, W)

        return {
            "uimg": uimg,
            "X": X,
            "conf_norm": conf_norm,
            "viewmat": viewmat,
            "depth": depth,
            "conf_map": conf_map,
            "s_scale": s_scale,
            "K": snap["K"].to(device) if snap["K"] is not None else None,
        }

    def _initial_log_radius(self, X: torch.Tensor, maps: dict, H: int, W: int) -> torch.Tensor:
        K = maps["K"]
        z = X[:, 2:3].clamp(min=1e-3)
        if K is not None:
            f_mean = ((K[0, 0] + K[1, 1]) * 0.5).clamp(min=1e-6)
            radius = z / f_mean
        else:
            X_grid = maps["X"].reshape(H, W, 3)
            dx = F.pad(X_grid[:, 1:] - X_grid[:, :-1], (0, 0, 0, 1))
            dy = F.pad(X_grid[1:] - X_grid[:-1], (0, 0, 0, 0, 0, 1))
            spacing = 0.5 * (
                torch.linalg.norm(dx, dim=-1, keepdim=True)
                + torch.linalg.norm(dy, dim=-1, keepdim=True)
            )
            radius = spacing.reshape(-1, 1).clamp(min=1e-6)
        radius = (radius * maps["s_scale"]).clamp(
            min=self.cfg.get("vt_min_radius", 1.0e-5),
            max=self.cfg.get("vt_max_radius", 0.20),
        )
        return torch.log(radius)

    def _make_gaussians_from_mask(self, snap: dict, maps: dict, mask: torch.Tensor, cap: int) -> dict:
        H, W = snap["img_shape"]
        idx = torch.where(mask.reshape(-1))[0]
        if idx.numel() > cap:
            conf = maps["conf_norm"][idx]
            idx = idx[torch.topk(conf, cap).indices]

        X = maps["X"][idx]
        rgb = maps["uimg"].reshape(-1, 3)[idx]
        conf = maps["conf_norm"][idx]

        return {
            "anchors": X.detach(),
            "source_kfs": torch.full((idx.numel(),), int(snap["kf_idx"]), device=self.device, dtype=torch.long),
            "sh_dc": ((rgb - 0.5) / SH_C0).detach(),
            "log_radius": self._initial_log_radius(X, maps, H, W).detach(),
            "opacities": torch.log(conf / (1.0 - conf)).detach(),
        }

    def _append_to_section(self, sec: VTSection, new_gs: dict) -> None:
        if new_gs["anchors"].numel() == 0:
            return
        if sec.anchors is None:
            sec.anchors = new_gs["anchors"]
            sec.source_kfs = new_gs["source_kfs"]
            sec.sh_dc = new_gs["sh_dc"].requires_grad_(True)
            sec.log_radius = new_gs["log_radius"].requires_grad_(True)
            sec.opacities = new_gs["opacities"].requires_grad_(True)
        else:
            sec.anchors = torch.cat([sec.anchors.detach(), new_gs["anchors"]], dim=0)
            sec.source_kfs = torch.cat([sec.source_kfs.detach(), new_gs["source_kfs"]], dim=0)
            sec.sh_dc = torch.cat([sec.sh_dc.detach(), new_gs["sh_dc"]], dim=0).requires_grad_(True)
            sec.log_radius = torch.cat(
                [sec.log_radius.detach(), new_gs["log_radius"]], dim=0
            ).requires_grad_(True)
            sec.opacities = torch.cat(
                [sec.opacities.detach(), new_gs["opacities"]], dim=0
            ).requires_grad_(True)
        self._cap_current_section()
        self._rebuild_optimizer()

    def _cap_current_section(self) -> None:
        sec = self.current_section
        max_sec = self.cfg.get("vt_max_section_gaussians", 500_000)
        if sec is None or sec.n_gaussians() <= max_sec:
            return
        keep = torch.topk(torch.sigmoid(sec.opacities.detach()), max_sec).indices
        sec.anchors = sec.anchors[keep].detach()
        sec.source_kfs = sec.source_kfs[keep].detach()
        sec.sh_dc = sec.sh_dc[keep].detach().requires_grad_(True)
        sec.log_radius = sec.log_radius[keep].detach().requires_grad_(True)
        sec.opacities = sec.opacities[keep].detach().requires_grad_(True)

    # ------------------------------------------------------------- rendering

    def _get_K(self, sec: VTSection, local_cam_idx: int) -> torch.Tensor:
        H, W = self.img_shape
        K = sec.Ks_list[local_cam_idx]
        if K is not None:
            return K.to(self.device).unsqueeze(0)
        f = float(max(H, W))
        K_fb = torch.tensor(
            [[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        return K_fb.unsqueeze(0)

    def _section_context(self, include_overlap: bool = True) -> List[VTSection]:
        if not self.sections:
            return []
        current = self.sections[-1]
        out = [current]
        if len(self.sections) >= 2:
            out.append(self.sections[-2])
        if include_overlap and len(self.sections) >= 3:
            overlap = self._select_overlap_section(current)
            if overlap is not None and not any(overlap is s for s in out):
                out.append(overlap)
        return out

    def _select_overlap_section(self, target: VTSection) -> Optional[VTSection]:
        candidates = self.sections[:-2]
        if not candidates or not target.viewmats:
            return None
        viewmat = target.viewmats[-1].to(self.device)
        depth = target.depth_maps[-1].to(self.device)
        H, W = depth.shape
        K = self._get_K(target, len(target.viewmats) - 1)[0]
        best_sec, best_score = None, -1.0
        sample_cap = self.cfg.get("vt_overlap_sample_cap", 5000)
        tol = self.cfg.get("vt_overlap_depth_tol", 0.05)
        for sec in candidates:
            if sec.n_gaussians() == 0:
                continue
            params = self._materialize([sec], detach=True)
            means = params["means"]
            if means.shape[0] > sample_cap:
                idx = torch.linspace(0, means.shape[0] - 1, sample_cap, device=self.device).long()
                means = means[idx]
            homog = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
            cam = (viewmat @ homog.T).T[:, :3]
            z = cam[:, 2]
            proj = cam[:, :2] / z.clamp(min=1e-6).unsqueeze(-1)
            u = proj[:, 0] * K[0, 0] + K[0, 2]
            v = proj[:, 1] * K[1, 1] + K[1, 2]
            inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inside.any():
                continue
            ui = u[inside].long().clamp(0, W - 1)
            vi = v[inside].long().clamp(0, H - 1)
            obs = depth[vi, ui]
            rel = (z[inside] - obs).abs() / obs.clamp(min=1e-3)
            score = (rel < tol).float().mean().item()
            if score > best_score:
                best_score = score
                best_sec = sec
        return best_sec

    def _materialize(self, sections: List[VTSection], detach: bool = False) -> dict:
        means_all, sh_all, log_r_all, opa_all = [], [], [], []
        for sec in sections:
            if sec.n_gaussians() == 0:
                continue
            means = torch.empty_like(sec.anchors)
            for kf_idx in torch.unique(sec.source_kfs).tolist():
                mask = sec.source_kfs == int(kf_idx)
                T_data = self.kf_pose_data[int(kf_idx)].to(self.device)
                means[mask] = lietorch.Sim3(T_data).act(sec.anchors[mask])
            sh = sec.sh_dc
            log_r = sec.log_radius
            opa = sec.opacities
            if detach:
                means = means.detach()
                sh = sh.detach()
                log_r = log_r.detach()
                opa = opa.detach()
            means_all.append(means)
            sh_all.append(sh)
            log_r_all.append(log_r)
            opa_all.append(opa)
        if not means_all:
            return {}
        means = torch.cat(means_all, dim=0)
        log_radius = torch.cat(log_r_all, dim=0)
        n = means.shape[0]
        quats = torch.zeros(n, 4, device=self.device, dtype=means.dtype)
        quats[:, 0] = 1.0
        return {
            "means": means,
            "sh_dc": torch.cat(sh_all, dim=0),
            "scales": log_radius.expand(-1, 3),
            "quats": quats,
            "opacities": torch.cat(opa_all, dim=0),
        }

    def _render(self, sections: List[VTSection], viewmat: torch.Tensor, K: torch.Tensor):
        from gsplat import rasterization

        params = self._materialize(sections)
        if not params:
            return None, None, None, None
        H, W = self.img_shape
        out, alphas, meta = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh_dc"].unsqueeze(1),
            viewmats=viewmat.to(self.device).unsqueeze(0),
            Ks=K,
            width=W,
            height=H,
            sh_degree=0,
            render_mode="RGB+D",
            packed=False,
        )
        return out[0, :, :, :3].clamp(0.0, 1.0), out[0, :, :, 3], alphas[0], meta

    def _coverage_mask(self, snap: dict, maps: dict) -> torch.Tensor:
        sec = self.current_section
        H, W = snap["img_shape"]
        valid = self._valid_mask(maps).reshape(H, W)
        if sec is None or sec.n_gaussians() == 0:
            return valid
        render, render_depth, alpha, _ = self._render(
            [sec],
            maps["viewmat"],
            self._K_from_maps(maps),
        )
        if render is None:
            return valid
        alpha = alpha.squeeze(-1) if alpha.dim() == 3 else alpha
        gt_depth = maps["depth"]
        rel = (render_depth - gt_depth).abs() / gt_depth.clamp(min=1e-3)
        uncovered = alpha < self.cfg.get("vt_silhouette_thresh", 0.2)
        depth_mismatch = rel > self.cfg.get("vt_depth_cover_tol", 0.05)
        return valid & (uncovered | depth_mismatch)

    def _K_from_maps(self, maps: dict) -> torch.Tensor:
        K = maps["K"]
        if K is not None:
            return K.unsqueeze(0)
        H, W = self.img_shape
        f = float(max(H, W))
        return torch.tensor(
            [[[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=self.device,
        )

    def _valid_mask(self, maps: dict) -> torch.Tensor:
        return (
            (maps["conf_norm"] > self.cfg.get("c_conf_threshold", 0.3))
            & (maps["X"][:, 2] > self.cfg.get("vt_min_depth", 1e-3))
        )

    # --------------------------------------------------------------- public API

    def insert_keyframe(self, snap: dict) -> None:
        maps = self._snapshot_to_maps(snap)
        kf_idx = int(snap["kf_idx"])
        self.kf_pose_data[kf_idx] = snap["T_WC_data"].to(self.device)

        section_size = max(1, int(self.cfg.get("vt_section_size", 8)))
        is_head = (self.n_kf % section_size) == 0 or self.current_section is None
        sec = self._start_section() if is_head else self.current_section

        sec.kf_indices.append(kf_idx)
        sec.viewmats.append(maps["viewmat"].detach().cpu())
        sec.gt_images.append(maps["uimg"].detach().cpu())
        sec.depth_maps.append(maps["depth"].detach().cpu())
        sec.conf_maps.append(maps["conf_map"].detach().cpu())
        sec.Ks_list.append(maps["K"].detach().cpu() if maps["K"] is not None else None)

        if is_head:
            insert_mask = self._valid_mask(maps)
            cap = self.cfg.get("vt_head_insert_cap", 120_000)
        else:
            insert_mask = self._coverage_mask(snap, maps).reshape(-1)
            cap = self.cfg.get("vt_regular_insert_cap", 60_000)
        new_gs = self._make_gaussians_from_mask(snap, maps, insert_mask, cap)
        self._append_to_section(sec, new_gs)

        self.n_kf += 1
        print(
            f"[VTGS] kf={self.n_kf:3d} section={sec.section_id:3d} "
            f"inserted={new_gs['anchors'].shape[0]:,} section_N={sec.n_gaussians():,}"
        )

    def train_step(self, n_steps: int = 1, use_all_sections: bool = False) -> dict:
        sec = self.current_section
        if sec is None or sec.n_gaussians() == 0 or self.optimizer is None:
            return {"loss": 0.0, "rgb": 0.0, "depth": 0.0, "ssim": 0.0, "n_steps": 0}
        if not sec.viewmats:
            return {"loss": 0.0, "rgb": 0.0, "depth": 0.0, "ssim": 0.0, "n_steps": 0}

        acc = {"loss": 0.0, "rgb": 0.0, "depth": 0.0, "ssim": 0.0, "n_steps": 0}
        context = self.sections if use_all_sections else self._section_context(include_overlap=True)
        lambda_rgb = self.cfg.get("vt_lambda_rgb", 1.0)
        lambda_depth = self.cfg.get("vt_lambda_depth", 0.1)
        lambda_ssim = self.cfg.get("vt_lambda_ssim", 0.2)
        depth_min_conf = self.cfg.get("depth_min_conf", 0.3)
        alpha_min = self.cfg.get("vt_rgb_alpha_min", 1e-4)

        for _ in range(n_steps):
            local_idx = torch.randint(0, len(sec.viewmats), (1,)).item()
            viewmat = sec.viewmats[local_idx].to(self.device)
            gt = sec.gt_images[local_idx].to(self.device)
            gt_depth = sec.depth_maps[local_idx].to(self.device)
            conf = sec.conf_maps[local_idx].to(self.device)
            K = self._get_K(sec, local_idx)

            self.optimizer.zero_grad(set_to_none=True)
            render, render_depth, alpha, _ = self._render(context, viewmat, K)
            if render is None or render.grad_fn is None:
                continue
            alpha = alpha.squeeze(-1) if alpha.dim() == 3 else alpha

            rgb_mask = (alpha > alpha_min) & (conf > depth_min_conf)
            if not rgb_mask.any():
                rgb_mask = conf > depth_min_conf
            l_rgb = (render[rgb_mask] - gt[rgb_mask]).abs().mean()
            l_ssim = (1.0 - _ssim(render, gt)) / 2.0 if lambda_ssim > 0 else torch.tensor(0.0, device=self.device)

            l_depth = torch.tensor(0.0, device=self.device)
            if lambda_depth > 0:
                valid = (gt_depth > 1e-3) & (render_depth > 1e-3) & (conf > depth_min_conf)
                if valid.any():
                    l_depth = (
                        conf[valid]
                        * (render_depth[valid].log() - gt_depth[valid].log()).abs()
                    ).mean()

            loss = lambda_rgb * l_rgb + lambda_ssim * l_ssim + lambda_depth * l_depth
            loss.backward()
            torch.nn.utils.clip_grad_norm_([sec.sh_dc, sec.log_radius, sec.opacities], 1.0)
            self.optimizer.step()

            with torch.no_grad():
                min_log = math.log(self.cfg.get("vt_min_radius", 1e-5))
                max_log = math.log(self.cfg.get("vt_max_radius", 0.20))
                sec.log_radius.data.clamp_(min_log, max_log)
                sec.opacities.data.clamp_(-10.0, 10.0)

            self.total_steps += 1
            acc["loss"] += loss.item()
            acc["rgb"] += l_rgb.item()
            acc["depth"] += l_depth.item()
            acc["ssim"] += l_ssim.item()
            acc["n_steps"] += 1
        return acc

    def sync_poses_from_shared(self, keyframes: SharedKeyframes) -> None:
        for sec in self.sections:
            for i, kf_idx in enumerate(sec.kf_indices):
                try:
                    kf = keyframes[kf_idx]
                except Exception:
                    continue
                self.kf_pose_data[kf_idx] = kf.T_WC.data.to(self.device)
                sec.viewmats[i] = _sim3_to_w2c(kf.T_WC).detach().cpu()

    def rebuild_from_shared_keyframes(self, keyframes: SharedKeyframes) -> None:
        print(f"[VTGS] Rebuilding sections from {len(keyframes)} final keyframes ...")
        self._reset()
        for kf_idx in range(len(keyframes)):
            snap = make_keyframe_snapshot(kf_idx, keyframes[kf_idx], self.use_calib)
            self.insert_keyframe(snap)
        print(f"[VTGS] Rebuild complete: {self.n_total_gaussians():,} Gaussians")

    def run_fine_refinement(self, n_iters: int) -> None:
        if n_iters <= 0 or not self.sections:
            return
        log_interval = self.cfg.get("fine_log_interval", 500)
        print(
            f"[VTGS] Fine refinement: {n_iters} iters | "
            f"{len(self.sections)} sections | {self.n_total_gaussians():,} Gaussians"
        )
        # Fine pass visits each section as active once, keeping old/new neighbours as context.
        original_current = self.current_section
        for it in range(n_iters):
            sec_idx = it % len(self.sections)
            sec = self.sections.pop(sec_idx)
            self.sections.append(sec)
            self._rebuild_optimizer()
            acc = self.train_step(n_steps=1, use_all_sections=False)
            if it % log_interval == 0 and acc["n_steps"] > 0:
                psnr = self._quick_psnr(max_cams=self.cfg.get("fine_psnr_cams", 20))
                print(
                    f"[VTGS] Fine {it:5d}/{n_iters} | "
                    f"loss={acc['loss']:.4f} rgb={acc['rgb']:.4f} "
                    f"dep={acc['depth']:.4f} psnr={psnr:.2f} dB | "
                    f"N={self.n_total_gaussians():,}"
                )
        # Restore chronological order for saving/debugging.
        self.sections.sort(key=lambda s: s.section_id)
        if original_current is not None and self.sections[-1] is not original_current:
            self._rebuild_optimizer()

    def _quick_psnr(self, max_cams: int = 20) -> float:
        if not self.sections:
            return 0.0
        vals = []
        sections = self.sections
        checked = 0
        with torch.no_grad():
            for sec in sections:
                for local_idx in range(len(sec.viewmats)):
                    if checked >= max_cams:
                        break
                    render, _, _, _ = self._render(
                        sections,
                        sec.viewmats[local_idx].to(self.device),
                        self._get_K(sec, local_idx),
                    )
                    if render is not None:
                        vals.append(_psnr(render, sec.gt_images[local_idx].to(self.device)))
                    checked += 1
                if checked >= max_cams:
                    break
        return sum(vals) / max(len(vals), 1)

    def n_total_gaussians(self) -> int:
        return sum(sec.n_gaussians() for sec in self.sections)

    def save_native(self, save_dir: Path, seq_name: str) -> None:
        out = Path(save_dir) / f"{seq_name}_vtgs.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": dict(self.cfg),
            "img_shape": self.img_shape,
            "kf_pose_data": {
                int(k): v.detach().cpu() for k, v in self.kf_pose_data.items()
            },
            "sections": [],
        }
        for sec in self.sections:
            payload["sections"].append(
                {
                    "section_id": sec.section_id,
                    "kf_indices": list(sec.kf_indices),
                    "anchors": None if sec.anchors is None else sec.anchors.detach().cpu(),
                    "source_kfs": None if sec.source_kfs is None else sec.source_kfs.detach().cpu(),
                    "sh_dc": None if sec.sh_dc is None else sec.sh_dc.detach().cpu(),
                    "log_radius": None if sec.log_radius is None else sec.log_radius.detach().cpu(),
                    "opacities": None if sec.opacities is None else sec.opacities.detach().cpu(),
                }
            )
        torch.save(payload, out)
        print(f"[VTGS] Native state written: {out}")

    def save_ply(self, save_dir: Path, seq_name: str) -> None:
        params = self._materialize(self.sections, detach=True)
        if not params:
            print("[VTGS] No Gaussians to export.")
            return
        _export_ply(Path(save_dir) / f"{seq_name}_vtgs_online_gs.ply", params)


def run_online_vtgs(
    cfg: dict,
    states,
    keyframes: SharedKeyframes,
    save_dir: str,
    seq_name: str,
    gs_queue,
) -> None:
    """Worker process entry point for view-tied online mapping."""

    from mast3r_slam.config import set_global_config

    set_global_config(cfg)
    torch.set_grad_enabled(True)

    gs_cfg = cfg.get("gaussian_splat", {})
    mapper = VTGaussianMapper(
        gs_cfg,
        use_calib=cfg.get("use_calib", False),
        device="cuda:0",
    )

    idle_count = 0
    sync_every = gs_cfg.get("vt_sync_every_idle", 10)
    log_interval = gs_cfg.get("coarse_log_interval", 100)
    loss_acc = {"loss": 0.0, "rgb": 0.0, "depth": 0.0, "ssim": 0.0, "n_steps": 0}

    def merge_acc(src: dict) -> None:
        for k in loss_acc:
            loss_acc[k] += src.get(k, 0.0)

    def maybe_log(force: bool = False) -> None:
        n = loss_acc["n_steps"]
        if n == 0:
            return
        if force or mapper.total_steps % log_interval < gs_cfg.get("vt_steps_idle", 20) + 1:
            print(
                f"[VTGS] step={mapper.total_steps:6d} | N={mapper.n_total_gaussians():,} | "
                f"loss={loss_acc['loss']/n:.4f} rgb={loss_acc['rgb']/n:.4f} "
                f"dep={loss_acc['depth']/n:.4f} ssim={loss_acc['ssim']/n:.4f}"
            )
            for k in loss_acc:
                loss_acc[k] = 0.0

    print("[VTGS] Mapping worker started.")

    while True:
        try:
            event = gs_queue.get(timeout=0.05)
        except Exception:
            if mapper.n_kf > 0 and not states.is_paused():
                acc = mapper.train_step(n_steps=gs_cfg.get("vt_steps_idle", 20))
                merge_acc(acc)
                idle_count += 1
                if idle_count % sync_every == 0:
                    mapper.sync_poses_from_shared(keyframes)
                maybe_log()
            continue

        etype = event.get("type")
        if etype == "new_kf":
            mapper.insert_keyframe(event)
            steps = (
                gs_cfg.get("vt_steps_head", 300)
                if (mapper.n_kf - 1) % max(1, int(gs_cfg.get("vt_section_size", 8))) == 0
                else gs_cfg.get("vt_steps_regular", 80)
            )
            acc = mapper.train_step(n_steps=steps)
            merge_acc(acc)
            maybe_log(force=True)

        elif etype == "terminate":
            print("[VTGS] Terminate received — syncing final poses ...")
            if mapper.n_kf > 0:
                mapper.sync_poses_from_shared(keyframes)
                if gs_cfg.get("vt_rebuild_from_final_keyframes", True):
                    mapper.rebuild_from_shared_keyframes(keyframes)
                mapper.run_fine_refinement(gs_cfg.get("vt_n_iters_fine", gs_cfg.get("n_iters_fine", 2000)))
                mapper.save_native(Path(save_dir), seq_name)
                mapper.save_ply(Path(save_dir), seq_name)
            break

        else:
            print(f"[VTGS] Ignoring unknown event type: {etype}")

    print("[VTGS] Worker finished.")
