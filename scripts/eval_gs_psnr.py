"""
Evaluate rendering quality (PSNR / SSIM) of a saved online-GS PLY.

Usage:
    python scripts/eval_gs_psnr.py logs/7scenes/chess1

The run directory must contain:
    <seq>.txt               — TUM-format trajectory (from SLAM)
    <seq>_online_gs.ply     — Gaussian Splat map
    keyframes/<seq>/*.png   — keyframe images saved during SLAM
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_traj(traj_path: Path) -> dict:
    """Parse TUM-format trajectory → {timestamp_str: (4,4) T_WC numpy array}."""
    poses = {}
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts = parts[0]
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        # Build 4×4 T_WC
        x, y, z, w = qx, qy, qz, qw
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
            [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        poses[ts] = T
    return poses


def _load_ply(ply_path: Path, device: str = "cuda") -> dict:
    """Load a 3DGS PLY into tensors (standard gsplat attribute naming)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    means      = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=1),
                               dtype=torch.float32, device=device)
    sh_dc      = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
                               dtype=torch.float32, device=device)
    log_scales = torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
                               dtype=torch.float32, device=device)
    quats      = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
                               dtype=torch.float32, device=device)   # wxyz
    opacity    = torch.tensor(v["opacity"], dtype=torch.float32, device=device)  # logit

    # Optional SH rest coefficients
    names = {p.name for p in v.properties}
    rest_fields = sorted([n for n in names if n.startswith("f_rest_")],
                         key=lambda n: int(n.split("_")[-1]))
    if rest_fields:
        n_rest = len(rest_fields) // 3
        rest = np.stack([v[n] for n in rest_fields], axis=1)  # (N, n_rest*3)
        rest = rest.reshape(-1, n_rest, 3)
        sh_rest = torch.tensor(rest, dtype=torch.float32, device=device)
    else:
        sh_rest = None

    return dict(means=means, sh_dc=sh_dc, log_scales=log_scales,
                quats=quats, opacity=opacity, sh_rest=sh_rest)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = (pred - gt).pow(2).mean().item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def _ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Fast single-scale SSIM (no padding, uses 11×11 Gaussian)."""
    C1, C2 = 0.01**2, 0.03**2
    p = pred.permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    kernel = _gaussian_kernel(11, 1.5, p.device, p.dtype)
    def conv(x): return F.conv2d(x, kernel, padding=5, groups=3)
    mu_p, mu_g = conv(p), conv(g)
    mu_p2, mu_g2, mu_pg = mu_p**2, mu_g**2, mu_p * mu_g
    sig_p = conv(p**2) - mu_p2
    sig_g = conv(g**2) - mu_g2
    sig_pg = conv(p * g) - mu_pg
    n = (2*mu_pg + C1) * (2*sig_pg + C2)
    d = (mu_p2 + mu_g2 + C1) * (sig_p + sig_g + C2)
    return (n / d).mean().item()


def _gaussian_kernel(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    k = g[:, None] * g[None, :]
    k = k.unsqueeze(0).unsqueeze(0).expand(3, 1, size, size)
    return k.contiguous()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GS PSNR/SSIM for a run directory")
    parser.add_argument("run_dir", type=Path, help="Log directory, e.g. logs/7scenes/chess1")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-cams", type=int, default=0,
                        help="Evaluate on at most N cameras (0 = all)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"[ERROR] Directory not found: {run_dir}"); sys.exit(1)

    # Find trajectory, PLY, keyframe dir
    txt_files = list(run_dir.glob("*.txt"))
    ply_files = list(run_dir.glob("*_online_gs.ply")) + list(run_dir.glob("*_gs.ply"))
    kf_dirs   = list((run_dir / "keyframes").glob("*")) if (run_dir / "keyframes").exists() else []

    if not txt_files:
        print("[ERROR] No .txt trajectory found in run dir."); sys.exit(1)
    if not ply_files:
        print("[ERROR] No *_online_gs.ply found in run dir."); sys.exit(1)
    if not kf_dirs:
        print("[ERROR] No keyframes/ subdirectory found."); sys.exit(1)

    traj_path = txt_files[0]
    ply_path  = ply_files[0]
    kf_dir    = kf_dirs[0]

    print(f"Trajectory : {traj_path.name}")
    print(f"PLY        : {ply_path.name}")
    print(f"Keyframes  : {kf_dir} ({len(list(kf_dir.glob('*.png')))} images)")

    # Load data
    print("\nLoading PLY ...", end=" ", flush=True)
    gs = _load_ply(ply_path, args.device)
    n_gs = gs["means"].shape[0]
    print(f"{n_gs:,} Gaussians")

    poses = _load_traj(traj_path)
    kf_images = sorted(kf_dir.glob("*.png"), key=lambda p: float(p.stem))

    # Match keyframe images → poses by timestamp
    matched = []
    for img_path in kf_images:
        ts = img_path.stem  # e.g. "38.0"
        pose = poses.get(ts)
        if pose is None:
            pose = poses.get(ts.split(".")[0])
        if pose is None:
            # Try approximate match (float comparison)
            ts_f = float(ts)
            best, best_d = None, 1e9
            for k, v in poses.items():
                try:
                    d = abs(float(k) - ts_f)
                    if d < best_d:
                        best_d, best = d, v
                except ValueError:
                    pass
            if best_d < 0.5:
                pose = best
        if pose is not None:
            matched.append((img_path, pose))

    if not matched:
        print("[ERROR] Could not match any keyframe images to trajectory poses."); sys.exit(1)
    print(f"Matched {len(matched)} keyframe images to poses")

    # Optional camera limit
    if args.max_cams > 0 and len(matched) > args.max_cams:
        step = len(matched) // args.max_cams
        matched = matched[::step][:args.max_cams]
        print(f"Evaluating on {len(matched)} cameras (--max-cams)")

    # Rendering
    from gsplat import rasterization

    # Prepare Gaussian tensors
    means      = gs["means"]
    quats      = F.normalize(gs["quats"], dim=-1)       # wxyz
    scales     = torch.exp(gs["log_scales"])
    opacities  = torch.sigmoid(gs["opacity"])
    if gs["sh_rest"] is not None:
        sh_colors = torch.cat([gs["sh_dc"].unsqueeze(1), gs["sh_rest"]], dim=1)
        sh_degree = int(round(sh_colors.shape[1]**0.5) - 1)
    else:
        sh_colors = gs["sh_dc"].unsqueeze(1)
        sh_degree = 0
    print(f"SH degree  : {sh_degree}")

    psnr_list, ssim_list = [], []

    for i, (img_path, T_WC) in enumerate(matched):
        # Load GT image
        gt_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W  = gt_np.shape[:2]
        gt    = torch.tensor(gt_np, device=args.device)

        # Build K (fallback: pinhole with f=max(H,W))
        f     = float(max(H, W))
        K_arr = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=np.float32)
        K     = torch.tensor(K_arr, device=args.device).unsqueeze(0)

        # Build w2c viewmat
        T_CW   = np.linalg.inv(T_WC)
        viewmat = torch.tensor(T_CW, dtype=torch.float32, device=args.device).unsqueeze(0)

        # Render
        with torch.no_grad():
            out, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=sh_colors, viewmats=viewmat, Ks=K,
                width=W, height=H, sh_degree=sh_degree,
                render_mode="RGB+D", packed=False,
            )
        render = out[0, :, :, :3].clamp(0.0, 1.0)

        p = _psnr(render, gt)
        s = _ssim_simple(render, gt)
        psnr_list.append(p)
        ssim_list.append(s)

        if (i + 1) % max(1, len(matched) // 10) == 0 or i == len(matched) - 1:
            print(f"  [{i+1:3d}/{len(matched)}] PSNR {p:.2f} dB  SSIM {s:.4f}  ({img_path.name})")

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    print(f"\n{'='*50}")
    print(f"  PLY    : {ply_path.name}")
    print(f"  N      : {n_gs:,} Gaussians")
    print(f"  Cams   : {len(matched)}")
    print(f"  PSNR   : {mean_psnr:.3f} dB  (std {np.std(psnr_list):.2f})")
    print(f"  SSIM   : {mean_ssim:.4f}  (std {np.std(ssim_list):.4f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
