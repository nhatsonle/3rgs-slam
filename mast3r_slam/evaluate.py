import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def _quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_to_center_rot(pose):
    if hasattr(pose, "data"):
        pose_data = pose.data.detach().cpu().numpy().reshape(-1)
    elif isinstance(pose, torch.Tensor):
        pose_data = pose.detach().cpu().numpy().reshape(-1)
    else:
        pose_data = np.asarray(pose, dtype=np.float32).reshape(-1)
    t = pose_data[:3].astype(np.float32)
    q = pose_data[3:7].astype(np.float32)  # xyzw
    R = _quat_xyzw_to_matrix(q)
    return t, R


def _build_pose_gizmo(center, R, axis_len=0.1, axis_steps=5):
    points = []
    colors = []
    axes = [
        (R[:, 0], np.array([255, 0, 0], dtype=np.uint8)),
        (R[:, 1], np.array([0, 255, 0], dtype=np.uint8)),
        (R[:, 2], np.array([0, 0, 255], dtype=np.uint8)),
    ]
    for axis_vec, axis_color in axes:
        for step in range(1, axis_steps + 1):
            scale = axis_len * (step / axis_steps)
            points.append(center + axis_vec * scale)
            colors.append(axis_color)
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def _sample_polyline(points, steps_per_segment=8):
    if len(points) < 2:
        return np.empty((0, 3), dtype=np.float32)
    sampled = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        for s in range(steps_per_segment + 1):
            a = s / max(1, steps_per_segment)
            sampled.append((1.0 - a) * p0 + a * p1)
    return np.asarray(sampled, dtype=np.float32)


def _build_camera_symbol(center, R, size=0.06, ring_points=20):
    # Camera frame: x=right, y=down, z=forward (camera-to-world rotation columns)
    right = R[:, 0]
    down = R[:, 1]
    forward = R[:, 2]

    # A red ring in the camera image plane
    ring = []
    for i in range(ring_points):
        theta = 2.0 * np.pi * i / max(1, ring_points)
        ring.append(center + size * (np.cos(theta) * right + np.sin(theta) * down))
    ring = np.asarray(ring, dtype=np.float32)
    ring_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (ring.shape[0], 1))

    # Short forward marker (camera look direction), also red
    fwd = np.stack(
        [
            center,
            center + 1.5 * size * forward,
        ],
        axis=0,
    ).astype(np.float32)
    fwd_colors = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (fwd.shape[0], 1))
    return ring, ring_colors, fwd, fwd_colors


def save_reconstruction(
    savedir,
    filename,
    keyframes,
    c_conf_threshold,
    realtime_poses=None,
    current_pose=None,
    traj_stride=5,
    traj_line_steps=8,
    camera_symbol_size=0.06,
    camera_symbol_stride=8,
):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    if realtime_poses:
        sampled_poses = realtime_poses[:: max(1, int(traj_stride))]
        centers = []
        for pose in sampled_poses:
            center, R = _pose_to_center_rot(pose)
            centers.append(center)

        traj_points = []
        traj_colors = []
        if centers:
            centers = np.asarray(centers, dtype=np.float32)
            line_pts = _sample_polyline(centers, steps_per_segment=max(1, int(traj_line_steps)))
            if line_pts.shape[0] > 0:
                traj_points.append(line_pts)
                traj_colors.append(
                    np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (line_pts.shape[0], 1))
                )

            # Add sparse camera symbols to avoid clutter
            symbol_stride = max(1, int(camera_symbol_stride))
            for pose in sampled_poses[::symbol_stride]:
                center, R = _pose_to_center_rot(pose)
                ring_pts, ring_cols, fwd_pts, fwd_cols = _build_camera_symbol(
                    center,
                    R,
                    size=camera_symbol_size,
                )
                traj_points.extend([ring_pts, fwd_pts])
                traj_colors.extend([ring_cols, fwd_cols])

        if traj_points:
            pointclouds = np.concatenate([pointclouds] + traj_points, axis=0)
            colors = np.concatenate([colors] + traj_colors, axis=0)

    if current_pose is not None:
        center, R = _pose_to_center_rot(current_pose)
        current_center = center[None, :]
        current_color = np.array([[255, 0, 0]], dtype=np.uint8)
        ring_pts, ring_cols, fwd_pts, fwd_cols = _build_camera_symbol(
            center,
            R,
            size=1.8 * camera_symbol_size,
        )
        pointclouds = np.concatenate(
            [pointclouds, current_center, ring_pts, fwd_pts],
            axis=0,
        )
        colors = np.concatenate([colors, current_color, ring_cols, fwd_cols], axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
