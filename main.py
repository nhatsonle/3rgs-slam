import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp

# GlobalGaussianMap import is deferred to run_backend to avoid
# importing gsplat in the main process.


def _blur_score(uimg: torch.Tensor) -> float:
    """Laplacian variance of image — lower = more blurry."""
    gray = uimg.float().mean(dim=-1)          # (H, W), CPU
    lap_k = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
    ).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(
        gray.unsqueeze(0).unsqueeze(0), lap_k, padding=1
    )
    return float(lap.var())


def _try_enqueue_aux(frame, age: int, last_kf_t, queue, cfg_gs: dict) -> None:
    """Score a non-KF frame and enqueue it for GS aux supervision if it passes."""
    if age > cfg_gs.get("aux_max_age", 8):
        return
    if _blur_score(frame.uimg) < cfg_gs.get("aux_blur_thresh", 50.0):
        return
    if last_kf_t is not None:
        t = frame.T_WC.matrix()[0, :3, 3].detach().cpu()
        if float((t - last_kf_t).norm()) < cfg_gs.get("aux_novelty_min", 0.05):
            return
    weight = cfg_gs.get("aux_decay", 0.95) ** age
    try:
        queue.put_nowait({
            "uimg":      frame.uimg.cpu().float(),
            "sim3_data": frame.T_WC.data.detach().cpu(),
            "weight":    weight,
        })
    except Exception:
        pass  # queue full — drop frame silently, never block main thread


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def _drain_queue(q):
    """Drain all currently available items from a multiprocessing queue."""
    batch = []
    while True:
        try:
            batch.append(q.get_nowait())
        except Exception:
            break
    return batch


def run_backend(cfg, model, states, keyframes, K,
                gs_save_dir=None, gs_seq_name=None,
                nonkf_queue=None, mapping_queue=None):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    # ── Select GS module ──────────────────────────────────────────────────────
    cfg_gs      = cfg.get("gaussian_splat", {})
    gs_module   = None
    gs_seen_kfs = set()
    gs_enabled  = cfg_gs.get("enabled") and cfg_gs.get("online") and gs_save_dir is not None

    if gs_enabled:
        use_global_map = cfg_gs.get("use_global_map", False)
        if use_global_map:
            from mast3r_slam.gaussian_map import GlobalGaussianMap
            gs_module = GlobalGaussianMap(
                cfg_gs, cfg.get("use_calib", False), K, device
            )
            print("[GMap] GlobalGaussianMap online mode enabled.")
        else:
            from mast3r_slam.gaussian_splat import OnlineGaussianSplat
            gs_module = OnlineGaussianSplat(
                cfg_gs, cfg.get("use_calib", False), K, device
            )
            print("[GS Online] Incremental GS training enabled.")

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue

        if mode == Mode.RELOC:
            frame = states.get_frame()

            # Step 11: capture poses before loop-closure solve
            old_poses = None
            if (gs_module is not None
                    and cfg_gs.get("reanchor_on_loop_closure", True)
                    and hasattr(gs_module, "capture_poses")):
                old_poses = gs_module.capture_poses(keyframes)

            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)

                # Step 11: re-anchor Gaussians after loop-closure pose correction
                if (old_poses is not None
                        and hasattr(gs_module, "reanchor")):
                    gs_module.reanchor(keyframes, old_poses)

            states.dequeue_reloc()
            continue

        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # ── Graph Construction ────────────────────────────────────────────────
        kf_idx = []
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        # ── Online GS ─────────────────────────────────────────────────────────
        if gs_module is not None:
            if idx not in gs_seen_kfs:
                gs_module.add_keyframe(keyframes[idx])
                gs_seen_kfs.add(idx)

            # Step 3/6/7: drain mapping frames BEFORE sync so fusion uses fresh poses
            if mapping_queue is not None and hasattr(gs_module, "add_mapping_frame"):
                for mf in _drain_queue(mapping_queue):
                    gs_module.add_mapping_frame(mf)

            # Sync GN-optimised poses into viewmats after fusion
            gs_module.sync_poses(keyframes)
            gs_module.train_gaussians(cfg_gs.get("online_iters_per_kf", 50))

            if cfg_gs.get("pose_refine", False):
                gs_module.refine_poses(cfg_gs.get("pose_refine_iters", 10))

            # Aux non-KF frames for photometric supervision
            if nonkf_queue is not None:
                aux_batch = _drain_queue(nonkf_queue)
                if aux_batch:
                    gs_module.add_aux_frames(aux_batch)

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

    # ── Finalize ──────────────────────────────────────────────────────────────
    if gs_module is not None:
        # Register any KF views the backend didn't reach before TERMINATED
        with states.lock:
            remaining_tasks = list(states.global_optimizer_tasks)
        for kf_idx in remaining_tasks:
            if kf_idx not in gs_seen_kfs:
                try:
                    gs_module.add_keyframe(keyframes[kf_idx])
                    gs_seen_kfs.add(kf_idx)
                except Exception:
                    pass
        # Also do a final GN solve pass so poses are up to date
        if remaining_tasks:
            print(f"[GMap] Finalize: registered {len(remaining_tasks)} missed KF views "
                  f"(total {gs_module.n_kf} views).")
            try:
                if config["use_calib"]:
                    factor_graph.solve_GN_calib()
                else:
                    factor_graph.solve_GN_rays()
            except Exception:
                pass

        # Flush any mapping frames that arrived after the last GN solve
        if mapping_queue is not None and hasattr(gs_module, "add_mapping_frame"):
            remaining_mf = _drain_queue(mapping_queue)
            for mf in remaining_mf:
                gs_module.add_mapping_frame(mf)
            if remaining_mf:
                print(f"[GMap] Finalize: flushed {len(remaining_mf)} remaining mapping frames.")

        finalize_iters = cfg_gs.get("online_finalize_iters", 0)
        if finalize_iters > 0:
            print(f"[GS/GMap] Final polish: {finalize_iters} additional steps "
                  f"({gs_module.n_kf} views, {gs_module.data['means'].shape[0] if gs_module.data else 0:,} Gaussians)...")
            gs_module.sync_poses(keyframes)
            gs_module.train_gaussians(finalize_iters)
        gs_module.save(gs_save_dir, gs_seq_name)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    # Pass save paths to backend so it can write the online GS PLY at termination
    gs_save_dir = gs_seq_name = None
    cfg_gs_main = config.get("gaussian_splat", {})
    gs_online = cfg_gs_main.get("enabled") and cfg_gs_main.get("online")
    if dataset.save_results and gs_online:
        gs_save_dir = save_dir
        gs_seq_name = seq_name

    # Queue for non-KF aux frames (main → backend); None when aux is disabled
    nonkf_queue = None
    if gs_online and cfg_gs_main.get("aux_ratio", 0.0) > 0.0:
        nonkf_queue = mp.Queue(maxsize=150)

    # Step 2: Mapping frame queue — non-KF frames selected by MappingFrameSelector
    mapping_queue = None
    use_global_map = gs_online and cfg_gs_main.get("use_global_map", False)
    if use_global_map:
        from mast3r_slam.gaussian_map import MappingFrameSelector
        mapping_frame_selector = MappingFrameSelector(cfg_gs_main)
        mapping_queue = mp.Queue(maxsize=300)
        print("[GMap] MappingFrameSelector enabled.")
    else:
        mapping_frame_selector = None

    backend = mp.Process(
        target=run_backend,
        args=(config, model, states, keyframes, K,
              gs_save_dir, gs_seq_name, nonkf_queue, mapping_queue),
    )
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []
    realtime_poses = []
    frames_since_kf = 0   # frames elapsed since the most recent keyframe
    last_kf_t = None      # (3,) CPU tensor — translation of last KF for novelty gate

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            realtime_poses.append(frame.T_WC.data.detach().cpu().clone())
            i += 1
            continue

        add_new_kf = False
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

            # Score and enqueue non-KF frames for aux GS supervision
            if nonkf_queue is not None and not add_new_kf:
                _try_enqueue_aux(
                    frame, frames_since_kf, last_kf_t,
                    nonkf_queue, cfg_gs_main,
                )

            # Step 2: Enqueue mapping frames for GlobalGaussianMap
            if (mapping_queue is not None
                    and mapping_frame_selector is not None
                    and not add_new_kf
                    and frame.X_canon is not None):
                if mapping_frame_selector.should_map(frame, last_kf_t):
                    H_img = int(frame.img_shape.flatten()[0].item())
                    W_img = int(frame.img_shape.flatten()[1].item())
                    mf_data = {
                        "uimg":      frame.uimg.cpu().float(),
                        "sim3_data": frame.T_WC.data.detach().cpu(),
                        "X_canon":   frame.X_canon.detach().cpu(),
                        "C":         frame.C.detach().cpu(),
                        "H_img":     H_img,
                        "W_img":     W_img,
                        "K": (frame.K.cpu() if frame.K is not None
                              else (K.cpu() if K is not None else None)),
                    }
                    try:
                        mapping_queue.put_nowait(mf_data)
                    except Exception:
                        pass  # queue full — drop silently, never block main

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        realtime_poses.append(frame.T_WC.data.detach().cpu().clone())

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # Update novelty gate for next aux frame scoring
            frames_since_kf = 0
            last_kf_t = frame.T_WC.matrix()[0, :3, 3].detach().cpu()
            # Reset mapping frame selector cooldown on new KF
            if mapping_frame_selector is not None:
                mapping_frame_selector.reset()
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        else:
            frames_since_kf += 1
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
            realtime_poses=realtime_poses,
            current_pose=realtime_poses[-1] if realtime_poses else None,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
        cfg_gs = config.get("gaussian_splat", {})
        if cfg_gs.get("enabled", False) and not cfg_gs.get("online", False):
            from mast3r_slam.gaussian_splat import train_gaussian_splat
            train_gaussian_splat(
                keyframes, K, save_dir, seq_name, config["use_calib"], cfg_gs
            )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done — waiting for backend GS finalize...")
    backend.join()
    if dataset.save_results and gs_online:
        gs_flag = "use_global_map" if cfg_gs_main.get("use_global_map") else "online"
        print(f"[GS] Backend finished. GS output → {gs_save_dir}/{gs_seq_name}_gs.ply  (mode: {gs_flag})")
    if not args.no_viz:
        viz.join()
