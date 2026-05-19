from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csmt.utils.smpl_features import (
    load_smpl_motion,
    parse_smpl_arrays,
    resample_smpl_tracks,
)


def _extract_motion_arrays(motion_data):
    """Return (joint_pos, base_trans, base_rot_xyzw, fps)."""
    if isinstance(motion_data, dict):
        fps = float(motion_data.get("fps", 30.0))
        joint_pos = np.asarray(
            motion_data.get("dof_pos", motion_data.get("joint_pos", motion_data.get("joint_positions", []))),
            dtype=np.float32,
        )
        base_trans = np.asarray(
            motion_data.get("root_pos", motion_data.get("base_trans", motion_data.get("base_translation", []))),
            dtype=np.float32,
        )
        base_rot = np.asarray(
            motion_data.get("root_rot", motion_data.get("base_quat", motion_data.get("base_rotation", []))),
            dtype=np.float32,
        )
        if len(joint_pos) == 0 or len(base_trans) == 0 or len(base_rot) == 0:
            raise ValueError("Input PKL dict missing required motion keys")
        return joint_pos, base_trans, base_rot, fps

    if isinstance(motion_data, list):
        if len(motion_data) == 0:
            raise ValueError("Input PKL list is empty")
        base_trans = np.asarray([item[0] for item in motion_data], dtype=np.float32)
        base_rot = np.asarray([item[1] for item in motion_data], dtype=np.float32)
        joint_pos = np.asarray([item[2] for item in motion_data], dtype=np.float32)
        return joint_pos, base_trans, base_rot, 30.0

    raise ValueError(f"Unsupported input PKL type: {type(motion_data)}")


def _compute_speed(pos: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / max(float(fps), 1e-8)
    vel = np.zeros_like(pos, dtype=np.float32)
    if len(pos) > 2:
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    if len(pos) > 1:
        vel[0] = (pos[1] - pos[0]) / dt
        vel[-1] = (pos[-1] - pos[-2]) / dt
    return np.linalg.norm(vel, axis=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple SMPL vs PKL temporal/trajectory visualizer.")
    p.add_argument("--smpl", type=str, required=True, help="SMPL npz/pkl path.")
    p.add_argument("--pkl", type=str, default=None, help="Paired source PKL path.")
    p.add_argument("--max-frames", type=int, default=0, help="0 means all.")
    p.add_argument("--resample-to-pkl-fps", action="store_true", default=True)
    p.add_argument("--no-resample-to-pkl-fps", dest="resample_to_pkl_fps", action="store_false")
    p.add_argument("--save-prefix", type=str, default=None, help="PNG output prefix.")
    p.add_argument("--show", action="store_true", help="Display figures interactively.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    smpl_path = Path(args.smpl).expanduser().resolve()
    smpl_payload = load_smpl_motion(smpl_path)
    pose_body, root_orient, trans, smpl_fps = parse_smpl_arrays(smpl_payload)

    pkl_pos = None
    pkl_fps = None
    if args.pkl is not None:
        pkl_path = Path(args.pkl).expanduser().resolve()
        with pkl_path.open("rb") as f:
            motion = pickle.load(f)
        _, pkl_pos, _, pkl_fps = _extract_motion_arrays(motion)
    else:
        pkl_path = None

    trans_rs = trans
    rs_fps = smpl_fps
    if pkl_pos is not None and args.resample_to_pkl_fps:
        _, _, trans_rs = resample_smpl_tracks(
            pose_body=pose_body,
            root_orient=root_orient,
            trans=trans,
            src_fps=smpl_fps,
            dst_fps=float(pkl_fps),
        )
        rs_fps = float(pkl_fps)

    if args.max_frames > 0:
        n = int(args.max_frames)
        trans = trans[:n]
        trans_rs = trans_rs[:n]
        if pkl_pos is not None:
            pkl_pos = pkl_pos[:n]

    print("SMPL/PKL Summary")
    print(f"  smpl file: {smpl_path}")
    print(f"  smpl fps:  {smpl_fps:.6f}")
    print(f"  smpl frames (orig): {len(trans)}")
    print(f"  smpl frames (resampled): {len(trans_rs)} @ {rs_fps:.6f} fps")
    if pkl_pos is not None:
        print(f"  pkl file:  {pkl_path}")
        print(f"  pkl fps:   {pkl_fps:.6f}")
        print(f"  pkl frames:{len(pkl_pos)}")

    t_smpl = np.arange(len(trans)) / max(float(smpl_fps), 1e-8)
    t_rs = np.arange(len(trans_rs)) / max(float(rs_fps), 1e-8)
    if pkl_pos is not None:
        t_pkl = np.arange(len(pkl_pos)) / max(float(pkl_fps), 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.reshape(-1)

    # XY trajectory
    axes[0].plot(trans[:, 0], trans[:, 1], label="smpl_orig_xy", alpha=0.7)
    axes[0].plot(trans_rs[:, 0], trans_rs[:, 1], label="smpl_resampled_xy", alpha=0.9)
    if pkl_pos is not None:
        axes[0].plot(pkl_pos[:, 0], pkl_pos[:, 1], label="pkl_xy", alpha=0.7)
    axes[0].set_title("Root XY Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Z vs time
    axes[1].plot(t_smpl, trans[:, 2], label="smpl_orig_z", alpha=0.7)
    axes[1].plot(t_rs, trans_rs[:, 2], label="smpl_resampled_z", alpha=0.9)
    if pkl_pos is not None:
        axes[1].plot(t_pkl, pkl_pos[:, 2], label="pkl_z", alpha=0.7)
    axes[1].set_title("Root Z vs Time")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("z")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # X/Y vs time
    axes[2].plot(t_smpl, trans[:, 0], label="smpl_orig_x", alpha=0.6)
    axes[2].plot(t_smpl, trans[:, 1], label="smpl_orig_y", alpha=0.6)
    axes[2].plot(t_rs, trans_rs[:, 0], label="smpl_resampled_x", alpha=0.8)
    axes[2].plot(t_rs, trans_rs[:, 1], label="smpl_resampled_y", alpha=0.8)
    if pkl_pos is not None:
        axes[2].plot(t_pkl, pkl_pos[:, 0], label="pkl_x", alpha=0.6)
        axes[2].plot(t_pkl, pkl_pos[:, 1], label="pkl_y", alpha=0.6)
    axes[2].set_title("Root X/Y vs Time")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("position")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Speed vs time
    v_smpl = _compute_speed(trans, smpl_fps)
    v_rs = _compute_speed(trans_rs, rs_fps)
    axes[3].plot(t_smpl, v_smpl, label="smpl_orig_speed", alpha=0.7)
    axes[3].plot(t_rs, v_rs, label="smpl_resampled_speed", alpha=0.9)
    if pkl_pos is not None:
        v_pkl = _compute_speed(pkl_pos, pkl_fps)
        axes[3].plot(t_pkl, v_pkl, label="pkl_speed", alpha=0.7)
    axes[3].set_title("Root Speed vs Time")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("speed (m/s)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()

    if args.save_prefix is None:
        save_prefix = smpl_path.with_suffix("")
        if pkl_path is not None:
            save_prefix = Path(str(save_prefix) + "_vs_pkl")
    else:
        save_prefix = Path(args.save_prefix).expanduser().resolve()
    out_png = Path(str(save_prefix) + "_summary.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
