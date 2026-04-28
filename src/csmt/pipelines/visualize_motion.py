from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from csmt.robots.registry import load_robot_spec


def _resolve_robot_xml(output_root: Path, robot_id: str, xml_override: Optional[str]) -> Path:
    if xml_override:
        return Path(xml_override).expanduser().resolve()
    robot_cfg = output_root / "configs" / "robots" / f"{robot_id}.yaml"
    spec = load_robot_spec(robot_cfg)
    xml_path = spec.source_xml if spec.source_xml.is_absolute() else (output_root / spec.source_xml)
    return xml_path.resolve()


def _extract_motion_arrays(payload: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    if isinstance(payload, dict):
        dof = payload.get("dof_pos", payload.get("joint_pos", payload.get("joint_positions", None)))
        pos = payload.get("root_pos", payload.get("base_trans", payload.get("base_translation", None)))
        rot = payload.get("root_rot", payload.get("base_rot", payload.get("base_rotation", None)))
        if dof is None:
            raise ValueError("PKL dict missing dof_pos/joint_pos")
        if pos is None:
            pos = np.zeros((len(dof), 3), dtype=np.float32)
        if rot is None:
            rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(dof), 1))
        fps = payload.get("fps", None)
        fps = float(fps) if fps is not None else None
        return np.asarray(dof, dtype=np.float32), np.asarray(pos, dtype=np.float32), np.asarray(rot, dtype=np.float32), fps

    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty list PKL")
        pos = np.asarray([f[0] for f in payload], dtype=np.float32)
        rot = np.asarray([f[1] for f in payload], dtype=np.float32)
        dof = np.asarray([f[2] for f in payload], dtype=np.float32)
        return dof, pos, rot, None

    raise ValueError(f"Unsupported PKL payload type: {type(payload)}")


def _load_motion(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return _extract_motion_arrays(payload)


def _to_wxyz(quat: np.ndarray, convention: str) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    if convention == "xyzw":
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return q


def _get_non_free_joint_qpos_addrs(model: mujoco.MjModel) -> list[tuple[int, str]]:
    out = []
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qadr = int(model.jnt_qposadr[j])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        out.append((qadr, name))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic robot motion viewer (qpos-based, joint-index order).")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--robot-id", type=str, required=True, help="Robot ID in configs/robots/<robot-id>.yaml")
    p.add_argument("--pkl", type=str, required=True, help="Motion PKL path")
    p.add_argument("--xml", type=str, default=None, help="Override XML path")
    p.add_argument("--fps", type=float, default=None, help="Playback FPS override")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--quat-convention", choices=["xyzw", "wxyz"], default="xyzw")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    pkl_path = Path(args.pkl).expanduser().resolve()
    xml_path = _resolve_robot_xml(output_root, args.robot_id, args.xml)

    dof_pos, root_pos, root_rot, fps_from_file = _load_motion(pkl_path)
    n_frames = int(dof_pos.shape[0])
    if n_frames == 0:
        raise ValueError("Motion has zero frames")

    play_fps = float(args.fps) if args.fps is not None else float(fps_from_file if fps_from_file else 30.0)
    dt = 1.0 / max(play_fps, 1e-6)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    has_free_base = model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) and model.nq >= 7
    joint_qpos = _get_non_free_joint_qpos_addrs(model)
    n_model_joints = len(joint_qpos)
    n_motion_joints = int(dof_pos.shape[1])
    map_dim = min(n_model_joints, n_motion_joints)

    print(f"XML: {xml_path}")
    print(f"PKL: {pkl_path}")
    print(f"Frames: {n_frames}, playback_fps: {play_fps:.3f}")
    print(f"Model: nq={model.nq} nv={model.nv} njnt={model.njnt}")
    print(f"Free base: {has_free_base}")
    print(f"Joint dims: motion={n_motion_joints}, model_non_free={n_model_joints}, mapped={map_dim}")
    if n_model_joints != n_motion_joints:
        print("[warn] Motion joint dim does not exactly match model non-free joints; using min(motion, model).")

    root_motion_mag = float(np.linalg.norm(root_pos[-1] - root_pos[0])) if n_frames > 1 else 0.0
    if (not has_free_base) and root_motion_mag > 1e-5:
        print("[warn] Motion has root translation, but XML has fixed base (no free joint). Root motion ignored.")

    frame = [0]
    paused = [False]

    def apply_frame(k: int) -> None:
        if has_free_base:
            data.qpos[0:3] = root_pos[k]
            data.qpos[3:7] = _to_wxyz(root_rot[k], args.quat_convention)
        for i in range(map_dim):
            qadr, _ = joint_qpos[i]
            data.qpos[qadr] = dof_pos[k, i]
        mujoco.mj_forward(model, data)

    def key_callback(key: int) -> None:
        if key == 32:
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Playing")
        elif key == ord("r"):
            frame[0] = 0
            print("Reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        while viewer.is_running():
            apply_frame(frame[0])
            if not paused[0]:
                frame[0] += 1
                if frame[0] >= n_frames:
                    if args.loop:
                        frame[0] = 0
                    else:
                        frame[0] = n_frames - 1
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
