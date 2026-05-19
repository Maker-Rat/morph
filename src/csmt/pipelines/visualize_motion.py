from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import torch

from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config
from csmt.utils.differentiable_fk import ForwardKinematics


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


@dataclass
class ContactDebug:
    dst_contact: np.ndarray      # [T, K]
    dst_feet_indices: np.ndarray # [K]
    src_time_gate: Optional[np.ndarray]
    dst_ground_z: Optional[float]


@dataclass
class EEDebug:
    src_ee_base_rel: np.ndarray   # [T, K, 3]
    target_ee_base_rel: np.ndarray  # [T, K, 3] target actually visualized
    dst_ref_base_rel: Optional[np.ndarray]  # [1, K, 3] anchor for displacement mode
    dst_ee_indices: np.ndarray    # [K]
    base_body_name: str
    src_robot: str
    dst_robot: str
    target_mode: str


def _load_contact_debug(path: Path) -> ContactDebug:
    z = np.load(path, allow_pickle=False)
    if "dst_contact" not in z or "dst_feet_indices" not in z:
        raise ValueError("contact debug npz must contain dst_contact and dst_feet_indices")

    dst_contact = np.asarray(z["dst_contact"], dtype=np.float32)
    if dst_contact.ndim == 1:
        dst_contact = dst_contact[:, None]
    if dst_contact.ndim != 2:
        raise ValueError(f"dst_contact must be [T,K], got shape {dst_contact.shape}")

    dst_feet_indices = np.asarray(z["dst_feet_indices"], dtype=np.int32).reshape(-1)
    if len(dst_feet_indices) == 0:
        raise ValueError("dst_feet_indices is empty")

    src_time_gate = None
    if "src_time_gate" in z:
        src_time_gate = np.asarray(z["src_time_gate"], dtype=np.float32).reshape(-1)

    dst_ground_z = None
    if "dst_ground_z" in z:
        g = np.asarray(z["dst_ground_z"], dtype=np.float32).reshape(-1)
        if g.size > 0:
            dst_ground_z = float(g[0])

    return ContactDebug(
        dst_contact=dst_contact,
        dst_feet_indices=dst_feet_indices,
        src_time_gate=src_time_gate,
        dst_ground_z=dst_ground_z,
    )


def _contact_rgba(c: float) -> np.ndarray:
    c = float(np.clip(c, 0.0, 1.0))
    # Red (no contact) -> Green (contact)
    return np.array([1.0 - c, c, 0.0, 0.92], dtype=np.float32)


def _extract_yaw_xyzw(base_rot_xyzw: np.ndarray) -> np.ndarray:
    x = base_rot_xyzw[:, 0]
    y = base_rot_xyzw[:, 1]
    z = base_rot_xyzw[:, 2]
    w = base_rot_xyzw[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_vel_to_local(lin_vel_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    lin_vel_local = np.zeros_like(lin_vel_world)
    lin_vel_local[:, 0] = cos_yaw * lin_vel_world[:, 0] + sin_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 1] = -sin_yaw * lin_vel_world[:, 0] + cos_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 2] = lin_vel_world[:, 2]
    return lin_vel_local


def _compute_world_linear_vel(base_trans: np.ndarray, dt: float, max_vel: float = 10.0) -> np.ndarray:
    n_frames = int(base_trans.shape[0])
    lin_vel = np.zeros_like(base_trans)
    if n_frames > 2:
        lin_vel[1:-1] = (base_trans[2:] - base_trans[:-2]) / (2 * dt)
    if n_frames > 1:
        lin_vel[0] = (base_trans[1] - base_trans[0]) / dt
        lin_vel[-1] = (base_trans[-1] - base_trans[-2]) / dt
    return np.clip(lin_vel, -max_vel, max_vel)


def _compute_yaw_rate(yaw: np.ndarray, dt: float) -> np.ndarray:
    yaw_diff = np.diff(yaw, prepend=yaw[0])
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    return yaw_diff / dt


def _build_motion_features_from_pkl(dof_pos: np.ndarray, root_pos: np.ndarray, root_rot_xyzw: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / max(float(fps), 1e-8)
    yaw = _extract_yaw_xyzw(root_rot_xyzw)
    lin_vel_world = _compute_world_linear_vel(root_pos, dt)
    lin_vel_local = _world_vel_to_local(lin_vel_world, yaw)
    yaw_rate = _compute_yaw_rate(yaw, dt)
    return np.concatenate([dof_pos, lin_vel_local, yaw_rate[:, None]], axis=-1).astype(np.float32)


def _init_geom_sphere(geom, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def _init_geom_plane_patch(geom, center: np.ndarray, z: float, rgba: np.ndarray) -> None:
    p = np.asarray(center, dtype=np.float64).copy()
    p[2] = float(z)
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.array([0.7, 0.7, 0.0015], dtype=np.float64),
        pos=p,
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def _draw_contact_overlay(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_idx: int,
    debug: ContactDebug,
    marker_radius: float,
    body_id_offset: int,
) -> int:
    scn = viewer.user_scn

    if debug.dst_contact.shape[0] <= 0:
        return int(scn.ngeom)
    t = int(np.clip(frame_idx, 0, debug.dst_contact.shape[0] - 1))

    # Center helper markers near base body if available.
    base_bid = int(np.clip(body_id_offset, 0, model.nbody - 1))
    base_pos = data.xpos[base_bid].copy()

    gidx = int(scn.ngeom)
    maxgeom = int(scn.maxgeom)

    # Contact reference plane at MuJoCo world ground (z=0).
    if gidx < maxgeom:
        _init_geom_plane_patch(
            scn.geoms[gidx],
            center=base_pos,
            z=0.0,
            rgba=np.array([0.25, 0.55, 1.0, 0.22], dtype=np.float32),
        )
        gidx += 1

    # Source time-gate indicator sphere above base.
    if debug.src_time_gate is not None and debug.src_time_gate.size > 0 and gidx < maxgeom:
        gate_v = float(np.clip(debug.src_time_gate[min(t, debug.src_time_gate.shape[0] - 1)], 0.0, 1.0))
        gate_pos = base_pos.copy()
        gate_pos[2] += 0.23
        _init_geom_sphere(scn.geoms[gidx], gate_pos, marker_radius * 0.9, _contact_rgba(gate_v))
        gidx += 1

    n_feet = min(debug.dst_contact.shape[1], debug.dst_feet_indices.shape[0])
    for k in range(n_feet):
        if gidx >= maxgeom:
            break
        # FK/body-index space excludes world body; MuJoCo body ids include world at 0.
        bid = int(debug.dst_feet_indices[k]) + int(body_id_offset)
        if bid < 0 or bid >= model.nbody:
            continue

        c = float(np.clip(debug.dst_contact[t, k], 0.0, 1.0))
        foot_pos = data.xpos[bid].copy()
        _init_geom_sphere(scn.geoms[gidx], foot_pos, marker_radius, _contact_rgba(c))
        gidx += 1

        # Vertical connector from MuJoCo world ground (z=0) to foot.
        if gidx < maxgeom:
            p0 = np.array([foot_pos[0], foot_pos[1], 0.0], dtype=np.float64)
            p1 = np.asarray(foot_pos, dtype=np.float64)
            mujoco.mjv_initGeom(
                scn.geoms[gidx],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=np.array([0.95, 0.95, 0.95, 0.85], dtype=np.float32),
            )
            mujoco.mjv_connector(scn.geoms[gidx], mujoco.mjtGeom.mjGEOM_LINE, 2.0, p0, p1)
            gidx += 1

    scn.ngeom = gidx
    return gidx


def _draw_ee_overlay(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_idx: int,
    debug: EEDebug,
    marker_radius: float,
    body_id_offset: int,
) -> int:
    scn = viewer.user_scn
    gidx = int(scn.ngeom)
    maxgeom = int(scn.maxgeom)

    if debug.target_ee_base_rel.shape[0] <= 0 or debug.dst_ee_indices.size <= 0:
        return gidx

    t = int(np.clip(frame_idx, 0, debug.target_ee_base_rel.shape[0] - 1))
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, debug.base_body_name)
    if base_bid < 0:
        base_bid = int(np.clip(body_id_offset, 0, model.nbody - 1))
    base_pos = data.xpos[base_bid].copy()

    n_pairs = min(int(debug.dst_ee_indices.size), int(debug.target_ee_base_rel.shape[1]))
    for k in range(n_pairs):
        if gidx >= maxgeom:
            break
        dst_bid = int(debug.dst_ee_indices[k]) + int(body_id_offset)
        if dst_bid < 0 or dst_bid >= model.nbody:
            continue

        target_pos = base_pos + debug.target_ee_base_rel[t, k]
        actual_pos = data.xpos[dst_bid].copy()

        # Displacement anchor (dst reference point) for better interpretation.
        if debug.target_mode == "displacement" and debug.dst_ref_base_rel is not None and gidx < maxgeom:
            anchor_pos = base_pos + debug.dst_ref_base_rel[0, k]
            _init_geom_sphere(
                scn.geoms[gidx],
                np.asarray(anchor_pos, dtype=np.float64),
                marker_radius * 0.55,
                np.array([1.0, 0.95, 0.2, 0.88], dtype=np.float32),
            )
            gidx += 1
            if gidx >= maxgeom:
                break

        # Target marker: cyan
        _init_geom_sphere(
            scn.geoms[gidx],
            np.asarray(target_pos, dtype=np.float64),
            marker_radius * 1.6,
            np.array([0.2, 0.8, 1.0, 0.92], dtype=np.float32),
        )
        gidx += 1
        if gidx >= maxgeom:
            break

        # Actual marker: magenta
        _init_geom_sphere(
            scn.geoms[gidx],
            np.asarray(actual_pos, dtype=np.float64),
            marker_radius * 0.55,
            np.array([1.0, 0.2, 0.9, 0.92], dtype=np.float32),
        )
        gidx += 1
        if gidx >= maxgeom:
            break

        # Connector line actual -> target
        mujoco.mjv_initGeom(
            scn.geoms[gidx],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.array([0.9, 0.9, 0.9, 0.85], dtype=np.float32),
        )
        mujoco.mjv_connector(
            scn.geoms[gidx],
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0,
            np.asarray(actual_pos, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
        )
        gidx += 1

    scn.ngeom = gidx
    return gidx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic robot motion viewer (qpos-based, joint-index order).")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--robot-id", type=str, required=True, help="Robot ID in configs/robots/<robot-id>.yaml")
    p.add_argument("--pkl", type=str, default=None, help="Optional motion PKL path")
    p.add_argument("--xml", type=str, default=None, help="Override XML path")
    p.add_argument("--fps", type=float, default=None, help="Playback FPS override")
    p.add_argument("--start-frame", type=int, default=0, help="First frame to visualize, inclusive.")
    p.add_argument("--end-frame", type=int, default=0, help="End frame to visualize, exclusive. 0 means end of clip.")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--quat-convention", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--contact-debug-npz", type=str, default=None,
                   help="Optional contact debug npz from infer_teacher --save-contact-debug")
    p.add_argument("--contact-marker-radius", type=float, default=0.028,
                   help="Radius of per-foot contact spheres")
    p.add_argument("--contact-body-id-offset", type=int, default=1,
                   help="Offset to map FK body indices to MuJoCo body ids (default: 1, because world body is id 0)")
    p.add_argument("--ee-source-pkl", type=str, default=None,
                   help="Optional source motion PKL used to compute EE target trajectories")
    p.add_argument("--ee-task-family", type=str, default=None,
                   help="Task family for EE correspondence lookup (required with --ee-source-pkl)")
    p.add_argument("--ee-pair-id", type=str, default=None,
                   help="Pair id for EE correspondence lookup (required with --ee-source-pkl)")
    p.add_argument("--ee-marker-radius", type=float, default=0.024,
                   help="Radius of EE target/actual markers")
    p.add_argument("--ee-body-id-offset", type=int, default=1,
                   help="Offset to map FK body indices to MuJoCo body ids for EE markers")
    p.add_argument("--ee-target-mode", choices=["absolute", "displacement"], default="displacement",
                   help="EE target visualization mode. 'displacement' matches displacement-based EE objective.")
    p.add_argument("--ee-ref-frames", type=int, default=10,
                   help="Reference frame count for displacement EE target mode.")
    p.add_argument("--ee-disp-scale-mode", choices=["none", "loss_ratio"], default="none",
                   help="In displacement mode, optionally scale src displacement by dst/src magnitude ratio "
                        "to better match current loss normalization behavior.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    robot_cfg_path = output_root / "configs" / "robots" / f"{args.robot_id}.yaml"
    robot_spec = load_robot_spec(robot_cfg_path)
    xml_path = Path(args.xml).expanduser().resolve() if args.xml else (
        robot_spec.source_xml if robot_spec.source_xml.is_absolute() else (output_root / robot_spec.source_xml)
    )
    xml_path = xml_path.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    has_free_base = model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) and model.nq >= 7
    joint_qpos = _get_non_free_joint_qpos_addrs(model)
    print(f"XML: {xml_path}")
    print(f"Model: nq={model.nq} nv={model.nv} njnt={model.njnt}")
    print(f"Free base: {has_free_base}")

    contact_debug = None
    if args.contact_debug_npz is not None:
        cpath = Path(args.contact_debug_npz).expanduser().resolve()
        contact_debug = _load_contact_debug(cpath)
        print(f"Contact debug: {cpath}")
        print(f"  dst_contact shape: {contact_debug.dst_contact.shape}")
        print(f"  dst_feet_indices: {contact_debug.dst_feet_indices.tolist()}")
        print("  viewer controls: press 'c' to toggle contact overlay")

    ee_debug = None

    # No PKL mode: spawn interactive viewer for manual slider/joint testing.
    if args.pkl is None:
        print("PKL: <none> (interactive mode)")
        print("Controls: use MuJoCo viewer UI sliders to move joints/controls. Press ESC to quit.")
        if contact_debug is not None:
            print("[warn] --contact-debug-npz ignored in interactive/no-PKL mode.")
        if args.ee_source_pkl is not None:
            print("[warn] --ee-source-pkl ignored in interactive/no-PKL mode.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
        return

    pkl_path = Path(args.pkl).expanduser().resolve()
    dof_pos, root_pos, root_rot, fps_from_file = _load_motion(pkl_path)
    original_n_frames = int(dof_pos.shape[0])
    if original_n_frames == 0:
        raise ValueError("Motion has zero frames")

    start_frame = max(0, int(args.start_frame))
    end_frame = int(args.end_frame) if int(args.end_frame) > 0 else original_n_frames
    end_frame = min(original_n_frames, end_frame)
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}) for {original_n_frames} frames")
    dof_pos = dof_pos[start_frame:end_frame]
    root_pos = root_pos[start_frame:end_frame]
    root_rot = root_rot[start_frame:end_frame]
    n_frames = int(dof_pos.shape[0])

    play_fps = float(args.fps) if args.fps is not None else float(fps_from_file if fps_from_file else 30.0)
    dt = 1.0 / max(play_fps, 1e-6)

    n_model_joints = len(joint_qpos)
    n_motion_joints = int(dof_pos.shape[1])
    map_dim = min(n_model_joints, n_motion_joints)
    print(f"PKL: {pkl_path}")
    print(f"Frames: {n_frames} from range [{start_frame}, {end_frame}) of {original_n_frames}, playback_fps: {play_fps:.3f}")
    print(f"Joint dims: motion={n_motion_joints}, model_non_free={n_model_joints}, mapped={map_dim}")
    if n_model_joints != n_motion_joints:
        print("[warn] Motion joint dim does not exactly match model non-free joints; using min(motion, model).")

    root_motion_mag = float(np.linalg.norm(root_pos[-1] - root_pos[0])) if n_frames > 1 else 0.0
    if (not has_free_base) and root_motion_mag > 1e-5:
        print("[warn] Motion has root translation, but XML has fixed base (no free joint). Root motion ignored.")

    if contact_debug is not None:
        contact_debug = ContactDebug(
            dst_contact=contact_debug.dst_contact[start_frame:end_frame],
            src_time_gate=contact_debug.src_time_gate[start_frame:end_frame],
            gated_contact=contact_debug.gated_contact[start_frame:end_frame],
            dst_feet_indices=contact_debug.dst_feet_indices,
        )

    if args.ee_source_pkl is not None:
        if args.ee_task_family is None or args.ee_pair_id is None:
            raise ValueError("--ee-task-family and --ee-pair-id are required when --ee-source-pkl is set")
        resolved = resolve_task_config(output_root, args.ee_task_family, args.ee_pair_id)
        if args.robot_id != resolved.dst_robot:
            print(
                f"[warn] EE overlay expects destination robot '{resolved.dst_robot}', "
                f"but --robot-id is '{args.robot_id}'. Skipping EE overlay."
            )
        elif len(resolved.src_ee_indices) == 0 or len(resolved.dst_ee_indices) == 0:
            print("[info] EE indices are empty for this pair; skipping EE overlay.")
        else:
            src_robot_spec = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.src_robot}.yaml")
            dst_robot_spec = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.dst_robot}.yaml")
            src_fk_path = src_robot_spec.fk_xml if src_robot_spec.fk_xml.is_absolute() else (output_root / src_robot_spec.fk_xml)
            dst_fk_path = dst_robot_spec.fk_xml if dst_robot_spec.fk_xml.is_absolute() else (output_root / dst_robot_spec.fk_xml)

            src_pkl_path = Path(args.ee_source_pkl).expanduser().resolve()
            src_dof_pos, src_root_pos, src_root_rot, src_fps_from_file = _load_motion(src_pkl_path)
            src_end_frame = min(int(src_dof_pos.shape[0]), end_frame)
            src_start_frame = min(start_frame, src_end_frame)
            src_dof_pos = src_dof_pos[src_start_frame:src_end_frame]
            src_root_pos = src_root_pos[src_start_frame:src_end_frame]
            src_root_rot = src_root_rot[src_start_frame:src_end_frame]
            src_fps = float(src_fps_from_file if src_fps_from_file else 30.0)
            src_motion_feat = _build_motion_features_from_pkl(src_dof_pos, src_root_pos, src_root_rot, src_fps)

            src_fk = ForwardKinematics(
                model_path=str(src_fk_path.resolve()),
                robot_name=str(resolved.src_robot).upper(),
                device="cpu",
            )
            src_motion_t = torch.from_numpy(src_motion_feat).float().unsqueeze(0)
            _, src_base_rel = src_fk.forward(src_motion_t, dt=1.0 / max(src_fps, 1e-8))
            src_base_rel_np = src_base_rel.squeeze(0).detach().cpu().numpy()

            dst_fps = float(play_fps)
            dst_motion_feat = _build_motion_features_from_pkl(dof_pos, root_pos, root_rot, dst_fps)
            dst_fk = ForwardKinematics(
                model_path=str(dst_fk_path.resolve()),
                robot_name=str(resolved.dst_robot).upper(),
                device="cpu",
            )
            dst_motion_t = torch.from_numpy(dst_motion_feat).float().unsqueeze(0)
            _, dst_base_rel = dst_fk.forward(dst_motion_t, dt=1.0 / max(dst_fps, 1e-8))
            dst_base_rel_np = dst_base_rel.squeeze(0).detach().cpu().numpy()

            n_pairs = min(len(resolved.src_ee_indices), len(resolved.dst_ee_indices))
            src_ee_base_rel = src_base_rel_np[:, list(resolved.src_ee_indices[:n_pairs]), :]
            dst_ee_base_rel = dst_base_rel_np[:, list(resolved.dst_ee_indices[:n_pairs]), :]
            dst_ee_indices = np.asarray(resolved.dst_ee_indices[:n_pairs], dtype=np.int32)

            t_common = min(src_ee_base_rel.shape[0], dst_ee_base_rel.shape[0], n_frames)
            src_ee_base_rel = src_ee_base_rel[:t_common]
            dst_ee_base_rel = dst_ee_base_rel[:t_common]

            target_mode = str(args.ee_target_mode).lower()
            dst_ref_base_rel = None
            if target_mode == "displacement":
                ref_frames = max(1, min(int(args.ee_ref_frames), t_common))
                src_ref = src_ee_base_rel[:ref_frames].mean(axis=0, keepdims=True)  # [1,K,3]
                dst_ref = dst_ee_base_rel[:ref_frames].mean(axis=0, keepdims=True)  # [1,K,3]
                src_disp = src_ee_base_rel - src_ref

                if str(args.ee_disp_scale_mode).lower() == "loss_ratio":
                    src_speed = float(np.linalg.norm(src_disp, axis=-1).mean())
                    dst_disp = dst_ee_base_rel - dst_ref
                    dst_speed = float(np.linalg.norm(dst_disp, axis=-1).mean())
                    if src_speed > 1e-8:
                        ratio = max(0.15, float(dst_speed / src_speed))
                        src_disp = src_disp * ratio
                    else:
                        ratio = 1.0
                    print(f"  ee_disp_scale: src={src_speed:.6f}, dst={dst_speed:.6f}, ratio={ratio:.6f}")

                target_ee_base_rel = dst_ref + src_disp
                dst_ref_base_rel = dst_ref
            else:
                target_ee_base_rel = src_ee_base_rel

            ee_debug = EEDebug(
                src_ee_base_rel=src_ee_base_rel,
                target_ee_base_rel=target_ee_base_rel,
                dst_ref_base_rel=dst_ref_base_rel,
                dst_ee_indices=dst_ee_indices,
                base_body_name=str(dst_robot_spec.base_body),
                src_robot=str(resolved.src_robot),
                dst_robot=str(resolved.dst_robot),
                target_mode=target_mode,
            )
            print(f"EE debug source: {src_pkl_path}")
            print(f"  pair: {resolved.src_robot} -> {resolved.dst_robot} ({resolved.task_family}/{resolved.pair_id})")
            print(f"  pairs: {n_pairs}, src_ee={list(resolved.src_ee_indices[:n_pairs])}, dst_ee={list(resolved.dst_ee_indices[:n_pairs])}")
            print(f"  ee_target_mode: {target_mode}, ref_frames={int(args.ee_ref_frames)}, disp_scale_mode={args.ee_disp_scale_mode}")
            print("  viewer controls: press 'e' to toggle EE overlay")

    frame = [0]
    paused = [False]
    show_contact_overlay = [True]
    show_ee_overlay = [True]

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
        elif key in (ord("c"), ord("C")):
            show_contact_overlay[0] = not show_contact_overlay[0]
            print("Contact overlay ON" if show_contact_overlay[0] else "Contact overlay OFF")
        elif key in (ord("e"), ord("E")):
            show_ee_overlay[0] = not show_ee_overlay[0]
            print("EE overlay ON" if show_ee_overlay[0] else "EE overlay OFF")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

        while viewer.is_running():
            apply_frame(frame[0])
            viewer.user_scn.ngeom = 0

            if contact_debug is not None:
                if show_contact_overlay[0]:
                    _draw_contact_overlay(
                        viewer=viewer,
                        model=model,
                        data=data,
                        frame_idx=frame[0],
                        debug=contact_debug,
                        marker_radius=float(args.contact_marker_radius),
                        body_id_offset=int(args.contact_body_id_offset),
                    )
            if ee_debug is not None and show_ee_overlay[0]:
                _draw_ee_overlay(
                    viewer=viewer,
                    model=model,
                    data=data,
                    frame_idx=frame[0],
                    debug=ee_debug,
                    marker_radius=float(args.ee_marker_radius),
                    body_id_offset=int(args.ee_body_id_offset),
                )

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
