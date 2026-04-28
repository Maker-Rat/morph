from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config


# ----------------------------------------------------------------------------
# Motion loading
# ----------------------------------------------------------------------------

def load_pkl_file(pkl_path: str) -> Tuple[List[Tuple], Optional[float]]:
    import pickle

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = None
    if isinstance(data, dict):
        fps = data.get("fps", None)
        base_trans = data.get("root_pos", data.get("base_trans", data.get("base_translation", None)))
        base_rot = data.get("root_rot", data.get("base_rot", data.get("base_rotation", None)))
        joint_pos = data.get("dof_pos", data.get("joint_pos", data.get("joint_positions", None)))
        if base_trans is None or base_rot is None or joint_pos is None:
            raise ValueError(f"Missing required keys. Available: {list(data.keys())}")
        n_frames = len(base_trans)
        motion_data = [(base_trans[i], base_rot[i], joint_pos[i]) for i in range(n_frames)]
        return motion_data, fps

    if isinstance(data, list):
        return data, None

    raise ValueError(f"Unsupported data type: {type(data)}")


# ----------------------------------------------------------------------------
# Root features
# ----------------------------------------------------------------------------

def compute_world_linear_vel(base_trans: np.ndarray, dt: float, max_vel: float = 10.0) -> np.ndarray:
    n_frames = len(base_trans)
    linear_vel = np.zeros_like(base_trans)

    if n_frames > 2:
        linear_vel[1:-1] = (base_trans[2:] - base_trans[:-2]) / (2 * dt)
    if n_frames > 1:
        linear_vel[0] = (base_trans[1] - base_trans[0]) / dt
        linear_vel[-1] = (base_trans[-1] - base_trans[-2]) / dt

    if n_frames > 0:
        linear_vel = np.clip(linear_vel, -max_vel, max_vel)
    return linear_vel


def extract_yaw(base_rot_xyzw: np.ndarray) -> np.ndarray:
    x = base_rot_xyzw[:, 0]
    y = base_rot_xyzw[:, 1]
    z = base_rot_xyzw[:, 2]
    w = base_rot_xyzw[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_vel_to_local(lin_vel_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    lin_vel_local = np.zeros_like(lin_vel_world)
    lin_vel_local[:, 0] = cos_yaw * lin_vel_world[:, 0] + sin_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 1] = -sin_yaw * lin_vel_world[:, 0] + cos_yaw * lin_vel_world[:, 1]
    lin_vel_local[:, 2] = lin_vel_world[:, 2]
    return lin_vel_local


def compute_yaw_rate(yaw: np.ndarray, dt: float) -> np.ndarray:
    yaw_diff = np.diff(yaw, prepend=yaw[0])
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    return (yaw_diff / dt)[:, np.newaxis]


def compute_root_features(base_trans: np.ndarray, base_rot: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lin_vel_world = compute_world_linear_vel(base_trans, dt)
    yaw = extract_yaw(base_rot)
    lin_vel_local = world_vel_to_local(lin_vel_world, yaw)
    yaw_rate = compute_yaw_rate(yaw, dt)
    return lin_vel_local, yaw_rate, yaw


# ----------------------------------------------------------------------------
# Jump split
# ----------------------------------------------------------------------------

def detect_position_jumps(base_trans: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    if len(base_trans) < 2:
        return np.zeros(len(base_trans), dtype=bool)
    diffs = np.diff(base_trans, axis=0)
    jump_magnitudes = np.linalg.norm(diffs, axis=1)
    jumps = np.zeros(len(base_trans), dtype=bool)
    jumps[1:] = jump_magnitudes > threshold
    return jumps


def split_at_jumps(motion_data: List[Tuple], jump_threshold: float = 2.0) -> List[List[Tuple]]:
    if len(motion_data) < 2:
        return [motion_data]
    base_trans = np.asarray([frame[0] for frame in motion_data])
    jumps = detect_position_jumps(base_trans, jump_threshold)
    jump_indices = np.where(jumps)[0]
    if len(jump_indices) == 0:
        return [motion_data]

    segments = []
    start_idx = 0
    for jump_idx in jump_indices:
        if jump_idx > start_idx:
            segments.append(motion_data[start_idx:jump_idx])
        start_idx = jump_idx
    if start_idx < len(motion_data):
        segments.append(motion_data[start_idx:])
    return segments


# ----------------------------------------------------------------------------
# Mirror augmentation
# ----------------------------------------------------------------------------

def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _rotmat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    return q


def mirror_quaternion_sequence_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    mirror_mat = np.diag([1.0, -1.0, 1.0])
    out = np.zeros_like(quat_xyzw)
    prev = None

    for i in range(quat_xyzw.shape[0]):
        q = quat_xyzw[i]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            q_m = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q = q / q_norm
            r = _quat_to_rotmat_xyzw(q)
            r_m = mirror_mat @ r @ mirror_mat
            q_m = _rotmat_to_quat_xyzw(r_m)

        if prev is not None and float(np.dot(prev, q_m)) < 0.0:
            q_m = -q_m
        out[i] = q_m
        prev = q_m

    return out.astype(np.float32)


def default_mirror_specs() -> Dict[str, Dict[str, List]]:
    return {
        "g1": {
            "swap_pairs": [
                [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11],
                [15, 22], [16, 23], [17, 24], [18, 25], [19, 26], [20, 27], [21, 28],
            ],
            "sign_flip_indices": [1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28],
        },
        "unitree_g1": {
            "swap_pairs": [
                [0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11],
                [15, 22], [16, 23], [17, 24], [18, 25], [19, 26], [20, 27], [21, 28],
            ],
            "sign_flip_indices": [1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28],
        },
        "go2": {
            "swap_pairs": [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]],
            "sign_flip_indices": [0, 3, 6, 9],
        },
        "unitree_go2": {
            "swap_pairs": [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]],
            "sign_flip_indices": [0, 3, 6, 9],
        },
    }


def mirror_joint_positions(joint_pos: np.ndarray, robot_id: str, specs: Dict[str, Dict[str, List]]) -> np.ndarray:
    if robot_id not in specs:
        raise ValueError(
            f"No mirror spec found for robot '{robot_id}'. "
            f"Provide --mirror-spec-json with swap_pairs/sign_flip_indices."
        )
    spec = specs[robot_id]
    swap_pairs = [tuple(x) for x in spec.get("swap_pairs", [])]
    sign_flip = list(spec.get("sign_flip_indices", []))

    out = joint_pos.copy()
    src = joint_pos.copy()
    for i, j in swap_pairs:
        out[:, i] = src[:, j]
        out[:, j] = src[:, i]
    out[:, sign_flip] *= -1.0
    return out


def mirror_motion_segment(segment: List[Tuple], robot_id: str, specs: Dict[str, Dict[str, List]]) -> List[Tuple]:
    if len(segment) == 0:
        return []

    base_trans = np.asarray([f[0] for f in segment], dtype=np.float32)
    base_rot = np.asarray([f[1] for f in segment], dtype=np.float32)
    joint_pos = np.asarray([f[2] for f in segment], dtype=np.float32)

    base_trans_m = base_trans.copy()
    base_trans_m[:, 1] *= -1.0
    base_rot_m = mirror_quaternion_sequence_xyzw(base_rot)
    joint_pos_m = mirror_joint_positions(joint_pos, robot_id, specs)

    return [(base_trans_m[i], base_rot_m[i], joint_pos_m[i]) for i in range(len(segment))]


# ----------------------------------------------------------------------------
# Skeleton and windows
# ----------------------------------------------------------------------------

def parse_mujoco_skeleton(xml_path: str) -> Dict:
    model = mujoco.MjModel.from_xml_path(xml_path)
    n_bodies = model.nbody - 1
    n_joints = model.njnt

    parents = [-1] * n_bodies
    offsets = np.zeros((n_bodies, 3), dtype=np.float32)
    body_names = []
    joint_names = []
    joint_ranges = []

    for body_id in range(1, model.nbody):
        body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        parent_id = model.body_parentid[body_id]
        parents[body_id - 1] = int(parent_id)
        offsets[body_id - 1] = model.body_pos[body_id]

    for jnt_id in range(n_joints):
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or "")
        joint_ranges.append(model.jnt_range[jnt_id].tolist())

    return {
        "parents": np.asarray(parents, dtype=np.int64),
        "offsets": offsets,
        "body_names": body_names,
        "joint_names": joint_names,
        "joint_ranges": np.asarray(joint_ranges, dtype=np.float32),
        "n_joints": int(n_joints),
        "n_bodies": int(n_bodies),
    }


def create_sliding_windows(
    base_trans: np.ndarray,
    base_rot: np.ndarray,
    joint_pos: np.ndarray,
    lin_vel_local: np.ndarray,
    yaw_rate: np.ndarray,
    yaw: np.ndarray,
    window_size: int,
    stride: int,
) -> Optional[Dict]:
    n_frames = len(base_trans)
    if n_frames < window_size:
        return None

    n_windows = (n_frames - window_size) // stride + 1
    if n_windows <= 0:
        return None

    out = {
        "joint_pos": [],
        "lin_vel_local": [],
        "yaw_rate": [],
        "base_trans": [],
        "base_rot": [],
        "yaw": [],
    }
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        out["joint_pos"].append(joint_pos[start:end])
        out["lin_vel_local"].append(lin_vel_local[start:end])
        out["yaw_rate"].append(yaw_rate[start:end])
        out["base_trans"].append(base_trans[start:end])
        out["base_rot"].append(base_rot[start:end])
        out["yaw"].append(yaw[start:end])

    return {k: np.asarray(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------

def process_pkl_directory(
    pkl_dir: Path,
    robot_id: str,
    dt_default: float,
    window_size: int,
    stride: int,
    handle_jumps: bool,
    jump_threshold: float,
    min_segment_length: int,
    max_frames: Optional[int],
    max_windows: Optional[int],
    augment_mirror: bool,
    mirror_specs: Dict[str, Dict[str, List]],
) -> Dict:
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No PKL files found in {pkl_dir}")

    all_data = {
        "joint_pos": [],
        "lin_vel_local": [],
        "yaw_rate": [],
        "base_trans": [],
        "base_rot": [],
        "yaw": [],
    }

    total_windows = 0

    for pkl_file in pkl_files:
        try:
            motion_data, file_fps = load_pkl_file(str(pkl_file))
            dt = (1.0 / file_fps) if file_fps else dt_default

            if max_frames is not None:
                motion_data = motion_data[:max_frames]

            segments = split_at_jumps(motion_data, jump_threshold) if handle_jumps else [motion_data]

            for segment in segments:
                if len(segment) < min_segment_length:
                    continue

                variants: List[List[Tuple]] = [segment]
                if augment_mirror:
                    variants.append(mirror_motion_segment(segment, robot_id, mirror_specs))

                for work_segment in variants:
                    base_trans = np.asarray([f[0] for f in work_segment], dtype=np.float32)
                    base_rot = np.asarray([f[1] for f in work_segment], dtype=np.float32)
                    joint_pos = np.asarray([f[2] for f in work_segment], dtype=np.float32)

                    lin_vel_local, yaw_rate, yaw = compute_root_features(base_trans, base_rot, dt)
                    windowed = create_sliding_windows(
                        base_trans=base_trans,
                        base_rot=base_rot,
                        joint_pos=joint_pos,
                        lin_vel_local=lin_vel_local,
                        yaw_rate=yaw_rate,
                        yaw=yaw,
                        window_size=window_size,
                        stride=stride,
                    )
                    if windowed is None:
                        continue

                    for key in all_data.keys():
                        all_data[key].append(windowed[key])

                    total_windows += int(len(windowed["joint_pos"]))
                    if max_windows is not None and total_windows >= max_windows:
                        break

                if max_windows is not None and total_windows >= max_windows:
                    break

        except Exception as exc:
            print(f"  [warn] skipping {pkl_file.name}: {exc}")

        if max_windows is not None and total_windows >= max_windows:
            break

    if not all_data["joint_pos"]:
        raise ValueError(f"No valid data produced from {pkl_dir}")

    return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------

def train_test_split(data: Dict, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Dict, Dict]:
    np.random.seed(seed)
    n = len(data["joint_pos"])
    n_train = int(n * train_ratio)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return ({k: v[train_idx] for k, v in data.items()}, {k: v[test_idx] for k, v in data.items()})


def compute_statistics(train_data: Dict) -> Dict:
    all_features = []
    for i in range(len(train_data["joint_pos"])):
        for t in range(train_data["joint_pos"].shape[1]):
            frame = np.concatenate(
                [
                    train_data["joint_pos"][i, t],
                    train_data["lin_vel_local"][i, t],
                    train_data["yaw_rate"][i, t],
                ]
            )
            all_features.append(frame)

    all_features = np.asarray(all_features, dtype=np.float32)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _resolve_robot_paths(output_root: Path, robot_id: str) -> Tuple[Path, Path]:
    robot_cfg = output_root / "configs" / "robots" / f"{robot_id}.yaml"
    spec = load_robot_spec(robot_cfg)

    source_xml = spec.source_xml if spec.source_xml.is_absolute() else (output_root / spec.source_xml)
    fk_xml = spec.fk_xml if spec.fk_xml.is_absolute() else (output_root / spec.fk_xml)
    return source_xml.resolve(), fk_xml.resolve()


def _create_robot_dataset(
    output_root: Path,
    robot_id: str,
    pkl_dir: Path,
    processed_dir: Path,
    dt_default: float,
    window_size: int,
    stride: int,
    train_ratio: float,
    seed: int,
    handle_jumps: bool,
    jump_threshold: float,
    min_segment_length: int,
    max_frames: Optional[int],
    max_windows: Optional[int],
    augment_mirror: bool,
    mirror_specs: Dict[str, Dict[str, List]],
) -> Dict:
    source_xml, _ = _resolve_robot_paths(output_root, robot_id)

    all_data = process_pkl_directory(
        pkl_dir=pkl_dir,
        robot_id=robot_id,
        dt_default=dt_default,
        window_size=window_size,
        stride=stride,
        handle_jumps=handle_jumps,
        jump_threshold=jump_threshold,
        min_segment_length=min_segment_length,
        max_frames=max_frames,
        max_windows=max_windows,
        augment_mirror=augment_mirror,
        mirror_specs=mirror_specs,
    )

    train_data, test_data = train_test_split(all_data, train_ratio, seed)
    stats = compute_statistics(train_data)
    skeleton = parse_mujoco_skeleton(str(source_xml))
    stats.update(
        {
            "parents": skeleton["parents"],
            "offsets": skeleton["offsets"],
            "body_names": np.asarray(skeleton["body_names"], dtype=object),
            "joint_names": np.asarray(skeleton["joint_names"], dtype=object),
            "joint_ranges": skeleton["joint_ranges"],
            "n_joints": skeleton["n_joints"],
            "n_bodies": skeleton["n_bodies"],
        }
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_path = processed_dir / f"{robot_id}_stats.npz"
    train_path = processed_dir / f"{robot_id}_train.npz"
    test_path = processed_dir / f"{robot_id}_test.npz"

    np.savez_compressed(stats_path, **stats)
    np.savez_compressed(train_path, **train_data)
    np.savez_compressed(test_path, **test_data)

    return {
        "robot_id": robot_id,
        "stats_path": str(stats_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "total_windows": int(len(all_data["joint_pos"])),
        "train_windows": int(len(train_data["joint_pos"])),
        "test_windows": int(len(test_data["joint_pos"])),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create refactor-ready processed datasets from PKL motions.")
    p.add_argument("--output-root", type=str, default=".")

    # Optional pair shortcut.
    p.add_argument("--task-family", type=str, default=None)
    p.add_argument("--pair-id", type=str, default=None)

    # Single robot mode.
    p.add_argument("--robot", type=str, default=None)
    p.add_argument("--pkl-dir", type=str, default=None)

    # Explicit dual-robot mode.
    p.add_argument("--src-robot", type=str, default=None)
    p.add_argument("--src-pkl-dir", type=str, default=None)
    p.add_argument("--dst-robot", type=str, default=None)
    p.add_argument("--dst-pkl-dir", type=str, default=None)

    p.add_argument("--processed-dir", type=str, default=None,
                   help="Default: <output-root>/data/processed")

    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt-default", type=float, default=1.0 / 30.0)

    p.add_argument("--handle-jumps", action="store_true", default=True)
    p.add_argument("--no-handle-jumps", dest="handle_jumps", action="store_false")
    p.add_argument("--jump-threshold", type=float, default=2.0)
    p.add_argument("--min-segment-length", type=int, default=64)

    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--max-windows", type=int, default=0,
                   help="Used in single-robot mode")
    p.add_argument("--max-windows-src", type=int, default=0)
    p.add_argument("--max-windows-dst", type=int, default=0)

    p.add_argument("--augment-mirror", action="store_true", default=False)
    p.add_argument("--mirror-spec-json", type=str, default=None,
                   help="Optional JSON map: robot_id -> {swap_pairs, sign_flip_indices}")

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_mode(args: argparse.Namespace, output_root: Path):
    pair_mode = args.task_family is not None or args.pair_id is not None
    single_mode = args.robot is not None or args.pkl_dir is not None
    explicit_dual_mode = any([args.src_robot, args.src_pkl_dir, args.dst_robot, args.dst_pkl_dir])

    if single_mode and explicit_dual_mode:
        raise ValueError("Use either single-robot args (--robot/--pkl-dir) or src/dst args, not both")

    if single_mode:
        if args.robot is None or args.pkl_dir is None:
            raise ValueError("Single-robot mode requires both --robot and --pkl-dir")
        return {
            "mode": "single",
            "robot_id": str(args.robot),
            "pkl_dir": Path(args.pkl_dir).expanduser().resolve(),
            "meta_name": f"{args.robot}_dataset_meta.json",
        }

    if pair_mode:
        if args.task_family is None or args.pair_id is None:
            raise ValueError("Both --task-family and --pair-id are required when using pair shortcut")
        resolved = resolve_task_config(output_root, args.task_family, args.pair_id)
        if args.src_pkl_dir is None or args.dst_pkl_dir is None:
            raise ValueError("Pair shortcut mode requires --src-pkl-dir and --dst-pkl-dir")
        return {
            "mode": "dual",
            "src_robot": resolved.src_robot,
            "dst_robot": resolved.dst_robot,
            "src_pkl_dir": Path(args.src_pkl_dir).expanduser().resolve(),
            "dst_pkl_dir": Path(args.dst_pkl_dir).expanduser().resolve(),
            "meta_name": f"{args.task_family}_{args.pair_id}_dataset_meta.json",
            "task_family": args.task_family,
            "pair_id": args.pair_id,
        }

    if explicit_dual_mode:
        if not all([args.src_robot, args.src_pkl_dir, args.dst_robot, args.dst_pkl_dir]):
            raise ValueError("Explicit dual mode requires --src-robot --src-pkl-dir --dst-robot --dst-pkl-dir")
        return {
            "mode": "dual",
            "src_robot": str(args.src_robot),
            "dst_robot": str(args.dst_robot),
            "src_pkl_dir": Path(args.src_pkl_dir).expanduser().resolve(),
            "dst_pkl_dir": Path(args.dst_pkl_dir).expanduser().resolve(),
            "meta_name": f"{args.src_robot}_to_{args.dst_robot}_dataset_meta.json",
            "task_family": None,
            "pair_id": None,
        }

    raise ValueError(
        "No valid mode selected. Use either:\n"
        "  1) --robot --pkl-dir\n"
        "  2) --src-robot --src-pkl-dir --dst-robot --dst-pkl-dir\n"
        "  3) --task-family --pair-id with --src-pkl-dir --dst-pkl-dir"
    )


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    mode_cfg = _resolve_mode(args, output_root)

    processed_dir = (
        Path(args.processed_dir).expanduser().resolve()
        if args.processed_dir
        else (output_root / "data" / "processed").resolve()
    )

    mirror_specs = default_mirror_specs()
    if args.mirror_spec_json:
        with open(args.mirror_spec_json, "r", encoding="utf-8") as f:
            custom_specs = json.load(f)
        for key, value in custom_specs.items():
            mirror_specs[str(key)] = value

    max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)
    max_windows = None if int(args.max_windows) <= 0 else int(args.max_windows)
    max_windows_src = None if int(args.max_windows_src) <= 0 else int(args.max_windows_src)
    max_windows_dst = None if int(args.max_windows_dst) <= 0 else int(args.max_windows_dst)

    print("Creating refactor datasets:")
    print(f"  mode: {mode_cfg['mode']}")
    print(f"  processed_dir: {processed_dir}")
    print(f"  window={args.window_size} stride={args.stride} train_ratio={args.train_ratio}")
    print(f"  mirror augmentation: {args.augment_mirror}")

    if mode_cfg["mode"] == "single":
        print(f"  robot: {mode_cfg['robot_id']}  from {mode_cfg['pkl_dir']}")
        if not mode_cfg["pkl_dir"].exists():
            raise FileNotFoundError(f"pkl-dir not found: {mode_cfg['pkl_dir']}")
    else:
        print(f"  src robot: {mode_cfg['src_robot']}  from {mode_cfg['src_pkl_dir']}")
        print(f"  dst robot: {mode_cfg['dst_robot']}  from {mode_cfg['dst_pkl_dir']}")
        if not mode_cfg["src_pkl_dir"].exists():
            raise FileNotFoundError(f"src-pkl-dir not found: {mode_cfg['src_pkl_dir']}")
        if not mode_cfg["dst_pkl_dir"].exists():
            raise FileNotFoundError(f"dst-pkl-dir not found: {mode_cfg['dst_pkl_dir']}")

    if args.dry_run:
        print("Dry-run mode enabled; no files written.")
        return

    if mode_cfg["mode"] == "single":
        robot_summary = _create_robot_dataset(
            output_root=output_root,
            robot_id=mode_cfg["robot_id"],
            pkl_dir=mode_cfg["pkl_dir"],
            processed_dir=processed_dir,
            dt_default=float(args.dt_default),
            window_size=int(args.window_size),
            stride=int(args.stride),
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            handle_jumps=bool(args.handle_jumps),
            jump_threshold=float(args.jump_threshold),
            min_segment_length=int(args.min_segment_length),
            max_frames=max_frames,
            max_windows=max_windows,
            augment_mirror=bool(args.augment_mirror),
            mirror_specs=mirror_specs,
        )

        summary = {"robot": robot_summary}
        meta_path = processed_dir / mode_cfg["meta_name"]
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Done.")
        print(
            f"  windows: total={robot_summary['total_windows']} "
            f"train={robot_summary['train_windows']} test={robot_summary['test_windows']}"
        )
        print(f"  meta: {meta_path}")
        return

    src_summary = _create_robot_dataset(
        output_root=output_root,
        robot_id=mode_cfg["src_robot"],
        pkl_dir=mode_cfg["src_pkl_dir"],
        processed_dir=processed_dir,
        dt_default=float(args.dt_default),
        window_size=int(args.window_size),
        stride=int(args.stride),
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        handle_jumps=bool(args.handle_jumps),
        jump_threshold=float(args.jump_threshold),
        min_segment_length=int(args.min_segment_length),
        max_frames=max_frames,
        max_windows=max_windows_src,
        augment_mirror=bool(args.augment_mirror),
        mirror_specs=mirror_specs,
    )

    dst_summary = _create_robot_dataset(
        output_root=output_root,
        robot_id=mode_cfg["dst_robot"],
        pkl_dir=mode_cfg["dst_pkl_dir"],
        processed_dir=processed_dir,
        dt_default=float(args.dt_default),
        window_size=int(args.window_size),
        stride=int(args.stride),
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        handle_jumps=bool(args.handle_jumps),
        jump_threshold=float(args.jump_threshold),
        min_segment_length=int(args.min_segment_length),
        max_frames=max_frames,
        max_windows=max_windows_dst,
        augment_mirror=bool(args.augment_mirror),
        mirror_specs=mirror_specs,
    )

    summary = {
        "task_family": mode_cfg.get("task_family"),
        "pair_id": mode_cfg.get("pair_id"),
        "src": src_summary,
        "dst": dst_summary,
    }
    meta_path = processed_dir / mode_cfg["meta_name"]
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"  src windows: total={src_summary['total_windows']} train={src_summary['train_windows']} test={src_summary['test_windows']}")
    print(f"  dst windows: total={dst_summary['total_windows']} train={dst_summary['train_windows']} test={dst_summary['test_windows']}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
