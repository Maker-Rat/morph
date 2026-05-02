from __future__ import annotations

import argparse
import os
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from csmt.models import create_model
from csmt.models.motion_corrector import MotionCorrector
from csmt.parser.base import dict_to_object
from csmt.pipelines.create_distill_dataset import load_teacher_args
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config
from csmt.utils.loss_function import estimate_contact_from_height
from csmt.utils.utils import get_body_part


class InferenceStats:
    """Lightweight stats-only dataset view used for teacher inference."""

    def __init__(self, stats_path: str, njoints: Optional[int] = None, nbodies: Optional[int] = None):
        payload = np.load(stats_path, allow_pickle=True)
        self.parents = payload["parents"]
        self.offsets = payload["offsets"]
        self.mean = torch.from_numpy(payload["mean"]).float()
        self.std = torch.from_numpy(payload["std"]).float()

        if njoints is None:
            if "n_joints" in payload:
                njoints = int(payload["n_joints"])
            else:
                njoints = int(self.mean.shape[0] - 4)
        if nbodies is None:
            if "n_bodies" in payload:
                nbodies = int(payload["n_bodies"])
            else:
                nbodies = int(len(self.parents))

        self.njoints = int(njoints)
        self.nbodies = int(nbodies)
        # Needed by model codepaths even though we don't train here.
        self.min_vel = 0.0
        self.max_vel = 1.0

    def denorm(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        if transpose:
            x = x.transpose(1, 2)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean


def _resolve_dataset_path(robot_id: str, kind: str, roots: list[Path]) -> Optional[Path]:
    for root in roots:
        candidates = [
            root / f"{robot_id}_{kind}.npz",
            root / f"unitree_{robot_id}_{kind}.npz",
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
    return None


def _resolve_existing_path_or_search(
    provided_path: Optional[str],
    robot_id: str,
    kind: str,
    roots: list[Path],
    strict_roots: bool = False,
) -> Optional[str]:
    if provided_path is not None:
        p = Path(provided_path).expanduser()
        if p.exists():
            p_resolved = p.resolve()
            if not strict_roots:
                return str(p_resolved)
            for root in roots:
                root_resolved = root.resolve()
                try:
                    p_resolved.relative_to(root_resolved)
                    return str(p_resolved)
                except ValueError:
                    continue
        # Cross-machine robustness: when absolute path from another machine is stale,
        # try to recover by filename in current processed roots.
        candidate_name = p.name
        for root in roots:
            c = root / candidate_name
            if c.exists():
                return str(c.resolve())
    found = _resolve_dataset_path(robot_id, kind, roots)
    return str(found) if found is not None else None


def _to_legacy_correspondence(resolved) -> tuple[list[dict], list[dict]]:
    body_groups = []
    for g in resolved.correspondence_body_groups:
        src = list(g.get("src_body_indices", []))
        dst = list(g.get("dst_body_indices", []))
        if len(src) == 0 or len(dst) == 0:
            continue
        body_groups.append({
            "src_bodies": src,
            "dst_bodies": dst,
            "hum_bodies": src,
            "dog_bodies": dst,
        })

    joint_groups = []
    for g in resolved.correspondence_joint_groups:
        src = list(g.get("src_joint_indices", []))
        dst = list(g.get("dst_joint_indices", []))
        if len(src) == 0 or len(dst) == 0:
            continue
        joint_groups.append({
            "src_joints": src,
            "dst_joints": dst,
            "hum_joints": src,
            "dog_joints": dst,
        })
    return body_groups, joint_groups


def _load_motion_pkl(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _save_motion_pkl(path: str, payload):
    out_dir = os.path.dirname(os.path.abspath(path))
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _extract_motion_arrays(motion_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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


def _compute_world_linear_vel(base_trans: np.ndarray, dt: float, max_vel: float = 10.0) -> np.ndarray:
    n_frames = len(base_trans)
    lin_vel = np.zeros_like(base_trans)
    if n_frames > 2:
        lin_vel[1:-1] = (base_trans[2:] - base_trans[:-2]) / (2 * dt)
    if n_frames > 1:
        lin_vel[0] = (base_trans[1] - base_trans[0]) / dt
        lin_vel[-1] = (base_trans[-1] - base_trans[-2]) / dt
    return np.clip(lin_vel, -max_vel, max_vel)


def _extract_yaw(base_rot_xyzw: np.ndarray) -> np.ndarray:
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


def _compute_yaw_rate(yaw: np.ndarray, dt: float) -> np.ndarray:
    yaw_diff = np.diff(yaw, prepend=yaw[0])
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    return yaw_diff / dt


def _prepare_src_input(
    motion_pkl,
    src_stats: InferenceStats,
    device: torch.device,
    max_frames: int = 0,
) -> Tuple[torch.Tensor, float, float, float]:
    joint_pos, base_trans, base_rot, fps = _extract_motion_arrays(motion_pkl)
    n_frames = len(joint_pos) if int(max_frames) <= 0 else min(len(joint_pos), int(max_frames))
    joint_pos = joint_pos[:n_frames]
    base_trans = base_trans[:n_frames]
    base_rot = base_rot[:n_frames]

    dt = 1.0 / float(fps)
    yaw = _extract_yaw(base_rot)
    lin_vel_world = _compute_world_linear_vel(base_trans, dt)
    lin_vel_local = _world_vel_to_local(lin_vel_world, yaw)
    yaw_rate = _compute_yaw_rate(yaw, dt)

    motion_np = np.concatenate([joint_pos, lin_vel_local, yaw_rate[:, None]], axis=-1)
    motion_t = torch.from_numpy(motion_np).float()
    motion_t = (motion_t - src_stats.mean) / (src_stats.std + 1e-8)
    # [B, T, C]
    return motion_t.unsqueeze(0).to(device), float(yaw[0]), float(fps), float(base_trans[0, 2])


def _motion_to_pkl(motion_denorm: np.ndarray, dst_stats: InferenceStats, yaw_init: float, fps: float, start_height: float):
    t_len = int(motion_denorm.shape[0])
    nj = int(dst_stats.njoints)
    dt = 1.0 / float(fps)

    joint_pos = motion_denorm[:, :nj]
    lin_vel_local = motion_denorm[:, nj:nj + 3]
    yaw_rate = motion_denorm[:, nj + 3]

    yaw = yaw_init + np.cumsum(yaw_rate * dt)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    world_vx = cos_yaw * lin_vel_local[:, 0] - sin_yaw * lin_vel_local[:, 1]
    world_vy = sin_yaw * lin_vel_local[:, 0] + cos_yaw * lin_vel_local[:, 1]
    world_vz = lin_vel_local[:, 2]

    base_trans = np.zeros((t_len, 3), dtype=np.float32)
    base_trans[0] = np.array([0.0, 0.0, float(start_height)], dtype=np.float32)
    for i in range(1, t_len):
        base_trans[i, 0] = base_trans[i - 1, 0] + world_vx[i] * dt
        base_trans[i, 1] = base_trans[i - 1, 1] + world_vy[i] * dt
        base_trans[i, 2] = base_trans[i - 1, 2] + world_vz[i] * dt

    half_yaw = yaw * 0.5
    base_quat = np.stack(
        [
            np.zeros(t_len, dtype=np.float32),
            np.zeros(t_len, dtype=np.float32),
            np.sin(half_yaw).astype(np.float32),
            np.cos(half_yaw).astype(np.float32),
        ],
        axis=-1,
    )

    return {
        "fps": float(fps),
        "dof_pos": joint_pos.astype(np.float32),
        "root_pos": base_trans.astype(np.float32),
        "root_rot": base_quat.astype(np.float32),
        "local_body_pos": None,
        "link_body_list": None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic teacher inference: src motion PKL -> dst retargeted PKL.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--processed-dir", type=str, default=None,
                   help="Directory containing <robot>_stats.npz files. If omitted, uses output-root/data/processed.")
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--teacher-dir", type=str, required=True,
                   help="Teacher run directory containing model checkpoints and refactor_teacher_run.json")
    p.add_argument("--teacher-epoch", type=int, default=None,
                   help="Checkpoint epoch to load. Default: latest")
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Run reverse retargeting (dst->src) using the same symmetric teacher checkpoint.",
    )
    p.add_argument("--input-pkl", type=str, required=True)
    p.add_argument("--output-pkl", type=str, required=True)
    p.add_argument(
        "--corrector-ckpt",
        type=str,
        default=None,
        help="Optional corrector checkpoint (best.pt/last.pt) to post-correct teacher output.",
    )
    p.add_argument(
        "--output-src-rec-pkl",
        type=str,
        default=None,
        help="Optional path for source-topology reconstruction output PKL. "
             "Defaults to <output-pkl-stem>_src_rec.pkl",
    )
    p.add_argument(
        "--output-src-cyc-pkl",
        type=str,
        default=None,
        help="Optional path for source-topology cycle output PKL. "
             "Defaults to <output-pkl-stem>_src_cyc.pkl",
    )
    p.add_argument(
        "--save-src-debug",
        dest="save_src_debug",
        action="store_true",
        help="Save source reconstruction + cycle debug PKLs (default: enabled).",
    )
    p.add_argument(
        "--no-save-src-debug",
        dest="save_src_debug",
        action="store_false",
        help="Disable source reconstruction + cycle debug PKLs.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--save-contact-debug",
        action="store_true",
        help="Save source/destination contact estimates and source-gated destination contact plots.",
    )
    p.add_argument(
        "--contact-debug-prefix",
        type=str,
        default=None,
        help="Output prefix for contact debug files. Defaults to <output-pkl-stem>_contact_debug",
    )
    p.add_argument("--ground-margin", type=float, default=0.05)
    p.add_argument("--physics-ref-frames", type=int, default=5)
    p.add_argument(
        "--dst-start-height-mode",
        type=str,
        choices=["fixed", "from_ground_estimate"],
        default="fixed",
        help=(
            "How to set output start height. "
            "'fixed': use --dst-start-height. "
            "'from_ground_estimate': estimate dst ground_z from retargeted FK first-frames and use NEGATIVE ground_z as start height."
        ),
    )
    p.add_argument(
        "--dst-start-height",
        type=float,
        default=0.28,
        help="Output topology start height (meters) for root trajectory integration.",
    )
    p.set_defaults(save_src_debug=True)
    return p.parse_args()


def _load_corrector(
    ckpt_path: str,
    device: torch.device,
    motion_dim: int,
    joint_dim: int,
) -> MotionCorrector:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = MotionCorrector(
        motion_dim=int(motion_dim),
        joint_dim=int(joint_dim),
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        num_blocks=int(cfg.get("num_blocks", 4)),
        kernel_size=int(cfg.get("kernel_size", 5)),
        dropout=float(cfg.get("dropout", 0.1)),
        joint_delta_max=float(cfg.get("joint_delta_max", 0.35)),
        linvel_delta_max=float(cfg.get("linvel_delta_max", 0.30)),
        yaw_delta_max=float(cfg.get("yaw_delta_max", 0.80)),
    ).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    cli = parse_args()
    output_root = Path(cli.output_root).expanduser().resolve()
    resolved = resolve_task_config(output_root, cli.task_family, cli.pair_id)

    src_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.src_robot}.yaml")
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.dst_robot}.yaml")

    teacher_args = load_teacher_args(cli.teacher_dir)
    teacher_args["is_train"] = False
    teacher_args["save_dir"] = str(Path(cli.teacher_dir).expanduser().resolve())
    teacher_args["batch_size"] = 1

    # Pair/task-dependent fields should come from pair config, not the saved run.
    corr_bodies, corr_joints = _to_legacy_correspondence(resolved)
    if len(corr_bodies) == 0 or len(corr_joints) == 0:
        raise ValueError("Pair correspondences are empty; cannot run inference")

    teacher_args.update(resolved.loss_weights)
    teacher_args["src_njoints"] = int(src_robot.njoints)
    teacher_args["dst_njoints"] = int(dst_robot.njoints)
    teacher_args["src_nbodies"] = int(src_robot.nbodies)
    teacher_args["dst_nbodies"] = int(dst_robot.nbodies)
    teacher_args["correspondence_bodies"] = corr_bodies
    teacher_args["correspondence_joints"] = corr_joints
    teacher_args["src_end"] = list(resolved.src_feet_indices)
    teacher_args["dst_end"] = list(resolved.dst_feet_indices)
    teacher_args["src_ee"] = list(resolved.src_ee_indices)
    teacher_args["dst_ee"] = list(resolved.dst_ee_indices)
    teacher_args["src_fk_path"] = str((output_root / src_robot.fk_xml).resolve() if not src_robot.fk_xml.is_absolute() else src_robot.fk_xml)
    teacher_args["dst_fk_path"] = str((output_root / dst_robot.fk_xml).resolve() if not dst_robot.fk_xml.is_absolute() else dst_robot.fk_xml)
    teacher_args["src_xml_path"] = str((output_root / src_robot.source_xml).resolve() if not src_robot.source_xml.is_absolute() else src_robot.source_xml)
    teacher_args["dst_xml_path"] = str((output_root / dst_robot.source_xml).resolve() if not dst_robot.source_xml.is_absolute() else dst_robot.source_xml)
    teacher_args["src_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["src_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dst_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dst_joint_limits_upper"] = list(dst_robot.joint_limit_upper)
    teacher_args["hum_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    teacher_args["hum_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    teacher_args["dog_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    teacher_args["dog_joint_limits_upper"] = list(dst_robot.joint_limit_upper)

    dataset_roots: list[Path] = []
    if cli.processed_dir is not None:
        dataset_roots.append(Path(cli.processed_dir).expanduser().resolve())
    else:
        dataset_roots.append((output_root / "data" / "processed").resolve())
    strict_roots = cli.processed_dir is not None

    src_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("srcstats_path"),
        robot_id=resolved.src_robot,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    dst_stats_path = _resolve_existing_path_or_search(
        provided_path=teacher_args.get("dststats_path"),
        robot_id=resolved.dst_robot,
        kind="stats",
        roots=dataset_roots,
        strict_roots=strict_roots,
    )
    if src_stats_path is None or dst_stats_path is None:
        raise FileNotFoundError(
            "Could not resolve stats paths for inference. "
            "Provide --processed-dir with correct refactor processed files."
        )

    src_stats = InferenceStats(src_stats_path, njoints=src_robot.njoints, nbodies=src_robot.nbodies)
    dst_stats = InferenceStats(dst_stats_path, njoints=dst_robot.njoints, nbodies=dst_robot.nbodies)
    datasets = [src_stats, dst_stats]

    args = dict_to_object(teacher_args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(cli.device, str) and "cuda" in cli.device:
        if ":" in cli.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = cli.device.split(":")[-1]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in str(cli.device) else "cpu")

    body_src_key = "src_bodies" if "src_bodies" in args.correspondence_bodies[0] else "hum_bodies"
    body_dst_key = "dst_bodies" if "dst_bodies" in args.correspondence_bodies[0] else "dog_bodies"
    joint_src_key = "src_joints" if "src_joints" in args.correspondence_joints[0] else "hum_joints"
    joint_dst_key = "dst_joints" if "dst_joints" in args.correspondence_joints[0] else "dog_joints"
    src_bodies = get_body_part(args.correspondence_bodies, body_src_key)
    dst_bodies = get_body_part(args.correspondence_bodies, body_dst_key)
    src_joints = get_body_part(args.correspondence_joints, joint_src_key)
    dst_joints = get_body_part(args.correspondence_joints, joint_dst_key)
    body_parts = [src_bodies, dst_bodies]
    joint_parts = [src_joints, dst_joints]

    model = create_model(args, body_parts, joint_parts, datasets, ["src", "dst"])
    model.load(epoch=cli.teacher_epoch)
    model.eval()

    src_idx = 1 if cli.reverse else 0
    dst_idx = 0 if cli.reverse else 1
    src_name = resolved.dst_robot if cli.reverse else resolved.src_robot
    dst_name = resolved.src_robot if cli.reverse else resolved.dst_robot
    src_stats_active = datasets[src_idx]
    dst_stats_active = datasets[dst_idx]
    corrector = None
    if cli.corrector_ckpt:
        corrector = _load_corrector(
            ckpt_path=str(Path(cli.corrector_ckpt).expanduser().resolve()),
            device=args.device,
            motion_dim=int(dst_stats_active.njoints + 4),
            joint_dim=int(dst_stats_active.njoints),
        )

    motion_pkl = _load_motion_pkl(cli.input_pkl)
    src_motion, yaw_init, fps, src_start_height = _prepare_src_input(
        motion_pkl=motion_pkl,
        src_stats=src_stats_active,
        device=args.device,
        max_frames=int(cli.max_frames),
    )
    t_len = int(src_motion.shape[1])

    with torch.no_grad():
        src_model = model.models[src_idx]
        dst_model = model.models[dst_idx]

        src_offsets = torch.tensor(src_stats_active.offsets, dtype=torch.float32, device=args.device).reshape(1, -1)
        dst_offsets = torch.tensor(dst_stats_active.offsets, dtype=torch.float32, device=args.device).reshape(1, -1)
        src_skel = src_model.skel_enc(src_offsets).unsqueeze(-1)
        dst_skel = dst_model.skel_enc(dst_offsets).unsqueeze(-1)

        src_motion_t = src_motion.transpose(1, 2)  # [1, C, T]
        ae_out = src_model.ae(src_motion_t, src_skel)
        if src_model.ae.use_vae:
            _, mu, _, rec_motion = ae_out
            retar_latent = mu
        else:
            retar_latent, rec_motion = ae_out
        retar_motion = dst_model.ae.dec(retar_latent, dst_skel)  # [1, C_dst, T]
        fake_retar_input = dst_model.ae.outformat2input(retar_motion)
        cyc_latent, _, _ = dst_model.ae.encode(fake_retar_input)
        cyc_motion = src_model.ae.dec(cyc_latent, src_skel)

        rec_denorm = src_stats_active.denorm(rec_motion, transpose=False).squeeze(0).detach().cpu().numpy()
        cyc_denorm = src_stats_active.denorm(cyc_motion, transpose=False).squeeze(0).detach().cpu().numpy()

        retar_denorm_t = dst_stats_active.denorm(retar_motion, transpose=False)  # [1,T,C]
        if corrector is not None:
            corrected_denorm_t, delta_t = corrector(retar_denorm_t)
            retar_denorm = corrected_denorm_t.squeeze(0).detach().cpu().numpy()
            corr_delta_abs = float(delta_t.abs().mean().detach().cpu().item())
        else:
            retar_denorm = retar_denorm_t.squeeze(0).detach().cpu().numpy()
            corr_delta_abs = 0.0

    dst_start_height_used = float(cli.dst_start_height)
    if str(cli.dst_start_height_mode).lower() == "from_ground_estimate":
        with torch.no_grad():
            retar_denorm_for_ground = torch.tensor(retar_denorm, dtype=torch.float32, device=args.device).unsqueeze(0)
            dst_pos_for_ground, _ = model.models[dst_idx].fk.forward(retar_denorm_for_ground)
            _, dst_ground_z_for_start = estimate_contact_from_height(
                dst_pos_for_ground,
                list(resolved.dst_feet_indices if not cli.reverse else resolved.src_feet_indices),
                ground_margin=float(cli.ground_margin),
                ground_mode="first_frames",
                fixed_ground_z=0.0,
                ref_frames=max(1, int(cli.physics_ref_frames)),
                smooth_steps=1,
            )
        dst_start_height_used = float(dst_ground_z_for_start.squeeze().detach().cpu().item())

    output_pkl = _motion_to_pkl(
        motion_denorm=retar_denorm,
        dst_stats=dst_stats_active,
        yaw_init=float(yaw_init),
        fps=float(fps),
        start_height=dst_start_height_used,
    )
    _save_motion_pkl(cli.output_pkl, output_pkl)

    if cli.save_contact_debug:
        out_path = Path(cli.output_pkl).expanduser().resolve()
        if cli.contact_debug_prefix is None:
            prefix = out_path.with_name(f"{out_path.stem}_contact_debug")
        else:
            prefix = Path(cli.contact_debug_prefix).expanduser().resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            src_motion_denorm_t = src_stats_active.denorm(src_motion, transpose=False)
            src_pos, _ = model.models[src_idx].fk.forward(src_motion_denorm_t)

            if corrector is not None:
                retar_denorm_t = torch.tensor(retar_denorm, dtype=torch.float32, device=args.device).unsqueeze(0)
            else:
                retar_denorm_t = torch.tensor(retar_denorm, dtype=torch.float32, device=args.device).unsqueeze(0)
            dst_pos, _ = model.models[dst_idx].fk.forward(retar_denorm_t)

            dst_contact, dst_ground_z = estimate_contact_from_height(
                dst_pos,
                list(resolved.dst_feet_indices if not cli.reverse else resolved.src_feet_indices),
                ground_margin=float(cli.ground_margin),
                ground_mode="first_frames",
                fixed_ground_z=0.0,
                ref_frames=max(1, int(cli.physics_ref_frames)),
                smooth_steps=1,
            )
            src_contact, _ = estimate_contact_from_height(
                src_pos,
                list(resolved.src_feet_indices if not cli.reverse else resolved.dst_feet_indices),
                ground_margin=float(cli.ground_margin),
                ground_mode="first_frames",
                fixed_ground_z=0.0,
                ref_frames=max(1, int(cli.physics_ref_frames)),
                smooth_steps=1,
            )
            # Robust time alignment for debug mode: teacher paths can differ by 1 frame.
            if src_contact.shape[1] != dst_contact.shape[1]:
                t_common = min(int(src_contact.shape[1]), int(dst_contact.shape[1]))
                src_contact = src_contact[:, :t_common, :]
                dst_contact = dst_contact[:, :t_common, :]
            src_time_gate = torch.max(src_contact, dim=-1, keepdim=True).values
            gated_contact = dst_contact * src_time_gate

        src_contact_np = src_contact.squeeze(0).detach().cpu().numpy()
        dst_contact_np = dst_contact.squeeze(0).detach().cpu().numpy()
        src_gate_np = src_time_gate.squeeze(0).detach().cpu().numpy()
        gated_np = gated_contact.squeeze(0).detach().cpu().numpy()
        dst_ground_np = dst_ground_z.squeeze().detach().cpu().numpy()
        src_foot_idx = list(resolved.src_feet_indices if not cli.reverse else resolved.dst_feet_indices)
        dst_foot_idx = list(resolved.dst_feet_indices if not cli.reverse else resolved.src_feet_indices)
        src_foot_xyz_np = src_pos[:, :, src_foot_idx, :].squeeze(0).detach().cpu().numpy()  # [T, n_src_feet, 3]
        dst_foot_xyz_np = dst_pos[:, :, dst_foot_idx, :].squeeze(0).detach().cpu().numpy()  # [T, n_dst_feet, 3]
        src_foot_z_np = src_foot_xyz_np[..., 2]
        dst_foot_z_np = dst_foot_xyz_np[..., 2]

        np.savez_compressed(
            str(prefix) + ".npz",
            src_contact=src_contact_np,
            dst_contact=dst_contact_np,
            src_time_gate=src_gate_np,
            gated_contact=gated_np,
            dst_ground_z=dst_ground_np,
            src_foot_xyz=src_foot_xyz_np,
            dst_foot_xyz=dst_foot_xyz_np,
            src_foot_z=src_foot_z_np,
            dst_foot_z=dst_foot_z_np,
            src_feet_indices=np.asarray(src_foot_idx, dtype=np.int32),
            dst_feet_indices=np.asarray(dst_foot_idx, dtype=np.int32),
            reverse=np.asarray([int(cli.reverse)], dtype=np.int32),
        )

        t = np.arange(min(src_contact_np.shape[0], dst_contact_np.shape[0], gated_np.shape[0]), dtype=np.int32)
        src_contact_np = src_contact_np[: len(t)]
        dst_contact_np = dst_contact_np[: len(t)]
        src_gate_np = src_gate_np[: len(t)]
        gated_np = gated_np[: len(t)]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for k in range(src_contact_np.shape[1]):
            axes[0].plot(t, src_contact_np[:, k], lw=1.2, label=f"src_f{k}")
        axes[0].set_ylabel("src contact")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)

        for k in range(dst_contact_np.shape[1]):
            axes[1].plot(t, dst_contact_np[:, k], lw=1.2, label=f"dst_f{k}")
        axes[1].set_ylabel("dst contact")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="upper right", ncol=4, fontsize=8)

        axes[2].plot(t, src_gate_np[:, 0], color="black", lw=1.6, label="src_time_gate")
        axes[2].set_ylabel("src gate")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc="upper right", fontsize=8)

        for k in range(gated_np.shape[1]):
            axes[3].plot(t, gated_np[:, k], lw=1.2, label=f"gated_dst_f{k}")
        axes[3].set_ylabel("gated contact")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_xlabel("frame")
        axes[3].legend(loc="upper right", ncol=4, fontsize=8)

        fig.tight_layout()
        fig.savefig(str(prefix) + ".png", dpi=160)
        plt.close(fig)

        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        for k in range(src_foot_z_np.shape[1]):
            axes2[0].plot(t, src_foot_z_np[: len(t), k], lw=1.2, label=f"src_foot{src_foot_idx[k]}_z")
        axes2[0].set_ylabel("src foot z (m)")
        axes2[0].legend(loc="upper right", ncol=2, fontsize=8)

        for k in range(dst_foot_z_np.shape[1]):
            axes2[1].plot(t, dst_foot_z_np[: len(t), k], lw=1.2, label=f"dst_foot{dst_foot_idx[k]}_z")
        if np.isscalar(dst_ground_np):
            axes2[1].axhline(float(dst_ground_np), color="black", ls="--", lw=1.2, label="dst_ground_z")
        else:
            axes2[1].plot(t, np.full_like(t, float(np.asarray(dst_ground_np).reshape(-1)[0]), dtype=np.float32),
                          color="black", ls="--", lw=1.2, label="dst_ground_z")
        axes2[1].set_ylabel("dst foot z (m)")
        axes2[1].set_xlabel("frame")
        axes2[1].legend(loc="upper right", ncol=2, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(str(prefix) + "_z.png", dpi=160)
        plt.close(fig2)

        dims = ["x", "y", "z"]
        fig3, axes3 = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
        for d, dname in enumerate(dims):
            for k in range(src_foot_xyz_np.shape[1]):
                axes3[d, 0].plot(t, src_foot_xyz_np[: len(t), k, d], lw=1.2, label=f"src_foot{src_foot_idx[k]}_{dname}")
            axes3[d, 0].set_ylabel(f"src {dname} (m)")
            axes3[d, 0].legend(loc="upper right", ncol=2, fontsize=8)

            for k in range(dst_foot_xyz_np.shape[1]):
                axes3[d, 1].plot(t, dst_foot_xyz_np[: len(t), k, d], lw=1.2, label=f"dst_foot{dst_foot_idx[k]}_{dname}")
            axes3[d, 1].set_ylabel(f"dst {dname} (m)")
            axes3[d, 1].legend(loc="upper right", ncol=2, fontsize=8)

        axes3[2, 0].set_xlabel("frame")
        axes3[2, 1].set_xlabel("frame")
        fig3.tight_layout()
        fig3.savefig(str(prefix) + "_xyz.png", dpi=160)
        plt.close(fig3)

    src_rec_path = cli.output_src_rec_pkl
    src_cyc_path = cli.output_src_cyc_pkl
    if cli.save_src_debug:
        out_path = Path(cli.output_pkl).expanduser().resolve()
        if src_rec_path is None:
            src_rec_path = str(out_path.with_name(f"{out_path.stem}_src_rec.pkl"))
        if src_cyc_path is None:
            src_cyc_path = str(out_path.with_name(f"{out_path.stem}_src_cyc.pkl"))

        src_rec_pkl = _motion_to_pkl(
            motion_denorm=rec_denorm,
            dst_stats=src_stats_active,
            yaw_init=float(yaw_init),
            fps=float(fps),
            start_height=float(src_start_height),
        )
        src_cyc_pkl = _motion_to_pkl(
            motion_denorm=cyc_denorm,
            dst_stats=src_stats_active,
            yaw_init=float(yaw_init),
            fps=float(fps),
            start_height=float(src_start_height),
        )
        _save_motion_pkl(src_rec_path, src_rec_pkl)
        _save_motion_pkl(src_cyc_path, src_cyc_pkl)

    print("Done.")
    print(f"  pair: {src_name} -> {dst_name} ({resolved.task_family}/{resolved.pair_id})")
    print(f"  direction: {'reverse' if cli.reverse else 'forward'}")
    print(f"  input:  {Path(cli.input_pkl).expanduser().resolve()}")
    print(f"  output: {Path(cli.output_pkl).expanduser().resolve()}")
    if cli.save_src_debug:
        print(f"  src rec: {Path(src_rec_path).expanduser().resolve()}")
        print(f"  src cyc: {Path(src_cyc_path).expanduser().resolve()}")
    if cli.corrector_ckpt:
        print(f"  corrector: {Path(cli.corrector_ckpt).expanduser().resolve()}")
        print(f"  corrector mean|delta|: {corr_delta_abs:.6f}")
    if cli.save_contact_debug:
        print(f"  contact debug npz: {str(prefix) + '.npz'}")
        print(f"  contact debug png: {str(prefix) + '.png'}")
        print(f"  contact debug z png: {str(prefix) + '_z.png'}")
        print(f"  contact debug xyz png: {str(prefix) + '_xyz.png'}")
    print(f"  dst start height mode: {cli.dst_start_height_mode}")
    print(f"  dst start height used: {dst_start_height_used:.6f}")
    print(f"  frames: {t_len}")
    print(f"  dims: src={src_stats_active.njoints + 4} dst={dst_stats_active.njoints + 4}")


if __name__ == "__main__":
    main()
