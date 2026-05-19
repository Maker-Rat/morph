from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


SMPL_INPUT_DIM = 69
POSE_BODY_DIM = 63
ROOT_LIN_LOCAL_SLICE = slice(63, 66)
ROOT_ANG_LOCAL_SLICE = slice(66, 69)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors [..., 3] to rotation matrices [..., 3, 3]."""
    aa = np.asarray(axis_angle, dtype=np.float64)
    orig_shape = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    mats = R.from_rotvec(flat).as_matrix()
    return mats.reshape(*orig_shape, 3, 3).astype(np.float32)


def _matrix_to_axis_angle(rot_mats: np.ndarray) -> np.ndarray:
    """Convert rotation matrices [..., 3, 3] to axis-angle vectors [..., 3]."""
    r = np.asarray(rot_mats, dtype=np.float64)
    orig_shape = r.shape[:-2]
    flat = r.reshape(-1, 3, 3)
    rotvec = R.from_matrix(flat).as_rotvec()
    return rotvec.reshape(*orig_shape, 3).astype(np.float32)


def _compute_world_linear_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float32)
    n = int(pos.shape[0])
    if n > 2:
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    if n > 1:
        vel[0] = (pos[1] - pos[0]) / dt
        vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def _compute_root_ang_vel_local(root_rot_mats: np.ndarray, dt: float) -> np.ndarray:
    n = int(root_rot_mats.shape[0])
    w_local = np.zeros((n, 3), dtype=np.float32)
    if n <= 1:
        return w_local
    rel = np.matmul(np.transpose(root_rot_mats[:-1], (0, 2, 1)), root_rot_mats[1:])
    d_aa = _matrix_to_axis_angle(rel)
    w_local[1:] = d_aa / dt
    w_local[0] = w_local[1]
    return w_local


def _to_local_linear_vel(world_vel: np.ndarray, root_rot_mats: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tj->ti", np.transpose(root_rot_mats, (0, 2, 1)), world_vel).astype(np.float32)


def load_smpl_motion(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        return payload
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported SMPL file content type at {path}: {type(payload)}")
    return payload


def parse_smpl_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pose_body = np.asarray(payload.get("pose_body"), dtype=np.float32)
    root_orient = np.asarray(
        payload.get("root_orient", payload.get("global_orient", payload.get("root_pose"))),
        dtype=np.float32,
    )
    trans = np.asarray(payload.get("trans", payload.get("translation")), dtype=np.float32)

    if pose_body.ndim != 2 or pose_body.shape[1] != 63:
        raise ValueError(f"Expected pose_body shape [T,63], got {pose_body.shape}")
    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        raise ValueError(f"Expected root_orient shape [T,3], got {root_orient.shape}")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Expected trans shape [T,3], got {trans.shape}")

    t = min(int(pose_body.shape[0]), int(root_orient.shape[0]), int(trans.shape[0]))
    pose_body = pose_body[:t]
    root_orient = root_orient[:t]
    trans = trans[:t]

    fps = float(
        payload.get(
            "fps",
            payload.get("mocap_frame_rate", payload.get("mocap_framerate", 30.0)),
        )
    )
    if fps <= 1e-6:
        fps = 30.0
    return pose_body, root_orient, trans, fps


def build_smpl_frame_features(
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    fps: float,
) -> np.ndarray:
    dt = 1.0 / float(fps)
    root_rot_mats = _axis_angle_to_matrix(root_orient)
    root_lin_vel_world = _compute_world_linear_vel(trans, dt)
    root_lin_vel_local = _to_local_linear_vel(root_lin_vel_world, root_rot_mats)
    root_ang_vel_local = _compute_root_ang_vel_local(root_rot_mats, dt)

    feat = np.concatenate(
        [pose_body, root_lin_vel_local, root_ang_vel_local],
        axis=-1,
    ).astype(np.float32)
    if feat.shape[-1] != SMPL_INPUT_DIM:
        raise ValueError(f"Expected SMPL feature dim {SMPL_INPUT_DIM}, got {feat.shape[-1]}")
    return feat


def root_motion_4d_from_smpl_features(x_smpl: np.ndarray) -> np.ndarray:
    """
    Map SMPL frame features [...,69] -> [...,4] in robot root-motion convention:
      [vx, vy, vz, yaw_rate]

    Layout:
      pose_body:            0:63
      root_lin_vel_local:  63:66
      root_ang_vel_local:  66:69

    Empirical convention alignment for this pipeline:
      robot vx      <- smpl lin_vel_local z (idx 65)
      robot vy      <- smpl lin_vel_local x (idx 63)
      robot vz      <- smpl lin_vel_local y (idx 64)
      robot yawrate <- smpl ang_vel_local y (idx 67)
    """
    x = np.asarray(x_smpl, dtype=np.float32)
    if x.shape[-1] != SMPL_INPUT_DIM:
        raise ValueError(
            f"expected smpl_input_dim={SMPL_INPUT_DIM}, got {x.shape[-1]}; "
            "regenerate distill dataset/checkpoint with 69D SMPL features"
        )
    vx = x[..., 65:66]
    vy = x[..., 63:64]
    vz = x[..., 64:65]
    yaw_rate = x[..., 67:68]
    return np.concatenate([vx, vy, vz, yaw_rate], axis=-1).astype(np.float32)


def root_motion_4d_from_smpl_arrays(
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    fps: float,
    mode: str = "local",
) -> np.ndarray:
    """
    Build robot root motion [vx, vy, vz, yaw_rate] from SMPL arrays.

    mode="local" preserves the legacy mapping:
      robot vx <- SMPL root-local z
      robot vy <- SMPL root-local x
      robot vz <- SMPL root-local y

    mode="world_z" preserves the legacy planar/yaw mapping but takes vertical
    velocity from SMPL world z. This avoids root-local roll/pitch leakage making
    walking/turning clips sink or float while still preserving jumps.
    """
    mode = str(mode).lower()
    feat = build_smpl_frame_features(pose_body, root_orient, trans, fps)
    root4 = root_motion_4d_from_smpl_features(feat)
    if mode == "local":
        return root4
    if mode == "world_z":
        world_vel = _compute_world_linear_vel(np.asarray(trans, dtype=np.float32), 1.0 / float(fps))
        root4 = root4.copy()
        root4[..., 2] = world_vel[..., 2]
        return root4.astype(np.float32)
    raise ValueError(f"Unsupported SMPL root motion map mode: {mode}")


def _resample_linear_track(track: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    Linear interpolation for tracks shaped [T, D].
    """
    d = int(track.shape[1])
    out = np.zeros((len(t_dst), d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(t_dst, t_src, track[:, i]).astype(np.float32)
    return out


def _slerp_axis_angle_track(track: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    SLERP-like interpolation for axis-angle rotation track [T, 3].
    """
    if len(track) == 1:
        return np.repeat(track.astype(np.float32), len(t_dst), axis=0)

    mats = _axis_angle_to_matrix(track.astype(np.float32))  # [T, 3, 3]
    out = np.zeros((len(t_dst), 3), dtype=np.float32)

    idx1_all = np.searchsorted(t_src, t_dst, side="right")
    idx1_all = np.clip(idx1_all, 1, len(t_src) - 1)
    idx0_all = idx1_all - 1

    t0 = t_src[idx0_all]
    t1 = t_src[idx1_all]
    denom = np.maximum(t1 - t0, 1e-8)
    alpha = ((t_dst - t0) / denom).astype(np.float32)

    for i in range(len(t_dst)):
        i0 = int(idx0_all[i])
        i1 = int(idx1_all[i])
        a = float(np.clip(alpha[i], 0.0, 1.0))
        r0 = mats[i0]
        r1 = mats[i1]
        rel = r0.T @ r1
        rel_aa = _matrix_to_axis_angle(rel[None, ...])[0]
        step = _axis_angle_to_matrix((rel_aa * a)[None, ...])[0]
        out_mat = r0 @ step
        out[i] = _matrix_to_axis_angle(out_mat[None, ...])[0]
    return out.astype(np.float32)


def resample_smpl_tracks(
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    src_fps: float,
    dst_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample SMPL tracks from src_fps to dst_fps while preserving duration.
    """
    if pose_body.shape[0] <= 1:
        return pose_body.astype(np.float32), root_orient.astype(np.float32), trans.astype(np.float32)

    src_fps = float(max(src_fps, 1e-6))
    dst_fps = float(max(dst_fps, 1e-6))
    if abs(src_fps - dst_fps) <= 1e-6:
        return pose_body.astype(np.float32), root_orient.astype(np.float32), trans.astype(np.float32)

    t_src = np.arange(pose_body.shape[0], dtype=np.float64) / src_fps
    duration = t_src[-1]
    n_dst = int(round(duration * dst_fps)) + 1
    n_dst = max(2, n_dst)
    t_dst = np.arange(n_dst, dtype=np.float64) / dst_fps
    t_dst = np.clip(t_dst, 0.0, duration)

    trans_rs = _resample_linear_track(trans.astype(np.float32), t_src, t_dst)
    root_rs = _slerp_axis_angle_track(root_orient.astype(np.float32), t_src, t_dst)

    pose = pose_body.astype(np.float32).reshape(pose_body.shape[0], 21, 3)
    pose_rs = np.zeros((len(t_dst), 21, 3), dtype=np.float32)
    for j in range(21):
        pose_rs[:, j, :] = _slerp_axis_angle_track(pose[:, j, :], t_src, t_dst)
    pose_rs = pose_rs.reshape(len(t_dst), 63)
    return pose_rs.astype(np.float32), root_rs.astype(np.float32), trans_rs.astype(np.float32)
