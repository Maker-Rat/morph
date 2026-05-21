from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _as_float_fps(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fps = float(value)
    except Exception:
        return None
    if fps <= 0:
        return None
    return fps


def _normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    q /= n
    return q


def _enforce_quat_continuity(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quat_array(quat)
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
    return q


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        out /= max(np.linalg.norm(out), 1e-12)
        return out

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    if abs(sin_theta_0) < 1e-12:
        return q0

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = s0 * q0 + s1 * q1
    out /= max(np.linalg.norm(out), 1e-12)
    return out


def _resample_quat_sequence(quat: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    q = _enforce_quat_continuity(quat)
    out = np.zeros((len(dst_t), 4), dtype=np.float64)
    n_src = len(src_t)

    if n_src == 1:
        out[:] = q[0]
        return out.astype(np.float32)

    for i, t in enumerate(dst_t):
        if t <= src_t[0]:
            out[i] = q[0]
            continue
        if t >= src_t[-1]:
            out[i] = q[-1]
            continue

        j = int(np.searchsorted(src_t, t, side="right") - 1)
        j = max(0, min(j, n_src - 2))
        t0, t1 = src_t[j], src_t[j + 1]
        a = 0.0 if t1 <= t0 else float((t - t0) / (t1 - t0))
        out[i] = _slerp(q[j], q[j + 1], a)
    return out.astype(np.float32)


def _resample_linear(arr: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.shape[0] == 1:
        rep = np.repeat(x, len(dst_t), axis=0)
        return rep.astype(np.float32)

    flat = x.reshape(x.shape[0], -1)
    out_flat = np.zeros((len(dst_t), flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out_flat[:, c] = np.interp(dst_t, src_t, flat[:, c])
    return out_flat.reshape((len(dst_t),) + x.shape[1:]).astype(np.float32)


def _build_time_axes(n_frames: int, src_fps: float, dst_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    if n_frames <= 1:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    duration = (n_frames - 1) / src_fps
    n_out = int(np.round(duration * dst_fps)) + 1
    n_out = max(n_out, 2)
    src_t = np.arange(n_frames, dtype=np.float64) / src_fps
    dst_t = np.arange(n_out, dtype=np.float64) / dst_fps
    return src_t, dst_t


def _load_motion(pkl_path: Path) -> Any:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def _save_motion(pkl_path: Path, payload: Any) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)


def _resample_dict_payload(data: Dict[str, Any], src_fps: float, dst_fps: float, quat_convention: str) -> Dict[str, Any]:
    if "dof_pos" not in data:
        raise ValueError("dict payload missing required key 'dof_pos'")

    dof = np.asarray(data["dof_pos"], dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"dof_pos must be [T, J], got {dof.shape}")

    n_frames = int(dof.shape[0])
    src_t, dst_t = _build_time_axes(n_frames, src_fps, dst_fps)

    out = dict(data)
    out["dof_pos"] = _resample_linear(dof, src_t, dst_t)

    if "root_pos" in data:
        out["root_pos"] = _resample_linear(np.asarray(data["root_pos"], dtype=np.float32), src_t, dst_t)
    for quat_key in ("root_rot", "root_heading_rot"):
        if quat_key in data:
            quat = np.asarray(data[quat_key], dtype=np.float32)
            if quat.shape[0] != n_frames or quat.shape[-1] != 4:
                raise ValueError(f"{quat_key} must be [T,4], got {quat.shape}")
            if quat_convention == "wxyz":
                rot_xyzw = np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=-1)
                rot_xyzw_r = _resample_quat_sequence(rot_xyzw, src_t, dst_t)
                out[quat_key] = np.stack(
                    [rot_xyzw_r[:, 3], rot_xyzw_r[:, 0], rot_xyzw_r[:, 1], rot_xyzw_r[:, 2]], axis=-1
                ).astype(np.float32)
            else:
                out[quat_key] = _resample_quat_sequence(quat, src_t, dst_t)

    for k, v in data.items():
        if k in ("dof_pos", "root_pos", "root_rot", "root_heading_rot", "fps"):
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_frames:
            out[k] = _resample_linear(v, src_t, dst_t)

    out["fps"] = float(dst_fps)
    return out


def _resample_list_payload(data: List[Any], src_fps: float, dst_fps: float, quat_convention: str) -> List[Any]:
    if len(data) == 0:
        return []

    try:
        root_pos = np.asarray([f[0] for f in data], dtype=np.float32)
        root_rot = np.asarray([f[1] for f in data], dtype=np.float32)
        dof_pos = np.asarray([f[2] for f in data], dtype=np.float32)
        has_heading_rot = len(data[0]) >= 4
        heading_rot = np.asarray([f[3] for f in data], dtype=np.float32) if has_heading_rot else None
    except Exception as exc:
        raise ValueError("list payload must contain frames like (root_pos, root_rot, dof_pos)") from exc

    n_frames = int(dof_pos.shape[0])
    src_t, dst_t = _build_time_axes(n_frames, src_fps, dst_fps)

    root_pos_r = _resample_linear(root_pos, src_t, dst_t)
    dof_pos_r = _resample_linear(dof_pos, src_t, dst_t)
    def resample_quat(quat: np.ndarray) -> np.ndarray:
        if quat_convention == "wxyz":
            rot_xyzw = np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=-1)
            rot_xyzw_r = _resample_quat_sequence(rot_xyzw, src_t, dst_t)
            return np.stack([rot_xyzw_r[:, 3], rot_xyzw_r[:, 0], rot_xyzw_r[:, 1], rot_xyzw_r[:, 2]], axis=-1)
        return _resample_quat_sequence(quat, src_t, dst_t)

    root_rot_r = resample_quat(root_rot)
    heading_rot_r = resample_quat(heading_rot) if heading_rot is not None else None

    if heading_rot_r is not None:
        return [(root_pos_r[i], root_rot_r[i], dof_pos_r[i], heading_rot_r[i]) for i in range(len(dst_t))]
    return [(root_pos_r[i], root_rot_r[i], dof_pos_r[i]) for i in range(len(dst_t))]


def _resolve_src_fps(payload: Any, src_fps_arg: Optional[float]) -> float:
    if isinstance(payload, dict):
        fps = _as_float_fps(payload.get("fps", None))
        if fps is not None:
            return fps
    if src_fps_arg is not None:
        return float(src_fps_arg)
    raise ValueError("Could not resolve source FPS. Add fps in PKL or pass --src-fps.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample all motion PKLs from source FPS to target FPS.")
    p.add_argument("--input-dir", type=str, required=True, help="Root directory containing source PKLs.")
    p.add_argument("--output-dir", type=str, required=True, help="Output root directory for resampled PKLs.")
    p.add_argument("--target-fps", type=float, required=True, help="Target FPS (e.g., 30).")
    p.add_argument("--src-fps", type=float, default=None,
                   help="Fallback source FPS when PKL has no fps field.")
    p.add_argument("--quat-convention", choices=["xyzw", "wxyz"], default="xyzw")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--dry-run", action="store_true", help="Only print planned operations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    target_fps = float(args.target_fps)

    if target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if not in_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_root}")

    pkl_files = sorted(in_root.rglob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"No PKL files found under: {in_root}")
        return

    print("Resampling dataset:")
    print(f"  input:  {in_root}")
    print(f"  output: {out_root}")
    print(f"  target fps: {target_fps}")
    print(f"  files: {len(pkl_files)}")
    if args.src_fps is not None:
        print(f"  fallback src fps: {float(args.src_fps)}")
    print(f"  mode: {'dry-run' if args.dry_run else 'write'}")

    ok = 0
    skipped = 0
    failed = 0

    for i, src_path in enumerate(pkl_files, start=1):
        rel = src_path.relative_to(in_root)
        dst_path = out_root / rel

        if dst_path.exists() and (not args.overwrite):
            skipped += 1
            print(f"[{i}/{len(pkl_files)}] skip exists: {dst_path}")
            continue

        try:
            payload = _load_motion(src_path)
            src_fps = _resolve_src_fps(payload, args.src_fps)

            if isinstance(payload, dict):
                out = _resample_dict_payload(payload, src_fps, target_fps, args.quat_convention)
                n_in = int(np.asarray(payload["dof_pos"]).shape[0])
                n_out = int(np.asarray(out["dof_pos"]).shape[0])
            elif isinstance(payload, list):
                out = _resample_list_payload(payload, src_fps, target_fps, args.quat_convention)
                n_in = len(payload)
                n_out = len(out)
            else:
                raise ValueError(f"Unsupported PKL type: {type(payload)}")

            print(f"[{i}/{len(pkl_files)}] {rel} | {src_fps:.4g} -> {target_fps:.4g} fps | {n_in} -> {n_out} frames")
            if not args.dry_run:
                _save_motion(dst_path, out)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(pkl_files)}] FAILED {rel}: {exc}")

    print("\nDone.")
    print(f"  converted: {ok}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")


if __name__ == "__main__":
    main()

