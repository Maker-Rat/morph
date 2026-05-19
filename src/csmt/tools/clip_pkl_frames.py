from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _clip_dict_motion(data: dict, start: int, n: int) -> tuple[dict, int]:
    out = dict(data)

    # Common sequence keys used across this project.
    sequence_keys = [
        "dof_pos",
        "joint_pos",
        "joint_positions",
        "root_pos",
        "base_trans",
        "base_translation",
        "root_rot",
        "base_quat",
        "base_rotation",
        "local_body_pos",
        "link_body_pos",
        "link_body_list",
        "root_vel",
        "dof_vel",
    ]

    original_len = None
    for k in sequence_keys:
        v = out.get(k, None)
        if v is None:
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim >= 1 and arr.shape[0] > 0:
            if original_len is None:
                original_len = int(arr.shape[0])
            out[k] = arr[start:start + n].copy()

    if original_len is None:
        # Fallback: first ndarray/list-like field with frame axis.
        for k, v in out.items():
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.ndim >= 1 and arr.shape[0] > 0:
                original_len = int(arr.shape[0])
                out[k] = arr[start:start + n].copy()
                break

    if original_len is None:
        raise ValueError("Could not find any frame-like sequence in dict PKL.")

    return out, original_len


def _clip_list_motion(data: list, start: int, n: int) -> tuple[list, int]:
    return data[start:start + n], len(data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clip a motion PKL to a fixed number of frames.")
    p.add_argument("--input-pkl", type=str, required=True)
    p.add_argument("--output-pkl", type=str, required=True)
    p.add_argument("--max-frames", type=int, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input_pkl).expanduser().resolve()
    out = Path(args.output_pkl).expanduser().resolve()

    if int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be > 0")
    if int(args.start_frame) < 0:
        raise ValueError("--start-frame must be >= 0")
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")
    if out.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output exists: {out} (pass --allow-overwrite)")

    payload = _load_pkl(inp)
    n = int(args.max_frames)
    start = int(args.start_frame)

    if isinstance(payload, dict):
        clipped, old_len = _clip_dict_motion(payload, start, n)
    elif isinstance(payload, list):
        clipped, old_len = _clip_list_motion(payload, start, n)
    else:
        raise ValueError(f"Unsupported PKL type: {type(payload)}. Expected dict or list.")

    if start >= old_len:
        new_len = 0
    else:
        new_len = min(old_len - start, n)
    _save_pkl(out, clipped)

    print("Done.")
    print(f"  input:  {inp}")
    print(f"  output: {out}")
    print(f"  slice:  [{start}:{start + n}]")
    print(f"  frames: {old_len} -> {new_len}")


if __name__ == "__main__":
    main()
