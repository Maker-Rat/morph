from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import load_pair_config, resolve_task_config


def _check_indices(indices: Iterable[int], upper_bound: int) -> list[int]:
    bad = []
    for idx in indices:
        if not isinstance(idx, int) or idx < 0 or idx >= upper_bound:
            bad.append(idx)
    return bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print and validate merged task+pair config.")
    p.add_argument("--output-root", default=".", help="Root containing configs/")
    p.add_argument("--task-family", required=True, help="Task family name")
    p.add_argument("--pair-id", required=True, help="Pair id (yaml filename without .yaml)")
    p.add_argument("--strict", action="store_true", help="Exit with non-zero status on validation errors")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()

    pair = load_pair_config(root, args.task_family, args.pair_id)
    resolved = resolve_task_config(root, args.task_family, args.pair_id)

    src_robot_cfg = root / "configs" / "robots" / f"{pair.src_robot}.yaml"
    dst_robot_cfg = root / "configs" / "robots" / f"{pair.dst_robot}.yaml"
    src = load_robot_spec(src_robot_cfg)
    dst = load_robot_spec(dst_robot_cfg)

    print("Resolved task config")
    print(f"  task_family: {resolved.task_family}")
    print(f"  pair_id: {resolved.pair_id}")
    print(f"  src_robot: {resolved.src_robot} (njoints={src.njoints}, nbodies={src.nbodies})")
    print(f"  dst_robot: {resolved.dst_robot} (njoints={dst.njoints}, nbodies={dst.nbodies})")
    print(f"  src_feet: {resolved.src_feet_indices}")
    print(f"  dst_feet: {resolved.dst_feet_indices}")
    print(f"  src_ee: {resolved.src_ee_indices}")
    print(f"  dst_ee: {resolved.dst_ee_indices}")
    print(f"  loss_weights: {resolved.loss_weights}")
    print(f"  body_groups: {len(resolved.correspondence_body_groups)}")
    print(f"  joint_groups: {len(resolved.correspondence_joint_groups)}")

    errors: list[str] = []

    bad = _check_indices(resolved.src_feet_indices, src.nbodies)
    if bad:
        errors.append(f"indices.src_feet invalid: {bad}")
    bad = _check_indices(resolved.dst_feet_indices, dst.nbodies)
    if bad:
        errors.append(f"indices.dst_feet invalid: {bad}")
    bad = _check_indices(resolved.src_ee_indices, src.nbodies)
    if bad:
        errors.append(f"indices.src_ee invalid: {bad}")
    bad = _check_indices(resolved.dst_ee_indices, dst.nbodies)
    if bad:
        errors.append(f"indices.dst_ee invalid: {bad}")

    for i, bg in enumerate(resolved.correspondence_body_groups):
        sidx = list(bg.get("src_body_indices", []))
        didx = list(bg.get("dst_body_indices", []))
        bad_s = _check_indices(sidx, src.nbodies)
        bad_d = _check_indices(didx, dst.nbodies)
        if bad_s:
            errors.append(f"body_groups[{i}] src_body_indices invalid: {bad_s}")
        if bad_d:
            errors.append(f"body_groups[{i}] dst_body_indices invalid: {bad_d}")

    for i, jg in enumerate(resolved.correspondence_joint_groups):
        sidx = list(jg.get("src_joint_indices", []))
        didx = list(jg.get("dst_joint_indices", []))
        bad_s = _check_indices(sidx, src.njoints)
        bad_d = _check_indices(didx, dst.njoints)
        if bad_s:
            errors.append(f"joint_groups[{i}] src_joint_indices invalid: {bad_s}")
        if bad_d:
            errors.append(f"joint_groups[{i}] dst_joint_indices invalid: {bad_d}")

    if errors:
        print("\nValidation issues:")
        for err in errors:
            print(f"  - {err}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nValidation: OK")

    print("\nPair-level correspondences:")
    for i, bg in enumerate(resolved.correspondence_body_groups):
        print(f"  body_group[{i}]: {bg}")
    for i, jg in enumerate(resolved.correspondence_joint_groups):
        print(f"  joint_group[{i}]: {jg}")


if __name__ == "__main__":
    main()
