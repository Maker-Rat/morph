from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_robot_yaml(config_root: Path, robot_id: str) -> dict:
    p = config_root / "configs" / "robots" / f"{robot_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Robot config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_pair_id(src_robot: str, dst_robot: str) -> str:
    return f"{src_robot}_to_{dst_robot}"


class _FlowNumListDumper(yaml.SafeDumper):
    pass


def _is_numeric_list(value) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True
    for item in value:
        if isinstance(item, bool):
            return False
        if isinstance(item, (int, float)):
            continue
        if isinstance(item, list) and _is_numeric_list(item):
            continue
        return False
    return True


def _represent_list(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        data,
        flow_style=_is_numeric_list(data),
    )


_FlowNumListDumper.add_representer(list, _represent_list)


def write_pair_yaml(
    *,
    config_root: Path,
    task_family: str,
    src_robot: str,
    dst_robot: str,
    pair_id: str,
    overwrite: bool,
) -> Path:
    _load_robot_yaml(config_root, src_robot)
    _load_robot_yaml(config_root, dst_robot)

    out_path = config_root / "configs" / "tasks" / task_family / "pairs" / f"{pair_id}.yaml"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Pair config already exists: {out_path}")

    data = {
        "pair_id": pair_id,
        "task_family": task_family,
        "src_robot": src_robot,
        "dst_robot": dst_robot,
        "indices": {
            "src_feet": [],
            "dst_feet": [],
            "src_ee": [],
            "dst_ee": [],
        },
        "correspondence": {
            "body_groups": [
                {
                    "name": "",
                    "src_body_indices": [],
                    "dst_body_indices": [],
                }
            ],
            "joint_groups": [
                {
                    "name": "",
                    "src_joint_indices": [],
                    "dst_joint_indices": [],
                }
            ],
        },
        "loss_overrides": {},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, Dumper=_FlowNumListDumper)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap a task pair YAML scaffold.")
    p.add_argument("--output-root", default=".", help="Root containing configs/")
    p.add_argument("--task-family", required=True, help="Task family name (e.g., locomotion, manipulation)")
    p.add_argument("--src-robot", required=True, help="Source robot id (must exist in configs/robots)")
    p.add_argument("--dst-robot", required=True, help="Destination robot id (must exist in configs/robots)")
    p.add_argument("--pair-id", default=None, help="Optional pair id (default: <src>_to_<dst>)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing pair config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    pair_id = args.pair_id or _default_pair_id(args.src_robot, args.dst_robot)
    out_path = write_pair_yaml(
        config_root=root,
        task_family=args.task_family,
        src_robot=args.src_robot,
        dst_robot=args.dst_robot,
        pair_id=pair_id,
        overwrite=args.overwrite,
    )

    print("Bootstrapped task-pair scaffold:")
    print(f"  task_family: {args.task_family}")
    print(f"  pair_id: {pair_id}")
    print(f"  src_robot: {args.src_robot}")
    print(f"  dst_robot: {args.dst_robot}")
    print(f"  config_yaml: {out_path}")


if __name__ == "__main__":
    main()
