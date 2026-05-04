from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

import yaml

from csmt.pipelines._legacy_pan_train import build_legacy_default_args, run_pan_training
from csmt.robots.registry import load_robot_spec
from csmt.tasks.registry import resolve_task_config


def _parse_value(raw: str):
    lo = raw.lower()
    if lo == "true":
        return True
    if lo == "false":
        return False
    try:
        if "." in raw or "e" in lo:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_set_overrides(args_dict: dict, set_items: list[str]) -> None:
    for item in set_items:
        if "=" not in item:
            raise ValueError(f"Invalid --set entry '{item}', expected key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set entry '{item}', empty key")
        args_dict[key] = _parse_value(raw.strip())


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
            # Backward aliases
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
            # Backward aliases
            "hum_joints": src,
            "dog_joints": dst,
        })

    return body_groups, joint_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refactor teacher training entrypoint.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help=(
            "Directory containing processed NPZ files "
            "(e.g., g1_train.npz, go2_with_arm_stats.npz). "
            "If omitted, uses output-root/data/processed."
        ),
    )
    p.add_argument("--task-family", type=str, required=True)
    p.add_argument("--pair-id", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--model-config", type=str, default=None,
                   help="Optional YAML with training/model defaults; defaults to configs/models/teacher_pan.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epoch-num", type=int, default=None)
    p.add_argument("--epoch-begin", type=int, default=None)
    p.add_argument("--save-iter", type=int, default=None)
    p.add_argument("--print-iter", type=int, default=None)
    p.add_argument("--lr-g", type=float, default=None)
    p.add_argument("--lr-d", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--set", action="append", default=[],
                   help="Arbitrary override: key=value (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    output_root = Path(cli.output_root).expanduser().resolve()

    model_cfg_path = (
        Path(cli.model_config).expanduser().resolve()
        if cli.model_config
        else output_root / "configs" / "models" / "teacher_pan.yaml"
    )

    model_cfg = {}
    if model_cfg_path.exists():
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}

    resolved = resolve_task_config(output_root, cli.task_family, cli.pair_id)
    src_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.src_robot}.yaml")
    dst_robot = load_robot_spec(output_root / "configs" / "robots" / f"{resolved.dst_robot}.yaml")

    def _resolve_path(p: Path) -> str:
        return str(p if p.is_absolute() else (output_root / p).resolve())

    dataset_roots: list[Path] = []
    if cli.processed_dir is not None:
        dataset_roots.append(Path(cli.processed_dir).expanduser().resolve())
    else:
        dataset_roots.append((output_root / "data" / "processed").resolve())

    def _resolve_dataset_path(robot_id: str, split: str, kind: str) -> str | None:
        for proc in dataset_roots:
            candidates = [
                proc / f"{robot_id}_{kind}.npz",
                proc / f"{robot_id}_{split}.npz",
                proc / f"unitree_{robot_id}_{kind}.npz",
                proc / f"unitree_{robot_id}_{split}.npz",
            ]
            for c in candidates:
                if c.exists():
                    return str(c.resolve())
        return None

    corr_bodies, corr_joints = _to_legacy_correspondence(resolved)
    if len(corr_bodies) == 0:
        raise ValueError("No usable body correspondences found for this pair")
    if len(corr_joints) == 0:
        raise ValueError("No usable joint correspondences found for this pair")

    args_dict = build_legacy_default_args()

    # model config yaml may include direct legacy parser keys.
    for k, v in model_cfg.items():
        if isinstance(v, (str, int, float, bool, list, dict)):
            args_dict[k] = v

    # task-family defaults + pair overrides (from resolver)
    args_dict.update(resolved.loss_weights)

    # topology-specific fields.
    args_dict["src_njoints"] = int(src_robot.njoints)
    args_dict["dst_njoints"] = int(dst_robot.njoints)
    args_dict["src_nbodies"] = int(src_robot.nbodies)
    args_dict["dst_nbodies"] = int(dst_robot.nbodies)
    args_dict["correspondence_bodies"] = corr_bodies
    args_dict["correspondence_joints"] = corr_joints
    args_dict["src_end"] = list(resolved.src_feet_indices)
    args_dict["dst_end"] = list(resolved.dst_feet_indices)
    args_dict["src_ee"] = list(resolved.src_ee_indices)
    args_dict["dst_ee"] = list(resolved.dst_ee_indices)
    args_dict["src_fk_path"] = _resolve_path(src_robot.fk_xml)
    args_dict["dst_fk_path"] = _resolve_path(dst_robot.fk_xml)
    args_dict["src_xml_path"] = _resolve_path(src_robot.source_xml)
    args_dict["dst_xml_path"] = _resolve_path(dst_robot.source_xml)
    args_dict["src_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    args_dict["src_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    args_dict["dst_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    args_dict["dst_joint_limits_upper"] = list(dst_robot.joint_limit_upper)

    # Nominal base heights used by zero_nominal physics grounding mode.
    if src_robot.nominal_base_height is not None:
        args_dict["src_start_height"] = float(src_robot.nominal_base_height)
    if dst_robot.nominal_base_height is not None:
        args_dict["dst_start_height"] = float(dst_robot.nominal_base_height)

    src_stats = _resolve_dataset_path(resolved.src_robot, "train", "stats")
    src_train = _resolve_dataset_path(resolved.src_robot, "train", "train")
    src_test = _resolve_dataset_path(resolved.src_robot, "test", "test")
    dst_stats = _resolve_dataset_path(resolved.dst_robot, "train", "stats")
    dst_train = _resolve_dataset_path(resolved.dst_robot, "train", "train")
    dst_test = _resolve_dataset_path(resolved.dst_robot, "test", "test")

    args_dict["srcstats_path"] = src_stats
    args_dict["src_train_path"] = src_train
    args_dict["src_test_path"] = src_test
    args_dict["dststats_path"] = dst_stats
    args_dict["dst_train_path"] = dst_train
    args_dict["dst_test_path"] = dst_test

    missing = []
    for key in ("srcstats_path", "src_train_path", "src_test_path",
                "dststats_path", "dst_train_path", "dst_test_path"):
        v = args_dict.get(key, None)
        if v is None or (not Path(v).exists()):
            missing.append(key)
    if missing:
        root_msg = ", ".join(str(p) for p in dataset_roots)
        raise FileNotFoundError(
            "Missing required processed dataset files for training. "
            f"Missing keys: {missing}. Searched roots: {root_msg}"
        )

    # Backward aliases retained.
    args_dict["hum_njoints"] = int(src_robot.njoints)
    args_dict["dog_njoints"] = int(dst_robot.njoints)
    args_dict["hum_nbodies"] = int(src_robot.nbodies)
    args_dict["dog_nbodies"] = int(dst_robot.nbodies)
    args_dict["hum_end"] = list(resolved.src_feet_indices)
    args_dict["dog_end"] = list(resolved.dst_feet_indices)
    args_dict["hum_ee"] = list(resolved.src_ee_indices)
    args_dict["dog_ee"] = list(resolved.dst_ee_indices)
    args_dict["humstats_path"] = args_dict.get("srcstats_path", args_dict.get("humstats_path"))
    args_dict["dogstats_path"] = args_dict.get("dststats_path", args_dict.get("dogstats_path"))
    args_dict["hum_train_path"] = args_dict.get("src_train_path", args_dict.get("hum_train_path"))
    args_dict["dog_train_path"] = args_dict.get("dst_train_path", args_dict.get("dog_train_path"))
    args_dict["hum_test_path"] = args_dict.get("src_test_path", args_dict.get("hum_test_path"))
    args_dict["dog_test_path"] = args_dict.get("dst_test_path", args_dict.get("dog_test_path"))
    args_dict["hum_joint_limits_lower"] = list(src_robot.joint_limit_lower)
    args_dict["hum_joint_limits_upper"] = list(src_robot.joint_limit_upper)
    args_dict["dog_joint_limits_lower"] = list(dst_robot.joint_limit_lower)
    args_dict["dog_joint_limits_upper"] = list(dst_robot.joint_limit_upper)

    # required runtime fields
    args_dict["save_dir"] = cli.save_dir
    args_dict["is_train"] = True
    args_dict.setdefault("architecture_name", "pan")

    # explicit CLI overrides
    if cli.device is not None:
        args_dict["device"] = cli.device
    if cli.batch_size is not None:
        args_dict["batch_size"] = int(cli.batch_size)
    if cli.epoch_num is not None:
        args_dict["epoch_num"] = int(cli.epoch_num)
    if cli.epoch_begin is not None:
        args_dict["epoch_begin"] = int(cli.epoch_begin)
    if cli.save_iter is not None:
        args_dict["save_iter"] = int(cli.save_iter)
    if cli.print_iter is not None:
        args_dict["print_iter"] = int(cli.print_iter)
    if cli.lr_g is not None:
        args_dict["lr_g"] = float(cli.lr_g)
    if cli.lr_d is not None:
        args_dict["lr_d"] = float(cli.lr_d)
    if cli.num_workers is not None:
        args_dict["num_workers"] = int(cli.num_workers)

    _apply_set_overrides(args_dict, cli.set)

    save_dir = Path(args_dict["save_dir"]).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    run_snapshot = {
        "task_family": resolved.task_family,
        "pair_id": resolved.pair_id,
        "src_robot": resolved.src_robot,
        "dst_robot": resolved.dst_robot,
        "model_config": str(model_cfg_path),
        "loss_weights": resolved.loss_weights,
        "legacy_args": args_dict,
    }
    with (save_dir / "refactor_teacher_run.json").open("w", encoding="utf-8") as f:
        json.dump(run_snapshot, f, indent=2)

    print("Prepared teacher training run:")
    print(f"  task_family: {resolved.task_family}")
    print(f"  pair_id: {resolved.pair_id}")
    print(f"  src->dst: {resolved.src_robot} -> {resolved.dst_robot}")
    print(f"  save_dir: {save_dir}")
    print(f"  batch_size: {args_dict.get('batch_size')}")
    print(f"  epoch_num: {args_dict.get('epoch_num')}")
    print(f"  device: {args_dict.get('device')}")
    print(f"  dataset_roots: {[str(x) for x in dataset_roots]}")

    if cli.dry_run:
        print("\nDry-run mode enabled; not starting training.")
        return

    cmd_preview = "python -m csmt.pipelines.train_teacher " + " ".join(shlex.quote(x) for x in [
        "--output-root", str(output_root),
        "--task-family", cli.task_family,
        "--pair-id", cli.pair_id,
        "--save-dir", str(save_dir),
    ])
    if cli.processed_dir is not None:
        cmd_preview += " " + " ".join(shlex.quote(x) for x in ["--processed-dir", cli.processed_dir])
    run_pan_training(args_dict, para_cmd=cmd_preview)


if __name__ == "__main__":
    main()
