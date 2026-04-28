from __future__ import annotations

from pathlib import Path
import yaml

from .base import PairConfig, ResolvedTaskConfig, TaskFamilyConfig


def load_task_family_defaults(config_root: str | Path, task_family: str) -> TaskFamilyConfig:
    p = Path(config_root) / "configs" / "tasks" / task_family / "defaults.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Task defaults not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return TaskFamilyConfig(
        name=str(cfg.get("task_family", task_family)),
        loss_weights=dict(cfg.get("loss_weights", {})),
    )


def load_pair_config(config_root: str | Path, task_family: str, pair_id: str) -> PairConfig:
    p = Path(config_root) / "configs" / "tasks" / task_family / "pairs" / f"{pair_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Task pair config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    corr = cfg.get("correspondence", {})
    indices = cfg.get("indices", {})
    return PairConfig(
        pair_id=str(cfg["pair_id"]),
        task_family=str(cfg.get("task_family", task_family)),
        src_robot=str(cfg["src_robot"]),
        dst_robot=str(cfg["dst_robot"]),
        src_feet_indices=list(indices.get("src_feet", [])),
        dst_feet_indices=list(indices.get("dst_feet", [])),
        src_ee_indices=list(indices.get("src_ee", [])),
        dst_ee_indices=list(indices.get("dst_ee", [])),
        correspondence_body_groups=list(corr.get("body_groups", [])),
        correspondence_joint_groups=list(corr.get("joint_groups", [])),
        loss_overrides=dict(cfg.get("loss_overrides", {})),
    )


def resolve_task_config(config_root: str | Path, task_family: str, pair_id: str) -> ResolvedTaskConfig:
    family = load_task_family_defaults(config_root, task_family)
    pair = load_pair_config(config_root, task_family, pair_id)

    merged_weights = dict(family.loss_weights)
    merged_weights.update(pair.loss_overrides)

    return ResolvedTaskConfig(
        task_family=pair.task_family,
        pair_id=pair.pair_id,
        src_robot=pair.src_robot,
        dst_robot=pair.dst_robot,
        src_feet_indices=list(pair.src_feet_indices),
        dst_feet_indices=list(pair.dst_feet_indices),
        src_ee_indices=list(pair.src_ee_indices),
        dst_ee_indices=list(pair.dst_ee_indices),
        loss_weights=merged_weights,
        correspondence_body_groups=list(pair.correspondence_body_groups),
        correspondence_joint_groups=list(pair.correspondence_joint_groups),
    )
