from dataclasses import dataclass, field


@dataclass
class TaskFamilyConfig:
    name: str
    loss_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class PairConfig:
    pair_id: str
    task_family: str
    src_robot: str
    dst_robot: str
    src_feet_indices: list[int] = field(default_factory=list)
    dst_feet_indices: list[int] = field(default_factory=list)
    src_ee_indices: list[int] = field(default_factory=list)
    dst_ee_indices: list[int] = field(default_factory=list)
    correspondence_body_groups: list[dict] = field(default_factory=list)
    correspondence_joint_groups: list[dict] = field(default_factory=list)
    loss_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class ResolvedTaskConfig:
    task_family: str
    pair_id: str
    src_robot: str
    dst_robot: str
    src_feet_indices: list[int] = field(default_factory=list)
    dst_feet_indices: list[int] = field(default_factory=list)
    src_ee_indices: list[int] = field(default_factory=list)
    dst_ee_indices: list[int] = field(default_factory=list)
    loss_weights: dict[str, float] = field(default_factory=dict)
    correspondence_body_groups: list[dict] = field(default_factory=list)
    correspondence_joint_groups: list[dict] = field(default_factory=list)
