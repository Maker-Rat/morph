from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotSpec:
    robot_id: str
    source_xml: Path
    fk_xml: Path
    njoints: int
    nbodies: int
    base_body: str = ""
    joint_limit_lower: list[float] = field(default_factory=list)
    joint_limit_upper: list[float] = field(default_factory=list)
    nominal_base_height: float | None = None
