from pathlib import Path
import yaml

from .base import RobotSpec


def load_robot_spec(path: str | Path) -> RobotSpec:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    limits = cfg.get("joint_limits", {})
    return RobotSpec(
        robot_id=str(cfg["robot_id"]),
        source_xml=Path(cfg["source_xml"]),
        fk_xml=Path(cfg["fk_xml"]),
        njoints=int(cfg["njoints"]),
        nbodies=int(cfg["nbodies"]),
        base_body=str(cfg.get("base_body", "")),
        joint_limit_lower=list(limits.get("lower", [])),
        joint_limit_upper=list(limits.get("upper", [])),
    )
