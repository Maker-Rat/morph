from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


_COMPILER_KEEP_ATTRS = {
    "angle",
    "autolimits",
    "eulerseq",
    "coordinate",
    "inertiafromgeom",
}

_BODY_KEEP_ATTRS = {
    "name",
    "pos",
    "quat",
    "euler",
    "axisangle",
    "xyaxes",
    "zaxis",
}

_JOINT_KEEP_ATTRS = {
    "name",
    "type",
    "pos",
    "axis",
    "range",
    "limited",
    "ref",
    "armature",
    "damping",
    "stiffness",
}

_INERTIAL_KEEP_ATTRS = {
    "pos",
    "quat",
    "mass",
    "diaginertia",
    "fullinertia",
}


@dataclass
class XmlSummary:
    joint_names: list[str]
    body_names: list[str]
    joint_limit_lower: list[float]
    joint_limit_upper: list[float]


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


def _relative_if_possible(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return str(path)




def _parse_range_attr(range_attr: str | None) -> tuple[float, float] | None:
    if not range_attr:
        return None
    parts = range_attr.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _build_default_class_joint_attrs(root: ET.Element) -> dict[str, dict[str, str]]:
    """
    Build class -> resolved joint attribute map from nested MJCF <default> hierarchy.

    Nested defaults inherit parent default joint attrs, and child attrs override parent attrs.
    """
    class_joint_attrs: dict[str, dict[str, str]] = {}

    def _recurse_default(default_elem: ET.Element, inherited_joint_attrs: dict[str, str]) -> None:
        joint_attrs = dict(inherited_joint_attrs)
        for joint_elem in default_elem.findall('joint'):
            joint_attrs.update(joint_elem.attrib)

        class_name = default_elem.get('class')
        if class_name:
            class_joint_attrs[class_name] = dict(joint_attrs)

        for nested in default_elem.findall('default'):
            _recurse_default(nested, joint_attrs)

    for default_elem in root.findall('default'):
        _recurse_default(default_elem, {})

    return class_joint_attrs


def _iter_body_joints_with_childclass(root: ET.Element):
    worldbody = root.find('worldbody')
    if worldbody is None:
        return

    def _walk_body(body: ET.Element, inherited_childclass: str | None):
        active_childclass = body.attrib.get('childclass', inherited_childclass)
        for joint in body.findall('joint'):
            yield joint, active_childclass
        for child in body.findall('body'):
            yield from _walk_body(child, active_childclass)

    for body in worldbody.findall('body'):
        yield from _walk_body(body, None)

def _iter_world_bodies(root: ET.Element):
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for body in worldbody.iter("body"):
        yield body


def summarize_mjcf(xml_path: Path) -> XmlSummary:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_joint_attrs = _build_default_class_joint_attrs(root)

    joint_names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []

    for joint, inherited_childclass in _iter_body_joints_with_childclass(root) or []:
        jname = joint.attrib.get('name')
        if not jname:
            continue

        joint_names.append(jname)

        # 1) direct per-joint range
        parsed = _parse_range_attr(joint.attrib.get('range'))

        # 2) class-based range (explicit class first, then body childclass)
        if parsed is None:
            cls = joint.attrib.get('class') or inherited_childclass
            if cls:
                parsed = _parse_range_attr(class_joint_attrs.get(cls, {}).get('range'))

        # 3) fallback to zeros when unresolved
        if parsed is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lo, hi = parsed
            lower.append(lo)
            upper.append(hi)

    body_names: list[str] = []
    unnamed_count = 0
    for b in _iter_world_bodies(root) or []:
        bname = b.attrib.get('name')
        if bname:
            body_names.append(bname)
        else:
            body_names.append(f'unnamed_body_{unnamed_count}')
            unnamed_count += 1

    return XmlSummary(
        joint_names=joint_names,
        body_names=body_names,
        joint_limit_lower=lower,
        joint_limit_upper=upper,
    )


def build_stripped_fk_xml(src_xml: Path, dst_fk_xml: Path, overwrite: bool = False) -> None:
    if dst_fk_xml.exists() and not overwrite:
        raise FileExistsError(f"FK xml already exists: {dst_fk_xml}")

    src_tree = ET.parse(src_xml)
    src_root = src_tree.getroot()
    dst_root = ET.Element("mujoco")
    if "model" in src_root.attrib:
        dst_root.set("model", f"{src_root.attrib['model']}_barebones")

    src_compiler = src_root.find("compiler")
    if src_compiler is not None:
        compiler_attrs = {
            key: value
            for key, value in src_compiler.attrib.items()
            if key in _COMPILER_KEEP_ATTRS
        }
        if compiler_attrs:
            ET.SubElement(dst_root, "compiler", compiler_attrs)

    default_joint_type = "hinge"
    src_default_joint = src_root.find("./default/joint")
    if src_default_joint is not None and src_default_joint.attrib.get("type"):
        default_joint_type = src_default_joint.attrib["type"]
    default_node = ET.SubElement(dst_root, "default")
    ET.SubElement(default_node, "joint", {"type": default_joint_type})

    src_worldbody = src_root.find("worldbody")
    if src_worldbody is None:
        raise ValueError(f"worldbody not found in XML: {src_xml}")
    dst_worldbody = ET.SubElement(dst_root, "worldbody")
    for src_body in src_worldbody.findall("body"):
        dst_worldbody.append(_clone_body_for_fk(src_body))

    tree = ET.ElementTree(dst_root)
    ET.indent(tree, space="  ")
    dst_fk_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_fk_xml, encoding="utf-8", xml_declaration=True)


def _clone_body_for_fk(src_body: ET.Element) -> ET.Element:
    dst_body = ET.Element(
        "body",
        {
            key: value
            for key, value in src_body.attrib.items()
            if key in _BODY_KEEP_ATTRS
        },
    )

    for src_child in src_body:
        if src_child.tag == "inertial":
            dst_body.append(
                ET.Element(
                    "inertial",
                    {
                        key: value
                        for key, value in src_child.attrib.items()
                        if key in _INERTIAL_KEEP_ATTRS
                    },
                )
            )
            continue
        if src_child.tag == "joint":
            joint_attrs = {
                key: value
                for key, value in src_child.attrib.items()
                if key in _JOINT_KEEP_ATTRS
            }
            if "type" not in joint_attrs:
                joint_attrs["type"] = "hinge"
            dst_body.append(ET.Element("joint", joint_attrs))
            continue
        if src_child.tag == "body":
            dst_body.append(_clone_body_for_fk(src_child))
            continue

    return dst_body


def write_robot_yaml(
    robot_id: str,
    src_xml: Path,
    fk_xml: Path,
    output_root: Path,
    summary: XmlSummary,
    cfg_path: Path,
    overwrite: bool = False,
) -> None:
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(f"Robot yaml already exists: {cfg_path}")

    data = {
        "robot_id": robot_id,
        "source_xml": _relative_if_possible(src_xml, output_root),
        "fk_xml": _relative_if_possible(fk_xml, output_root),
        "njoints": len(summary.joint_names),
        "nbodies": len(summary.body_names),
        "joint_limits": {
            "lower": summary.joint_limit_lower,
            "upper": summary.joint_limit_upper,
            "source": "xml_effective_range",
        },
        "base_body": summary.body_names[0] if summary.body_names else "",
    }
    yaml_text = yaml.dump(data, sort_keys=False, Dumper=_FlowNumListDumper)
    lines = yaml_text.splitlines()

    insert_after = None
    for idx, line in enumerate(lines):
        if line.startswith("nbodies:"):
            insert_after = idx
            break

    comment_block = [
        "# joint_index_reference:",
        *[f"#   [{i}] {name}" for i, name in enumerate(summary.joint_names)],
        "# body_index_reference:",
        *[f"#   [{i}] {name}" for i, name in enumerate(summary.body_names)],
    ]

    if insert_after is None:
        insert_after = len(lines) - 1
    new_lines = lines[: insert_after + 1] + [""] + comment_block + lines[insert_after + 1 :]

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap robot config + FK xml from MJCF xml.")
    p.add_argument("--xml", required=True, help="Path to source MJCF/XML")
    p.add_argument("--robot-id", required=True, help="Robot id (e.g., g1, go2, anymal)")
    p.add_argument(
        "--output-root",
        default=".",
        help="Refactor root where configs/assets will be written",
    )
    p.add_argument(
        "--robot-config-path",
        default=None,
        help="Optional explicit robot yaml path (default: <output-root>/configs/robots/<robot-id>.yaml)",
    )
    p.add_argument(
        "--fk-xml-path",
        default=None,
        help="Optional explicit FK xml path (default: <output-root>/assets/fk/<robot-id>_fk.xml)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_xml = Path(args.xml).expanduser().resolve()
    if not src_xml.exists():
        raise FileNotFoundError(f"XML not found: {src_xml}")

    output_root = Path(args.output_root).expanduser().resolve()
    cfg_path = (
        Path(args.robot_config_path).expanduser().resolve()
        if args.robot_config_path
        else output_root / "configs" / "robots" / f"{args.robot_id}.yaml"
    )
    fk_path = (
        Path(args.fk_xml_path).expanduser().resolve()
        if args.fk_xml_path
        else output_root / "assets" / "fk" / f"{args.robot_id}_fk.xml"
    )

    summary = summarize_mjcf(src_xml)
    build_stripped_fk_xml(src_xml, fk_path, overwrite=args.overwrite)
    write_robot_yaml(
        args.robot_id,
        src_xml,
        fk_path,
        output_root,
        summary,
        cfg_path,
        overwrite=args.overwrite,
    )

    print("Bootstrapped robot scaffold:")
    print(f"  robot_id: {args.robot_id}")
    print(f"  source_xml: {src_xml}")
    print(f"  fk_xml: {fk_path}")
    print(f"  config_yaml: {cfg_path}")
    print(f"  njoints: {len(summary.joint_names)}")
    print(f"  nbodies: {len(summary.body_names)}")


if __name__ == "__main__":
    main()
