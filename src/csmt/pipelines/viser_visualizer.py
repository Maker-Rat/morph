from __future__ import annotations

import argparse
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from csmt.robots.registry import load_robot_spec


@dataclass
class MotionArrays:
    dof_pos: np.ndarray
    root_pos: np.ndarray
    root_rot: np.ndarray
    fps: float


@dataclass
class ViewerState:
    frame: int
    playing: bool
    history: list[int]
    quit: bool = False


# ── Motion loading ────────────────────────────────────────────────────────────

def _extract_motion_arrays(payload: Any) -> MotionArrays:
    if isinstance(payload, dict):
        dof = payload.get("dof_pos", payload.get("joint_pos", payload.get("joint_positions", None)))
        pos = payload.get("root_pos", payload.get("base_trans", payload.get("base_translation", None)))
        rot = payload.get("root_rot", payload.get("base_quat", payload.get("base_rotation", None)))
        if dof is None:
            raise ValueError("PKL dict missing dof_pos/joint_pos")
        dof = np.asarray(dof, dtype=np.float32)
        if pos is None:
            pos = np.zeros((len(dof), 3), dtype=np.float32)
        if rot is None:
            rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(dof), 1))
        fps = float(payload.get("fps", 30.0))
        return MotionArrays(
            dof_pos=dof,
            root_pos=np.asarray(pos, dtype=np.float32),
            root_rot=np.asarray(rot, dtype=np.float32),
            fps=fps,
        )
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("Empty list PKL")
        return MotionArrays(
            dof_pos=np.asarray([f[2] for f in payload], dtype=np.float32),
            root_pos=np.asarray([f[0] for f in payload], dtype=np.float32),
            root_rot=np.asarray([f[1] for f in payload], dtype=np.float32),
            fps=30.0,
        )
    raise ValueError(f"Unsupported PKL payload type: {type(payload)}")


def _load_motion(path: Path) -> MotionArrays:
    with path.open("rb") as f:
        return _extract_motion_arrays(pickle.load(f))


def _resolve_user_path(path_text: str, output_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = output_root / path
    return path.resolve()


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s], dtype=np.float64)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s], dtype=np.float64)
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s], dtype=np.float64)
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


# ── Robot spec / path resolution ─────────────────────────────────────────────

def _resolve_robot_xml(output_root: Path, robot_id: str, xml_override: str | None) -> Path:
    if xml_override:
        return Path(xml_override).expanduser().resolve()
    spec = load_robot_spec(output_root / "configs" / "robots" / f"{robot_id}.yaml")
    return (spec.source_xml if spec.source_xml.is_absolute() else output_root / spec.source_xml).resolve()


# ── MuJoCo joint helpers (kinematics only) ───────────────────────────────────

def _non_free_joint_qpos(model: Any) -> list[int]:
    import mujoco
    out: list[int] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        out.append(int(model.jnt_qposadr[j]))
    return out


def _mujoco_joint_names(model: Any) -> list[str]:
    """Return ordered list of non-free joint names from a MuJoCo model."""
    import mujoco
    names: list[str] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        names.append(name)
    return names


def _apply_motion_frame(
    model: Any,
    data: Any,
    motion: MotionArrays,
    joint_qpos: list[int],
    frame: int,
    anchor_root: np.ndarray,
) -> None:
    import mujoco
    k = int(np.clip(frame, 0, motion.dof_pos.shape[0] - 1))
    data.qpos[:] = 0.0
    has_free_base = (model.njnt > 0
                     and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
                     and model.nq >= 7)
    if has_free_base:
        data.qpos[0:3] = motion.root_pos[k] - anchor_root
        data.qpos[3:7] = _to_wxyz(motion.root_rot[k])
    map_dim = min(len(joint_qpos), int(motion.dof_pos.shape[1]))
    for i in range(map_dim):
        data.qpos[joint_qpos[i]] = motion.dof_pos[k, i]
    mujoco.mj_forward(model, data)


# ── Trail ─────────────────────────────────────────────────────────────────────

def _root_trail_points(motion: MotionArrays, history: list[int], anchor_root: np.ndarray) -> np.ndarray:
    if len(history) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.clip(np.asarray(history, dtype=np.int64), 0, motion.root_pos.shape[0] - 1)
    return (motion.root_pos[idx] - anchor_root[None, :]).astype(np.float32)


# ── Bone overlay (MuJoCo-based) ───────────────────────────────────────────────

def _body_line_segments(model: Any, data: Any) -> np.ndarray:
    import mujoco
    segs = []
    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        if parent <= 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if any(s in name.lower() for s in ("mocap", "imu", "contour")):
            continue
        p0 = data.xpos[parent].copy()
        p1 = data.xpos[body_id].copy()
        if float(np.linalg.norm(p1 - p0)) < 0.025:
            continue
        segs.append([p0, p1])
    return np.asarray(segs, dtype=np.float32)


# ── MuJoCo mesh rendering helpers ────────────────────────────────────────────

@dataclass
class MujocoGeomHandle:
    geom_id: int
    handle: Any


@dataclass
class MujocoBatchedBodyHandle:
    body_id: int
    handle: Any


@dataclass
class FootPoint:
    name: str
    kind: str
    idx: int


def _is_fixed_mujoco_body(model: Any, body_id: int) -> bool:
    root_id = int(model.body_rootid[body_id])
    return int(model.body_weldid[body_id]) == 0 and int(model.body_mocapid[root_id]) < 0


def _is_visual_geom(model: Any, geom_id: int) -> bool:
    return (
        not _is_fixed_mujoco_body(model, int(model.geom_bodyid[geom_id]))
        and int(model.geom_contype[geom_id]) == 0
        and int(model.geom_conaffinity[geom_id]) == 0
    )


def _is_dynamic_geom(model: Any, geom_id: int) -> bool:
    return not _is_fixed_mujoco_body(model, int(model.geom_bodyid[geom_id]))


def _render_geom_ids(model: Any) -> list[int]:
    visual_ids = [geom_id for geom_id in range(model.ngeom) if _is_visual_geom(model, geom_id)]
    if visual_ids:
        return visual_ids
    fallback_ids = [geom_id for geom_id in range(model.ngeom) if _is_dynamic_geom(model, geom_id)]
    if fallback_ids:
        print(
            "[warn] No non-collision visual geoms found; rendering dynamic geoms "
            "as visuals. This is expected for robots whose XML reuses visual "
            "geoms as collision geoms."
        )
    return fallback_ids


def _geom_name(model: Any, geom_id: int) -> str:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name or f"geom_{geom_id}"


def _pbr_texture_image(model: Any, mat_id: int) -> Any | None:
    import mujoco
    from PIL import Image

    tex_id = int(model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    if tex_id < 0:
        tex_id = int(model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
    if tex_id < 0:
        return None

    width = int(model.tex_width[tex_id])
    height = int(model.tex_height[tex_id])
    channels = int(model.tex_nchannel[tex_id])
    adr = int(model.tex_adr[tex_id])
    raw = model.tex_data[adr: adr + width * height * channels]
    if channels == 1:
        return Image.fromarray(np.flipud(raw.reshape(height, width)).astype(np.uint8), mode="L")
    if channels == 3:
        return Image.fromarray(np.flipud(raw.reshape(height, width, 3)).astype(np.uint8))
    if channels == 4:
        return Image.fromarray(np.flipud(raw.reshape(height, width, 4)).astype(np.uint8))
    return None


def _mujoco_material_rgba(model: Any, geom_id: int) -> np.ndarray:
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return np.asarray(model.mat_rgba[mat_id], dtype=np.float64)
    return np.asarray(model.geom_rgba[geom_id], dtype=np.float64)


def _mujoco_mesh_geom_to_trimesh(model: Any, geom_id: int) -> Any:
    import mujoco
    import trimesh
    import trimesh.visual
    import trimesh.visual.material

    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_start: vert_start + vert_count], dtype=np.float32)
    faces = np.asarray(model.mesh_face[face_start: face_start + face_count], dtype=np.int64)

    texcoord_count = int(model.mesh_texcoordnum[mesh_id])
    mat_id = int(model.geom_matid[geom_id])
    texture = _pbr_texture_image(model, mat_id) if mat_id >= 0 else None
    if texcoord_count > 0 and texture is not None:
        texcoord_start = int(model.mesh_texcoordadr[mesh_id])
        uvs = model.mesh_texcoord[texcoord_start: texcoord_start + texcoord_count]
        face_uvs = model.mesh_facetexcoord[face_start: face_start + face_count]
        flat_vertices = vertices[faces.reshape(-1)]
        flat_uvs = uvs[face_uvs.reshape(-1)]
        flat_faces = np.arange(face_count * 3, dtype=np.int64).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=flat_vertices, faces=flat_faces, process=False)
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.asarray(model.mat_rgba[mat_id], dtype=np.float64),
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=flat_uvs, material=material)
        return mesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = (_mujoco_material_rgba(model, geom_id).clip(0.0, 1.0) * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
    )
    return mesh


def _mujoco_primitive_geom_to_trimesh(model: Any, geom_id: int) -> Any:
    import mujoco
    import trimesh
    import trimesh.visual

    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]))
    else:
        raise ValueError(f"Unsupported MuJoCo geom type for paper viewer: {geom_type}")

    rgba = (_mujoco_material_rgba(model, geom_id).clip(0.0, 1.0) * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1))
    )
    return mesh


def _mujoco_geom_to_trimesh(model: Any, geom_id: int) -> Any:
    import mujoco

    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        return _mujoco_mesh_geom_to_trimesh(model, geom_id)
    return _mujoco_primitive_geom_to_trimesh(model, geom_id)


def _wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _visual_geoms_by_body(model: Any) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for geom_id in _render_geom_ids(model):
        body_id = int(model.geom_bodyid[geom_id])
        out.setdefault(body_id, []).append(geom_id)
    return out


def _mujoco_body_visual_mesh(model: Any, geom_ids: list[int]) -> Any:
    import trimesh

    meshes = []
    for geom_id in geom_ids:
        mesh = _mujoco_geom_to_trimesh(model, geom_id).copy()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _wxyz_to_matrix(model.geom_quat[geom_id])
        transform[:3, 3] = model.geom_pos[geom_id]
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _create_mujoco_render_handles(
    server: Any,
    model: Any,
    data: Any,
    mesh_saturation: float,
    mesh_value: float,
) -> list[MujocoGeomHandle]:
    handles: list[MujocoGeomHandle] = []
    visual_geom_ids = _render_geom_ids(model)
    for geom_id in visual_geom_ids:
        mesh = _mujoco_geom_to_trimesh(model, geom_id)
        _boost_mesh_colors(mesh, saturation=mesh_saturation, value=mesh_value)
        name = f"/robot_mj/{geom_id:03d}_{_geom_name(model, geom_id)}"
        if type(mesh.visual).__name__ == "TextureVisuals":
            handle = server.scene.add_mesh_trimesh(
                name,
                mesh,
                position=data.geom_xpos[geom_id].astype(np.float64),
                wxyz=_wxyz_from_matrix(data.geom_xmat[geom_id]),
                cast_shadow=True,
                receive_shadow=True,
            )
        else:
            rgba = _boost_rgba(
                _mujoco_material_rgba(model, geom_id),
                saturation=mesh_saturation,
                value=mesh_value,
            )
            handle = server.scene.add_mesh_simple(
                name,
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32),
                color=tuple(int(c) for c in (rgba[:3] * 255.0).astype(np.uint8)),
                material="standard",
                flat_shading=False,
                position=data.geom_xpos[geom_id].astype(np.float64),
                wxyz=_wxyz_from_matrix(data.geom_xmat[geom_id]),
                cast_shadow=True,
                receive_shadow=True,
            )
        handles.append(MujocoGeomHandle(geom_id=geom_id, handle=handle))
    print(f"MuJoCo visual geoms: {len(handles)}")
    return handles


def _boost_mesh_colors(mesh: Any, saturation: float, value: float) -> None:
    if abs(float(saturation) - 1.0) < 1e-6 and abs(float(value) - 1.0) < 1e-6:
        return
    if type(getattr(mesh, "visual", None)).__name__ != "ColorVisuals":
        return
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)
    if colors.size == 0 or colors.shape[-1] < 3:
        return
    rgb = colors[:, :3] / 255.0
    mean = rgb.mean(axis=1, keepdims=True)
    boosted = mean + (rgb - mean) * max(0.0, float(saturation))
    boosted = np.clip(boosted * max(0.0, float(value)), 0.0, 1.0)
    colors[:, :3] = boosted * 255.0
    mesh.visual.vertex_colors = colors.astype(np.uint8)


def _boost_rgba(rgba: np.ndarray, saturation: float = 1.35, value: float = 1.08) -> np.ndarray:
    rgb = np.asarray(rgba, dtype=np.float64)[:3].clip(0.0, 1.0)
    mean = float(np.mean(rgb))
    boosted = mean + (rgb - mean) * max(0.0, float(saturation))
    boosted = boosted * max(0.0, float(value))
    return np.concatenate([boosted.clip(0.0, 1.0), [float(np.asarray(rgba)[3]) if len(rgba) > 3 else 1.0]])


def _create_mujoco_ghost_handles(
    server: Any,
    model: Any,
    ghost_slots: int,
    ghost_color: tuple[int, int, int],
    base_opacity: float,
    face_stride: int,
) -> list[MujocoBatchedBodyHandle]:
    ghost_handles: list[MujocoBatchedBodyHandle] = []
    if ghost_slots <= 0:
        return ghost_handles

    hidden_positions = np.tile(np.array([0.0, 0.0, -1000.0], dtype=np.float32), (ghost_slots, 1))
    identity_wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (ghost_slots, 1))
    opacities = np.asarray(
        [
            float(np.clip(
                base_opacity * (1.0 - (slot_idx / max(ghost_slots, 1)) * 0.72),
                0.01,
                0.65,
            ))
            for slot_idx in range(ghost_slots)
        ],
        dtype=np.float32,
    )
    for body_id, geom_ids in _visual_geoms_by_body(model).items():
        mesh = _mujoco_body_visual_mesh(model, geom_ids)
        vertices, faces = _strided_mesh_vertices_faces(mesh, face_stride)
        handle = server.scene.add_batched_meshes_simple(
            f"/ghost_mj/body_{body_id:03d}",
            vertices,
            faces,
            batched_wxyzs=identity_wxyzs,
            batched_positions=hidden_positions,
            batched_colors=np.asarray(ghost_color, dtype=np.uint8),
            batched_opacities=opacities,
            lod="off",
            material="standard",
            flat_shading=False,
            side="double",
            visible=True,
            cast_shadow=False,
            receive_shadow=False,
        )
        ghost_handles.append(MujocoBatchedBodyHandle(body_id=body_id, handle=handle))
    return ghost_handles


def _update_mujoco_batched_ghosts(
    handles: list[MujocoBatchedBodyHandle],
    slot_data: list[Any | None],
) -> None:
    if not handles:
        return
    slot_count = len(slot_data)
    hidden_positions = np.tile(np.array([0.0, 0.0, -1000.0], dtype=np.float32), (slot_count, 1))
    identity_wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (slot_count, 1))
    for item in handles:
        positions = hidden_positions.copy()
        wxyzs = identity_wxyzs.copy()
        for slot_idx, data in enumerate(slot_data):
            if data is None:
                continue
            positions[slot_idx] = data.xpos[item.body_id].astype(np.float32)
            wxyzs[slot_idx] = _wxyz_from_matrix(data.xmat[item.body_id]).astype(np.float32)
        item.handle.batched_positions = positions
        item.handle.batched_wxyzs = wxyzs


def _strided_mesh_vertices_faces(mesh: Any, face_stride: int) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    stride = max(1, int(face_stride))
    if stride == 1 or len(faces) <= 64:
        return vertices, faces.astype(np.int32)
    sparse_faces = faces[::stride]
    used, inverse = np.unique(sparse_faces.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def _update_mujoco_handles(handles: list[MujocoGeomHandle], data: Any, visible: bool = True) -> None:
    for item in handles:
        geom_id = item.geom_id
        item.handle.position = data.geom_xpos[geom_id].astype(np.float64)
        item.handle.wxyz = _wxyz_from_matrix(data.geom_xmat[geom_id])
        if item.handle.visible != visible:
            item.handle.visible = visible


# ── Foot contact / slip overlays ─────────────────────────────────────────────

def _resolve_foot_points(model: Any, requested: list[str] | None = None) -> list[FootPoint]:
    import mujoco

    points: list[FootPoint] = []
    if requested:
        for name in requested:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                points.append(FootPoint(name=name, kind="geom", idx=int(geom_id)))
                continue
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                points.append(FootPoint(name=name, kind="body", idx=int(body_id)))
                continue
            print(f"[warn] foot marker target not found as geom/body: {name}")
        return points

    for name in ("FR", "FL", "RR", "RL"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            points.append(FootPoint(name=name, kind="geom", idx=int(geom_id)))
    if points:
        return points

    foot_body_names: list[str] = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        low = body_name.lower()
        if "foot" in low and any(token in low for token in ("fl", "fr", "ml", "mr", "bl", "br", "end")):
            foot_body_names.append(body_name)
    if foot_body_names:
        for body_name in sorted(foot_body_names):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            points.append(FootPoint(name=body_name, kind="body", idx=int(body_id)))
        return points

    for side in ("left", "right"):
        candidates: list[tuple[int, str]] = []
        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            low = body_name.lower()
            if side in low and any(token in low for token in ("toe", "foot", "ankle_roll")):
                priority = 0 if "toe" in low else 1 if "foot" in low else 2
                candidates.append((priority, body_name))
        if candidates:
            _, body_name = sorted(candidates)[0]
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            points.append(FootPoint(name=body_name, kind="body", idx=int(body_id)))
    return points


def _foot_position(data: Any, point: FootPoint) -> np.ndarray:
    if point.kind == "geom":
        return np.asarray(data.geom_xpos[point.idx], dtype=np.float64)
    return np.asarray(data.xpos[point.idx], dtype=np.float64)


def _foot_positions_at_frame(
    model: Any,
    scratch_data: Any,
    motion: MotionArrays,
    joint_qpos: list[int],
    frame: int,
    anchor_root: np.ndarray,
    foot_points: list[FootPoint],
) -> list[np.ndarray]:
    _apply_motion_frame(model, scratch_data, motion, joint_qpos, frame, anchor_root)
    return [_foot_position(scratch_data, p).copy() for p in foot_points]


def _foot_contact_status(
    pos: np.ndarray,
    ref_pos: np.ndarray,
    fps: float,
    floor_z: float,
    contact_height: float,
    penetration_depth: float,
    slip_speed: float,
) -> tuple[str, float]:
    height = float(pos[2] - floor_z)
    xy_speed = float(np.linalg.norm(pos[:2] - ref_pos[:2]) * fps)
    near_ground = height <= contact_height
    if height < -abs(penetration_depth):
        return "penetration", xy_speed
    if near_ground and xy_speed >= slip_speed:
        return "slip", xy_speed
    if near_ground:
        return "contact", xy_speed
    return "swing", xy_speed


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Viser PKL viewer for paper figures.")
    p.add_argument("--output-root", type=str, default=".")
    p.add_argument("--robot-id", type=str, required=True)
    p.add_argument("--pkl", type=str, required=True)
    p.add_argument("--xml", type=str, default=None, help="MuJoCo XML for rendering/kinematics")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--play", action="store_true")
    # ── Theme ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--theme", choices=["dark_studio", "light_studio", "white"], default="dark_studio",
        help="dark_studio: near-black bg + cool rim (Go2 dark body).  "
             "light_studio: off-white bg + warm grey floor.  "
             "white: pure white, minimal.",
    )
    # ── Floor ────────────────────────────────────────────────────────────────
    p.add_argument("--floor", choices=["none", "grid", "solid"], default="grid")
    p.add_argument("--grid-size", type=float, default=8.0)
    p.add_argument("--cell-size", type=float, default=0.25)
    # ── Mesh color boost for non-textured/simple geoms ───────────────────────
    p.add_argument("--mesh-saturation", type=float, default=1.0)
    p.add_argument("--mesh-value", type=float, default=0.78)
    # ── Ghost meshes ─────────────────────────────────────────────────────────
    p.add_argument("--ghost", action="store_true")
    p.add_argument("--ghost-frames", type=int, default=3, help="Ghost slot count")
    p.add_argument("--max-ghost-frames", type=int, default=12, help="Maximum ghost slots available in the GUI")
    p.add_argument("--ghost-stride", type=int, default=6, help="History frames between ghost slots")
    p.add_argument("--ghost-opacity", type=float, default=0.35, help="Opacity of nearest ghost")
    p.add_argument("--ghost-color", type=int, nargs=3, default=(65, 65, 65))
    p.add_argument(
        "--ghost-face-stride",
        type=int,
        default=1,
        help="Render every nth face for ghost meshes; keep 1 for solid paper-quality ghosts.",
    )
    p.add_argument("--ghost-trajectory", action="store_true")
    p.add_argument("--ghost-trail-color", type=int, nargs=3, default=(90, 170, 255))
    p.add_argument("--ghost-trail-width", type=float, default=7.0)
    # ── Bone overlay ─────────────────────────────────────────────────────────
    p.add_argument("--bone-overlay", action="store_true", help="Show MuJoCo skeleton lines (debug)")
    p.add_argument("--bone-color", type=int, nargs=3, default=(80, 160, 255))
    p.add_argument("--bone-width", type=float, default=2.0)
    # ── Foot contact / slip overlay ──────────────────────────────────────────
    p.add_argument("--foot-contacts", action="store_true", help="Show foot contact markers")
    p.add_argument("--foot-points", nargs="*", default=None, help="Optional geom/body names for feet")
    p.add_argument("--foot-contact-height", type=float, default=0.05)
    p.add_argument("--foot-penetration-depth", type=float, default=0.015)
    p.add_argument("--foot-slip-speed", type=float, default=0.18)
    p.add_argument("--foot-marker-radius", type=float, default=0.055)
    p.add_argument("--foot-marker-opacity", type=float, default=1.0)
    p.add_argument("--foot-contact-color", type=int, nargs=3, default=(0, 190, 75))
    p.add_argument("--foot-error-color", type=int, nargs=3, default=(255, 45, 20))
    p.add_argument("--foot-slip-vectors", action="store_true", help="Draw red floor streaks for lateral slip")
    p.add_argument("--foot-slip-width", type=float, default=8.0)
    # ── Root anchor ──────────────────────────────────────────────────────────
    p.add_argument("--anchor-root", action="store_true", default=True)
    p.add_argument("--no-anchor-root", dest="anchor_root", action="store_false")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import mujoco
    import viser

    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    pkl_path = Path(args.pkl).expanduser().resolve()
    xml_path = _resolve_robot_xml(output_root, args.robot_id, args.xml)

    print(f"MuJoCo XML : {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_qpos = _non_free_joint_qpos(model)

    motion = _load_motion(pkl_path)
    if motion.dof_pos.shape[0] == 0:
        raise ValueError(f"Motion has zero frames: {pkl_path}")

    anchor_root = np.zeros(3, dtype=np.float32)
    if args.anchor_root:
        anchor_root[:2] = motion.root_pos[0, :2]

    n_frames = int(motion.dof_pos.shape[0])
    max_ghost_frames = max(0, int(args.ghost_frames), int(args.max_ghost_frames))
    state = ViewerState(
        frame=int(np.clip(args.start_frame, 0, n_frames - 1)),
        playing=bool(args.play),
        history=[],
    )
    lock = threading.RLock()
    _apply_motion_frame(model, data, motion, joint_qpos, state.frame, anchor_root)

    # ── Viser server + theme ──────────────────────────────────────────────────
    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(dark_mode=False, control_layout="floating")
    server.scene.configure_default_lights(enabled=False, cast_shadow=True)

    theme = args.theme
    if theme == "dark_studio":
        server.scene.configure_environment_map(
            "studio", background=False, background_blurriness=0.0,
            background_intensity=0.0, environment_intensity=0.9,
        )
        server.scene.set_background_image(
            np.broadcast_to(np.array([14, 15, 18], dtype=np.uint8), (2, 2, 3)).copy(), format="png"
        )
        server.scene.add_light_directional("/lights/key", color=(255, 252, 245),
            intensity=3.2, cast_shadow=True, position=(2.5, -3.2, 5.2))
        server.scene.add_light_directional("/lights/rim", color=(160, 200, 255),
            intensity=1.25, cast_shadow=False, position=(-2.5, 3.0, 3.8))
        server.scene.add_light_directional("/lights/fill", color=(255, 240, 210),
            intensity=0.22, cast_shadow=False, position=(0.0, -2.0, 1.5))
        server.scene.add_light_ambient("/lights/ambient", color=(180, 195, 220), intensity=0.08)
        floor_plane, floor_cell, floor_section = (30, 32, 36), (48, 52, 60), (74, 82, 96)
        shadow_opacity = 0.75

    elif theme == "light_studio":
        server.scene.configure_environment_map(
            "studio", background=False, background_blurriness=0.0,
            background_intensity=0.0, environment_intensity=0.8,
        )
        server.scene.set_background_image(
            np.broadcast_to(np.array([232, 235, 239], dtype=np.uint8), (2, 2, 3)).copy(), format="png"
        )
        server.scene.add_light_directional("/lights/key", color=(255, 255, 255),
            intensity=2.0, cast_shadow=True, position=(2.8, -3.4, 4.8))
        server.scene.add_light_directional("/lights/rim", color=(190, 215, 255),
            intensity=0.75, cast_shadow=False, position=(-2.8, 2.8, 3.4))
        server.scene.add_light_directional("/lights/warm_fill", color=(255, 236, 210),
            intensity=0.22, cast_shadow=False, position=(-1.5, -1.0, 1.5))
        server.scene.add_light_ambient("/lights/fill", color=(255, 255, 255), intensity=0.15)
        floor_plane, floor_cell, floor_section = (198, 202, 208), (164, 171, 182), (118, 130, 146)
        shadow_opacity = 0.48

    else:  # white
        server.scene.configure_environment_map(
            "studio", background=False, background_blurriness=0.0,
            background_intensity=0.0, environment_intensity=0.85,
        )
        server.scene.set_background_image(
            np.broadcast_to(np.array([255, 255, 255], dtype=np.uint8), (2, 2, 3)).copy(), format="png"
        )
        server.scene.add_light_directional("/lights/key", color=(255, 255, 255),
            intensity=1.8, cast_shadow=True, position=(2.5, -3.0, 4.2))
        server.scene.add_light_directional("/lights/rim", color=(205, 225, 255),
            intensity=0.45, cast_shadow=False, position=(-2.5, 2.0, 3.0))
        server.scene.add_light_ambient("/lights/fill", color=(255, 255, 255), intensity=0.25)
        floor_plane, floor_cell, floor_section = (250, 250, 250), (235, 235, 235), (200, 200, 200)
        shadow_opacity = 0.2

    if args.floor != "none":
        server.scene.add_grid(
            "/floor",
            width=float(args.grid_size), height=float(args.grid_size), plane="xy",
            cell_size=float(args.cell_size) if args.floor == "grid" else float(args.cell_size) * 100,
            section_size=max(1.0, float(args.cell_size) * 4.0),
            cell_color=floor_cell if args.floor == "grid" else floor_plane,
            section_color=floor_section if args.floor == "grid" else floor_plane,
            cell_thickness=0.5, section_thickness=0.9, infinite_grid=True,
            fade_distance=20.0, fade_strength=1.5,
            plane_color=floor_plane, plane_opacity=0.99, shadow_opacity=shadow_opacity,
        )

    ghost_color = tuple(int(c) for c in args.ghost_color)
    ghost_opacity = float(args.ghost_opacity)
    if theme == "light_studio" and ghost_color == (120, 180, 255):
        ghost_color = (72, 78, 88)
        if abs(ghost_opacity - 0.2) < 1e-6:
            ghost_opacity = 0.28

    print(f"MuJoCo joints ({len(_mujoco_joint_names(model))}): {_mujoco_joint_names(model)}")

    # ── Primary robot: render MuJoCo visual geoms directly ────────────────────
    robot_handles = _create_mujoco_render_handles(
        server,
        model,
        data,
        mesh_saturation=float(args.mesh_saturation),
        mesh_value=float(args.mesh_value),
    )

    def _update_primary(frame: int) -> None:
        _update_mujoco_handles(robot_handles, data, visible=True)

    # ── Ghost robots ──────────────────────────────────────────────────────────
    ghost_data_slots = [mujoco.MjData(model) for _ in range(max_ghost_frames)]
    ghost_sets = _create_mujoco_ghost_handles(
        server,
        model,
        max_ghost_frames,
        ghost_color,
        ghost_opacity,
        int(args.ghost_face_stride),
    )

    # ── Overlay state ─────────────────────────────────────────────────────────
    show_bones = [bool(args.bone_overlay)]
    show_ghosts = [bool(args.ghost)]
    show_ghost_traj = [bool(args.ghost_trajectory)]
    show_foot_contacts = [bool(args.foot_contacts)]
    show_foot_slip_vectors = [bool(args.foot_slip_vectors)]
    ghost_stride = [max(1, int(args.ghost_stride))]
    bone_handle = None
    trail_handle = None
    foot_slip_handle = None

    if show_bones[0]:
        bone_handle = server.scene.add_line_segments(
            "/overlay/bones", _body_line_segments(model, data),
            colors=tuple(int(c) for c in args.bone_color),
            line_width=float(args.bone_width),
        )

    foot_points = _resolve_foot_points(model, args.foot_points)
    foot_scratch = mujoco.MjData(model)
    if args.foot_contacts:
        print(f"Foot markers: {[p.name for p in foot_points]}")
    marker_handles: list[dict[str, Any]] = []
    marker_colors = {
        "contact": tuple(int(c) for c in args.foot_contact_color),
        "swing": tuple(int(c) for c in args.foot_error_color),
        "slip": tuple(int(c) for c in args.foot_error_color),
        "penetration": tuple(int(c) for c in args.foot_error_color),
    }
    marker_opacity = float(np.clip(args.foot_marker_opacity, 0.0, 1.0))

    def create_marker_handles(points: list[FootPoint]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for foot_idx, point in enumerate(points):
            states: dict[str, Any] = {}
            for status_name, color in marker_colors.items():
                states[status_name] = server.scene.add_icosphere(
                    f"/overlay/foot_contacts/{foot_idx}_{point.name}/{status_name}",
                    radius=float(args.foot_marker_radius),
                    color=color,
                    opacity=marker_opacity,
                    subdivisions=2,
                    material="toon5",
                    flat_shading=True,
                    position=(0.0, 0.0, -1000.0),
                    visible=False,
                    cast_shadow=False,
                    receive_shadow=False,
                )
            out.append(states)
        return out

    marker_handles = create_marker_handles(foot_points)

    # ── GUI ───────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Load Motion", expand_by_default=True):
        robot_text = server.gui.add_text("Robot ID", initial_value=args.robot_id)
        pkl_text = server.gui.add_text("PKL Path", initial_value=args.pkl)
        xml_text = server.gui.add_text("XML Override", initial_value=args.xml or "")
        load_button = server.gui.add_button("Load Motion")
        reset_button = server.gui.add_button("Reset Motion")
        load_status = server.gui.add_markdown(f"Loaded `{args.robot_id}`")

    play_cb = server.gui.add_checkbox("Play", initial_value=state.playing)
    frame_slider = server.gui.add_slider("Frame", min=0, max=max(100000, n_frames - 1), step=1,
                                         initial_value=state.frame)
    speed_slider = server.gui.add_slider("Speed", min=0.05, max=3.0, step=0.05,
                                          initial_value=float(args.speed))
    bone_cb = server.gui.add_checkbox("Bone Overlay", initial_value=show_bones[0])
    ghost_cb = server.gui.add_checkbox("Ghost Meshes", initial_value=show_ghosts[0])
    ghost_traj_cb = server.gui.add_checkbox("Ghost Trajectory", initial_value=show_ghost_traj[0])
    foot_cb = server.gui.add_checkbox("Foot Contacts", initial_value=show_foot_contacts[0])
    slip_cb = server.gui.add_checkbox("Slip Vectors", initial_value=show_foot_slip_vectors[0])
    active_ghost_slider = server.gui.add_slider(
        "Ghost Frames", min=0, max=max_ghost_frames, step=1,
        initial_value=min(int(args.ghost_frames), max_ghost_frames) if show_ghosts[0] else 0,
    )
    ghost_stride_slider = server.gui.add_slider(
        "Ghost Stride", min=1, max=60, step=1, initial_value=ghost_stride[0],
    )
    server.gui.add_markdown("Hotkeys: `Space` play/pause · `p` prev · `n` next")
    prev_button = server.gui.add_button("Previous Frame")
    next_button = server.gui.add_button("Next Frame")
    toggle_cmd = server.gui.add_command("Play / Pause", hotkey="Space")
    prev_cmd = server.gui.add_command("Previous Frame", hotkey="p")
    next_cmd = server.gui.add_command("Next Frame", hotkey="n")

    # ── Frame update ──────────────────────────────────────────────────────────

    def update_ghosts() -> None:
        nonlocal trail_handle
        active = int(active_ghost_slider.value) if show_ghosts[0] else 0
        stride = max(1, int(ghost_stride[0]))

        visible_slot_data: list[Any | None] = [None] * max_ghost_frames
        for slot_idx in range(max_ghost_frames):
            hist_idx = -(slot_idx + 1) * stride
            visible = active > slot_idx and len(state.history) >= abs(hist_idx)
            if visible:
                _apply_motion_frame(
                    model,
                    ghost_data_slots[slot_idx],
                    motion,
                    joint_qpos,
                    state.history[hist_idx],
                    anchor_root,
                )
                visible_slot_data[slot_idx] = ghost_data_slots[slot_idx]
        _update_mujoco_batched_ghosts(ghost_sets, visible_slot_data)

        if trail_handle is not None:
            trail_handle.remove()
            trail_handle = None
        if show_ghost_traj[0] and show_ghosts[0] and len(state.history) >= 2:
            pts = _root_trail_points(motion, state.history[-max(2, active * stride):], anchor_root)
            if pts.shape[0] >= 2:
                trail_handle = server.scene.add_spline_catmull_rom(
                    "/overlay/ghost_trajectory", pts,
                    color=tuple(int(c) for c in args.ghost_trail_color),
                    line_width=float(args.ghost_trail_width),
                )

    def update_foot_contacts() -> None:
        nonlocal foot_slip_handle
        if foot_slip_handle is not None:
            foot_slip_handle.remove()
            foot_slip_handle = None

        if not foot_points or not show_foot_contacts[0]:
            for states in marker_handles:
                for handle in states.values():
                    handle.visible = False
            return

        fps = float(args.fps if args.fps is not None else motion.fps)
        ref_frame = max(0, state.frame - 1)
        if ref_frame == state.frame and n_frames > 1:
            ref_frame = min(n_frames - 1, state.frame + 1)
        ref_positions = _foot_positions_at_frame(
            model, foot_scratch, motion, joint_qpos, ref_frame, anchor_root, foot_points
        )
        slip_segments: list[list[np.ndarray]] = []

        for foot_idx, point in enumerate(foot_points):
            pos = _foot_position(data, point)
            ref_pos = ref_positions[foot_idx]
            status, xy_speed = _foot_contact_status(
                pos=pos,
                ref_pos=ref_pos,
                fps=fps,
                floor_z=0.0,
                contact_height=float(args.foot_contact_height),
                penetration_depth=float(args.foot_penetration_depth),
                slip_speed=float(args.foot_slip_speed),
            )
            marker_pos = pos.copy()
            if status in ("contact", "slip", "penetration"):
                marker_pos[2] = max(0.0, min(float(pos[2]), float(args.foot_contact_height))) + 0.015
            for name, handle in marker_handles[foot_idx].items():
                handle.position = marker_pos
                handle.visible = name == status

            if status == "slip" and show_foot_slip_vectors[0]:
                start = np.array([ref_pos[0], ref_pos[1], 0.025], dtype=np.float32)
                end = np.array([pos[0], pos[1], 0.025], dtype=np.float32)
                if float(np.linalg.norm(end[:2] - start[:2])) > 1e-4:
                    direction = end - start
                    scale = min(3.0, max(1.0, xy_speed / max(float(args.foot_slip_speed), 1e-6)))
                    end = start + direction * scale
                    slip_segments.append([start, end])

        if slip_segments:
            foot_slip_handle = server.scene.add_line_segments(
                "/overlay/foot_slip_vectors",
                np.asarray(slip_segments, dtype=np.float32),
                colors=tuple(int(c) for c in args.foot_error_color),
                line_width=float(args.foot_slip_width),
            )

    def set_frame(frame: int, record_history: bool = True) -> None:
        nonlocal bone_handle
        raw_frame = int(frame)
        with lock:
            prev_frame = state.frame
            state.frame = raw_frame % n_frames
            if record_history:
                wrapped = (
                    raw_frame < 0 or raw_frame >= n_frames
                    or (state.frame < prev_frame and raw_frame > prev_frame)
                    or (state.frame > prev_frame and raw_frame < prev_frame)
                )
                if wrapped:
                    state.history.clear()
                state.history.append(state.frame)
                max_hist = max(2, max_ghost_frames * max(1, int(ghost_stride[0])) + 4)
                state.history = state.history[-max_hist:]

            _apply_motion_frame(model, data, motion, joint_qpos, state.frame, anchor_root)
            _update_primary(state.frame)

            if bone_handle is not None:
                bone_handle.remove()
                bone_handle = None
            if show_bones[0]:
                bone_handle = server.scene.add_line_segments(
                    "/overlay/bones", _body_line_segments(model, data),
                    colors=tuple(int(c) for c in args.bone_color),
                    line_width=float(args.bone_width),
                )
            update_ghosts()
            update_foot_contacts()
            frame_slider.value = state.frame

    def step(delta: int) -> None:
        set_frame(state.frame + int(delta))

    def remove_scene_handles() -> None:
        nonlocal bone_handle, trail_handle, foot_slip_handle
        for item in robot_handles:
            item.handle.remove()
        for item in ghost_sets:
            item.handle.remove()
        for states in marker_handles:
            for handle in states.values():
                handle.remove()
        if bone_handle is not None:
            bone_handle.remove()
        if trail_handle is not None:
            trail_handle.remove()
        if foot_slip_handle is not None:
            foot_slip_handle.remove()
        bone_handle = None
        trail_handle = None
        foot_slip_handle = None

    def rebuild_scene(robot_id: str, pkl_value: str, xml_value: str | None) -> None:
        nonlocal model, data, joint_qpos, motion, anchor_root, n_frames
        nonlocal robot_handles, ghost_data_slots, ghost_sets
        nonlocal foot_points, foot_scratch, marker_handles
        robot_id = robot_id.strip()
        if not robot_id:
            load_status.content = "Robot ID is empty."
            return
        try:
            new_pkl_path = _resolve_user_path(pkl_value.strip(), output_root)
            new_xml_path = _resolve_robot_xml(output_root, robot_id, xml_value.strip() if xml_value else None)
            new_motion = _load_motion(new_pkl_path)
            if new_motion.dof_pos.shape[0] == 0:
                raise ValueError(f"Motion has zero frames: {new_pkl_path}")
            new_model = mujoco.MjModel.from_xml_path(str(new_xml_path))
            new_data = mujoco.MjData(new_model)
            new_joint_qpos = _non_free_joint_qpos(new_model)
            new_anchor_root = np.zeros(3, dtype=np.float32)
            if args.anchor_root:
                new_anchor_root[:2] = new_motion.root_pos[0, :2]

            with lock:
                state.playing = False
                play_cb.value = False
                remove_scene_handles()

                model = new_model
                data = new_data
                joint_qpos = new_joint_qpos
                motion = new_motion
                anchor_root = new_anchor_root
                n_frames = int(motion.dof_pos.shape[0])
                state.frame = 0
                state.history.clear()
                _apply_motion_frame(model, data, motion, joint_qpos, state.frame, anchor_root)

                print(f"MuJoCo XML : {new_xml_path}")
                print(f"MuJoCo joints ({len(_mujoco_joint_names(model))}): {_mujoco_joint_names(model)}")
                robot_handles = _create_mujoco_render_handles(
                    server,
                    model,
                    data,
                    mesh_saturation=float(args.mesh_saturation),
                    mesh_value=float(args.mesh_value),
                )
                ghost_data_slots = [mujoco.MjData(model) for _ in range(max_ghost_frames)]
                ghost_sets = _create_mujoco_ghost_handles(
                    server,
                    model,
                    max_ghost_frames,
                    ghost_color,
                    ghost_opacity,
                    int(args.ghost_face_stride),
                )
                foot_points = _resolve_foot_points(model, args.foot_points)
                foot_scratch = mujoco.MjData(model)
                marker_handles = create_marker_handles(foot_points)
                if args.foot_contacts:
                    print(f"Foot markers: {[p.name for p in foot_points]}")

                load_status.content = (
                    f"Loaded `{robot_id}` · `{new_pkl_path.name}` · "
                    f"{n_frames} frames"
                )
                frame_slider.value = 0
                set_frame(0, record_history=False)
        except Exception as exc:
            load_status.content = f"Load failed: `{type(exc).__name__}: {exc}`"
            print(f"[load failed] {type(exc).__name__}: {exc}")

    # ── GUI callbacks ─────────────────────────────────────────────────────────

    @load_button.on_click
    def _(_e) -> None:
        rebuild_scene(str(robot_text.value), str(pkl_text.value), str(xml_text.value))

    @reset_button.on_click
    def _(_e) -> None:
        with lock:
            state.playing = False
            play_cb.value = False
            state.history.clear()
        set_frame(0, record_history=False)
        load_status.content = f"Reset `{robot_text.value}` · {n_frames} frames"

    @play_cb.on_update
    def _(_e) -> None:
        with lock:
            state.playing = bool(play_cb.value)

    @frame_slider.on_update
    def _(_e) -> None:
        if int(frame_slider.value) != state.frame:
            set_frame(int(frame_slider.value))

    @bone_cb.on_update
    def _(_e) -> None:
        show_bones[0] = bool(bone_cb.value)
        set_frame(state.frame, record_history=False)

    @ghost_cb.on_update
    def _(_e) -> None:
        show_ghosts[0] = bool(ghost_cb.value)
        if show_ghosts[0] and int(active_ghost_slider.value) == 0 and max_ghost_frames > 0:
            active_ghost_slider.value = max_ghost_frames
        set_frame(state.frame, record_history=False)

    @ghost_traj_cb.on_update
    def _(_e) -> None:
        show_ghost_traj[0] = bool(ghost_traj_cb.value)
        set_frame(state.frame, record_history=False)

    @foot_cb.on_update
    def _(_e) -> None:
        show_foot_contacts[0] = bool(foot_cb.value)
        set_frame(state.frame, record_history=False)

    @slip_cb.on_update
    def _(_e) -> None:
        show_foot_slip_vectors[0] = bool(slip_cb.value)
        set_frame(state.frame, record_history=False)

    @active_ghost_slider.on_update
    def _(_e) -> None:
        show_ghosts[0] = int(active_ghost_slider.value) > 0
        ghost_cb.value = show_ghosts[0]
        set_frame(state.frame, record_history=False)

    @ghost_stride_slider.on_update
    def _(_e) -> None:
        ghost_stride[0] = max(1, int(ghost_stride_slider.value))
        set_frame(state.frame, record_history=False)

    @prev_button.on_click
    def _(_e) -> None:
        with lock:
            state.playing = False; play_cb.value = False
        step(-1)

    @next_button.on_click
    def _(_e) -> None:
        with lock:
            state.playing = False; play_cb.value = False
        step(1)

    @toggle_cmd.on_trigger
    def _(_e) -> None:
        with lock:
            state.playing = not state.playing
            play_cb.value = state.playing

    @prev_cmd.on_trigger
    def _(_e) -> None:
        with lock:
            state.playing = False; play_cb.value = False
        step(-1)

    @next_cmd.on_trigger
    def _(_e) -> None:
        with lock:
            state.playing = False; play_cb.value = False
        step(1)

    # ── Initial camera ────────────────────────────────────────────────────────
    extent_xy = np.ptp(motion.root_pos[:, :2] - anchor_root[:2], axis=0) if n_frames > 1 else np.ones(2)
    cam_dist = float(max(2.4, np.linalg.norm(extent_xy) * 0.6 + 2.0))
    server.initial_camera.position = np.array([1.8, -cam_dist, 1.35])
    server.initial_camera.look_at = np.array([0.0, 0.0, 0.55])
    server.initial_camera.fov = np.deg2rad(34.0)

    set_frame(state.frame, record_history=False)

    print(f"PKL:    {pkl_path}")
    print(f"Frames: {n_frames}  fps: {float(args.fps or motion.fps):.1f}")
    print(f"Open:   http://{args.host}:{args.port}")
    print("Hotkeys: Space play/pause  |  p/n step frames")

    while True:
        t0 = time.time()
        with lock:
            playing = bool(state.playing)
            speed = float(speed_slider.value)
            base_fps = float(args.fps if args.fps is not None else motion.fps)
        if playing:
            step(1)
        dt = 1.0 / max(base_fps * max(speed, 1e-6), 1e-6)
        time.sleep(max(0.001, dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
