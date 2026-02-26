# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cloth Dataset Pickup
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free.
#
# Command: python -m newton.examples cloth_dataset_pickup
#
###########################################################################

from __future__ import annotations

import json
import os
import sys

import numpy as np
import warp as wp

if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
    _usd_root = os.environ.get("OPENUSD_ROOT") or os.environ.get("USD_ROOT")
    if _usd_root and os.path.isdir(_usd_root):
        for _subdir in ("bin", "lib"):
            _dll_dir = os.path.join(_usd_root, _subdir)
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)

try:
    from pxr import Usd, UsdGeom
except ImportError as e:
    Usd = UsdGeom = None
    _PXR_IMPORT_ERROR = e
else:
    _PXR_IMPORT_ERROR = None

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.utils import transform_twist


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


def compute_body_jacobian(
    model: Model,
    joint_q: wp.array,
    joint_qd: wp.array,
    body_id: int | str,  # Can be either body index or body name
    offset: wp.transform | None = None,
    velocity: bool = True,
    include_rotation: bool = False,
):
    if isinstance(body_id, str):
        body_id = model.body_name.get(body_id)
    if offset is None:
        offset = wp.transform_identity()

    joint_q.requires_grad = True
    joint_qd.requires_grad = True

    if velocity:

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = transform_twist(offset, body_qd[body_id])
            if wp.static(include_rotation):
                for i in range(6):
                    body_out[i] = mv[i]
            else:
                for i in range(3):
                    body_out[i] = mv[3 + i]

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
    else:

        @wp.kernel
        def compute_body_out(body_q: wp.array(dtype=wp.transform), body_out: wp.array(dtype=float)):
            tf = body_q[body_id] * offset
            if wp.static(include_rotation):
                for i in range(7):
                    body_out[i] = tf[i]
            else:
                for i in range(3):
                    body_out[i] = tf[i]

        in_dim = model.joint_coord_count
        out_dim = 7 if include_rotation else 3

    out_state = model.state(requires_grad=True)
    body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        eval_fk(model, joint_q, joint_qd, out_state)
        wp.launch(compute_body_out, 1, inputs=[out_state.body_qd if velocity else out_state.body_q], outputs=[body_out])

    def onehot(i):
        x = np.zeros(out_dim, dtype=np.float32)
        x[i] = 1.0
        return wp.array(x)

    J = np.empty((out_dim, in_dim), dtype=wp.float32)
    for i in range(out_dim):
        tape.backward(grads={body_out: onehot(i)})
        J[i] = joint_qd.grad.numpy() if velocity else joint_q.grad.numpy()
        tape.zero()
    return J.astype(np.float32)


def _find_first_mesh_prim(stage: Usd.Stage):
    if UsdGeom is None:
        raise ImportError("pxr.UsdGeom is unavailable") from _PXR_IMPORT_ERROR
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim
    return None


def _load_cloth_mesh_from_path(cloth_path: str, cloth_usd_prim_path: str | None = None):
    ext = os.path.splitext(cloth_path)[1].lower()
    if ext in {".usd", ".usda", ".usdc"}:
        if Usd is None:
            raise ImportError(
                "USD cloth loading requires pxr/OpenUSD. "
                "If using a local OpenUSD install on Windows, ensure its bin directory is on PATH."
            ) from _PXR_IMPORT_ERROR
        stage = Usd.Stage.Open(cloth_path)
        if stage is None:
            raise ValueError(f"Failed to open USD cloth file: {cloth_path}")
        if cloth_usd_prim_path:
            prim = stage.GetPrimAtPath(cloth_usd_prim_path)
            if not prim or not prim.IsValid():
                raise ValueError(f"USD cloth prim not found: {cloth_usd_prim_path}")
        else:
            prim = _find_first_mesh_prim(stage)
            if prim is None:
                raise ValueError(f"No UsdGeom.Mesh found in cloth file: {cloth_path}")
        cloth_mesh = newton.usd.get_mesh(prim)
        return [wp.vec3(v) for v in cloth_mesh.vertices], cloth_mesh.indices

    if ext == ".obj":
        try:
            import trimesh  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Loading OBJ cloth meshes requires trimesh. Install with `uv sync --extra examples`."
            ) from e

        mesh = trimesh.load(cloth_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError(f"OBJ file contains no geometry: {cloth_path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type {type(mesh)!r} for cloth file: {cloth_path}")
        if mesh.faces is None or len(mesh.faces) == 0:
            raise ValueError(f"OBJ file contains no faces: {cloth_path}")
        if mesh.faces.shape[1] != 3:
            raise ValueError(
                f"OBJ file must be triangulated for add_cloth_mesh(); found {mesh.faces.shape[1]} vertices/face: {cloth_path}"
            )

        vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in np.asarray(mesh.vertices)]
        indices = np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist()
        return vertices, indices

    raise ValueError(f"Unsupported cloth file extension: {ext}. Supported: .obj, .usd, .usda, .usdc")


def _load_robot_key_poses_from_json(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("robot_key_poses", data.get("key_poses", data))
    key_poses = np.asarray(data, dtype=np.float32)
    if key_poses.ndim != 2 or key_poses.shape[1] != 9:
        raise ValueError(f"Trajectory JSON must have shape [N, 9], got {key_poses.shape}")
    return key_poses


def _compute_vertex_bbox_extent(vertices: list[wp.vec3]) -> np.ndarray:
    verts = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    if verts.size == 0:
        raise ValueError("Cloth mesh has no vertices")
    return np.max(verts, axis=0) - np.min(verts, axis=0)


def _infer_cloth_scale_from_vertices(vertices: list[wp.vec3]) -> tuple[float, str, np.ndarray]:
    """Infer unit scale for garment-like meshes from raw bounding-box extent.

    Heuristic:
      - very large extents (hundreds+) are likely millimeters
      - medium extents (tens to low hundreds) are likely centimeters
      - small extents are assumed meters
    """
    bbox_extent = _compute_vertex_bbox_extent(vertices)
    max_dim = float(np.max(bbox_extent))

    if max_dim > 250.0:
        return 0.001, "mm", bbox_extent
    if max_dim > 3.0:
        return 0.01, "cm", bbox_extent
    return 1.0, "m", bbox_extent


# Reference center of the built-in unisex_shirt.usd mesh (in raw cm coordinates).
# External garment meshes are shifted to this center so that the standard cloth_franka
# transforms (scale=0.01, yaw=180°, pos=(0, 0.70, 0.28)) place them over the table.
_SHIRT_REF_CENTER = np.array([0.0, 120.0, -6.4], dtype=np.float32)


def _shift_to_shirt_space(vertices: list[wp.vec3]) -> list[wp.vec3]:
    """Shift vertices so their center matches the built-in shirt mesh center.

    This makes the standard cloth_franka transforms place external meshes
    in the same world-space region as the built-in shirt (over the table).
    """
    verts = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    mesh_center = 0.5 * (np.min(verts, axis=0) + np.max(verts, axis=0))
    shift = _SHIRT_REF_CENTER - mesh_center
    verts = verts + shift
    return [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]


def _recenter_cloth_vertices(vertices: list[wp.vec3], mode: str) -> tuple[list[wp.vec3], np.ndarray, np.ndarray]:
    """Recenter cloth vertices so placement args behave predictably.

    Args:
        vertices: Raw cloth vertices.
        mode: One of {"none", "bbox-center", "bbox-center-ground"}.

    Returns:
        Tuple of (recentered_vertices, bbox_min, bbox_max) in raw mesh units.
    """
    if not vertices:
        raise ValueError("Cloth mesh has no vertices")

    verts = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)

    if mode == "none":
        offset = np.zeros(3, dtype=np.float32)
    elif mode == "bbox-center":
        offset = bbox_center
    elif mode == "bbox-center-ground":
        offset = np.array([bbox_center[0], bbox_center[1], bbox_min[2]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported cloth recenter mode: {mode}")

    verts = verts - offset
    recentered = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    return recentered, bbox_min, bbox_max


def _rotate_vertices_y_up_to_z_up(vertices: list[wp.vec3]) -> list[wp.vec3]:
    """Rotate vertices from Y-up coordinate system to Z-up (Newton convention).

    Applies a -90 degree rotation around the X-axis: (x, y, z) -> (x, -z, y).
    """
    return [wp.vec3(float(v[0]), -float(v[2]), float(v[1])) for v in vertices]


def _detect_y_up_mesh(vertices: list[wp.vec3]) -> bool:
    """Detect if a mesh is in Y-up coordinates by checking which axis has the largest extent.

    Garment meshes from Y-up DCCs (Blender, Maya) have body height along Y.
    In Z-up (Newton), the tallest axis of a garment should be Z.
    If Y extent is significantly larger than Z extent, the mesh is likely Y-up.
    """
    verts = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    extent = np.max(verts, axis=0) - np.min(verts, axis=0)
    # Y is tallest and significantly taller than Z → likely Y-up garment
    return float(extent[1]) > float(extent[2]) * 1.5 and int(np.argmax(extent)) == 1


def _estimate_world_bbox_for_placed_cloth(
    vertices: list[wp.vec3],
    scale: float,
    yaw_deg: float,
    pos: wp.vec3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate world-space bbox after the add_cloth_mesh transform (yaw + scale + translation)."""
    verts = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
    if verts.size == 0:
        raise ValueError("Cloth mesh has no vertices")
    yaw = float(np.deg2rad(np.float32(yaw_deg)))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    verts_world = (verts * float(scale)) @ rot.T + np.array(
        [float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32
    )
    return np.min(verts_world, axis=0), np.max(verts_world, axis=0)


def _build_bbox_robot_key_poses(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    anchor_pos: np.ndarray,
    support_z: float,
    mode: str,
    clamp_open_activation_val: float,
    clamp_close_activation_val: float,
) -> np.ndarray:
    """Build a bbox-driven pickup/fold trajectory [N, 9].

    Generates multiple pick-drag-release cycles that fold the garment
    from its edges toward the center, similar to the hardcoded default
    trajectory but adapted to each garment's world-space bounding box.

    All targets are clamped to the Franka's reachable workspace:
        X: [-0.30, 0.31],  Y: [-0.60, -0.21],  Z: [support_z, support_z+0.20]
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    center = 0.5 * (bbox_min + bbox_max)

    # Franka reachable workspace (from the working default key poses)
    X_LO, X_HI = -0.30, 0.31
    Y_LO, Y_HI = -0.60, -0.21

    # Clamp the garment bbox to the reachable zone for pick targets
    reach_x_lo = float(np.clip(bbox_min[0], X_LO, X_HI))
    reach_x_hi = float(np.clip(bbox_max[0], X_LO, X_HI))
    reach_y_lo = float(np.clip(bbox_min[1], Y_LO, Y_HI))
    reach_y_hi = float(np.clip(bbox_max[1], Y_LO, Y_HI))
    reach_cx = 0.5 * (reach_x_lo + reach_x_hi)
    reach_cy = 0.5 * (reach_y_lo + reach_y_hi)

    # Z heights relative to table surface
    table_z = float(support_z)
    hover_z = float(np.clip(table_z + 0.15, 0.22, 0.40))
    down_z = float(np.clip(table_z + 0.01, 0.15, 0.30))
    grab_z = float(np.clip(table_z + 0.005, 0.15, 0.30))
    lift_z = float(np.clip(table_z + 0.12, 0.22, 0.38))
    place_z = float(np.clip(table_z + 0.03, 0.16, 0.34))

    grip_open = clamp_open_activation_val
    grip_close = clamp_close_activation_val
    quat = [1.0, 0.0, 0.0, 0.0]

    def kp(dur: float, x: float, y: float, z: float, grip: float) -> list[float]:
        return [dur, float(np.clip(x, X_LO, X_HI)), float(np.clip(y, Y_LO, Y_HI)), z, *quat, grip]

    def fold_cycle(pick_x: float, pick_y: float, place_x: float, place_y: float) -> list[list[float]]:
        """One pick-drag-release cycle: hover → down → grab → lift → move → place → release → hover."""
        return [
            kp(2.0, pick_x, pick_y, hover_z, grip_open),
            kp(1.5, pick_x, pick_y, down_z, grip_open),
            kp(1.0, pick_x, pick_y, grab_z, grip_close),
            kp(1.5, pick_x, pick_y, lift_z, grip_close),
            kp(2.0, place_x, place_y, lift_z, grip_close),
            kp(1.5, place_x, place_y, place_z, grip_close),
            kp(0.8, place_x, place_y, place_z, grip_open),
            kp(1.0, place_x, place_y, hover_z, grip_open),
        ]

    key_poses: list[list[float]] = []

    if mode == "bbox-fold":
        # Generate 4 fold cycles from the edges toward the center,
        # matching the default trajectory's pattern of: top-right, bottom-right,
        # top-left, bottom corners.
        # Fold 1: +X edge → center (right side toward middle)
        key_poses.extend(fold_cycle(reach_x_hi, reach_cy, reach_cx, reach_cy))
        # Fold 2: -X edge → center (left side toward middle)
        key_poses.extend(fold_cycle(reach_x_lo, reach_cy, reach_cx, reach_cy))
        # Fold 3: far Y edge → center (top toward middle)
        key_poses.extend(fold_cycle(reach_cx, reach_y_lo, reach_cx, reach_cy))
        # Fold 4: near Y edge → center (bottom toward middle)
        key_poses.extend(fold_cycle(reach_cx, reach_y_hi, reach_cx, reach_cy))
    else:
        # bbox-grasp: single pick-lift-release
        key_poses.extend(
            [
                kp(2.0, reach_cx, reach_cy, hover_z, grip_open),
                kp(1.5, reach_cx, reach_cy, down_z, grip_open),
                kp(1.0, reach_cx, reach_cy, grab_z, grip_close),
                kp(1.5, reach_cx, reach_cy, lift_z, grip_close),
                kp(0.8, reach_cx, reach_cy, lift_z, grip_open),
                kp(1.0, reach_cx, reach_cy, hover_z, grip_open),
            ]
        )

    return np.asarray(key_poses, dtype=np.float32)


class Example:
    def __init__(self, viewer, args=None):
        # parameters
        #   simulation
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = int(getattr(args, "sim_substeps", 15) if args is not None else 15)
        self.iterations = int(getattr(args, "iterations", 5) if args is not None else 5)
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        #   contact
        #       body-cloth contact
        self.cloth_particle_radius = float(getattr(args, "cloth_particle_radius", 0.008) if args is not None else 0.008)
        self.cloth_body_contact_margin = float(
            getattr(args, "cloth_body_contact_margin", 0.01) if args is not None else 0.01
        )
        #       self-contact
        self.particle_self_contact_radius = 0.002
        self.particle_self_contact_margin = 0.003

        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        self.robot_friction = 1.0
        self.table_friction = 1.0
        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6

        self.bending_ke = 1e-4
        self.bending_kd = 1e-3

        self.scene = ModelBuilder()
        self.soft_contact_max = 1000000

        self.viewer = viewer
        self.args = args

        self.seed = getattr(args, "seed", None) if args is not None else None
        if self.seed is not None:
            np.random.seed(self.seed)

        self.scene_usd_path = getattr(args, "scene_usd_path", None) if args is not None else None
        self.scene_root_path = getattr(args, "scene_root_path", "/") if args is not None else "/"
        self.scene_apply_up_axis = getattr(args, "scene_apply_up_axis", False) if args is not None else False
        self.scene_use_default_table = getattr(args, "scene_use_default_table", True) if args is not None else True
        self.platform_enable = getattr(args, "platform_enable", True) if args is not None else True
        # Match cloth_franka table: center=(0, -0.5, 0.1), half=(0.4, 0.4, 0.1), top_z=0.2
        platform_pos = getattr(args, "platform_pos", (0.0, -0.5, 0.1)) if args is not None else (0.0, -0.5, 0.1)
        platform_size = getattr(args, "platform_size", (0.8, 0.8, 0.2)) if args is not None else (0.8, 0.8, 0.2)
        self.platform_pos = np.asarray(platform_pos, dtype=np.float32)
        self.platform_size = np.asarray(platform_size, dtype=np.float32)
        platform_top_z_arg = getattr(args, "platform_top_z", None) if args is not None else None
        if platform_top_z_arg is not None:
            self.platform_pos[2] = float(platform_top_z_arg) - 0.5 * float(self.platform_size[2])
        self.platform_top_z = float(self.platform_pos[2] + 0.5 * self.platform_size[2])
        self.cloth_place_on_platform = getattr(args, "cloth_place_on_platform", True) if args is not None else True
        self.cloth_platform_clearance = float(
            getattr(args, "cloth_platform_clearance", 0.01) if args is not None else 0.01
        )

        self.cloth_path = getattr(args, "cloth_path", None) if args is not None else None
        self.cloth_usd_prim_path = getattr(args, "cloth_usd_prim_path", None) if args is not None else None
        self.cloth_scale = getattr(args, "cloth_scale", 0.01) if args is not None else 0.01
        self.cloth_scale_auto = getattr(args, "cloth_scale_auto", False) if args is not None else False
        self.cloth_center_mode = getattr(args, "cloth_center_mode", "auto") if args is not None else "auto"
        self.cloth_y_up = getattr(args, "cloth_y_up", "no") if args is not None else "no"
        self.cloth_density = getattr(args, "cloth_density", 0.2) if args is not None else 0.2
        cloth_pos = getattr(args, "cloth_pos", (0.0, 0.70, 0.28)) if args is not None else (0.0, 0.70, 0.28)
        cloth_vel = getattr(args, "cloth_vel", (0.0, 0.0, 0.0)) if args is not None else (0.0, 0.0, 0.0)
        self.cloth_pos = wp.vec3(*cloth_pos)
        self.cloth_vel = wp.vec3(*cloth_vel)
        self.cloth_yaw_deg = getattr(args, "cloth_yaw_deg", 180.0) if args is not None else 180.0

        self.trajectory_json_path = getattr(args, "trajectory_json", None) if args is not None else None
        self.robot_target_mode = getattr(args, "robot_target_mode", "default") if args is not None else "default"
        self.robot_start_delay = float(getattr(args, "robot_start_delay", 2.0) if args is not None else 2.0)
        # Match cloth_franka robot base: (-0.5, -0.5, -0.1)
        robot_base_pos = getattr(args, "robot_base_pos", (-0.5, -0.5, -0.1)) if args is not None else (-0.5, -0.5, -0.1)
        self.robot_base_pos = np.asarray(robot_base_pos, dtype=np.float32)
        robot_target_pos_offset = (
            getattr(args, "robot_target_pos_offset", (0.0, 0.0, 0.0)) if args is not None else (0.0, 0.0, 0.0)
        )
        self.robot_target_pos_offset = np.asarray(robot_target_pos_offset, dtype=np.float32)
        self.cloth_scale_inferred_units: str | None = None
        self.cloth_bbox_extent_raw: list[float] | None = None
        self.cloth_bbox_min_raw: list[float] | None = None
        self.cloth_bbox_max_raw: list[float] | None = None
        self.cloth_center_mode_applied: str | None = None
        self.cloth_bbox_min_world: list[float] | None = None
        self.cloth_bbox_max_world: list[float] | None = None
        self.robot_target_mode_applied = "default"
        self.robot_delay_park_target: np.ndarray | None = None

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        if self.scene_usd_path:
            self.scene.add_usd(
                self.scene_usd_path,
                floating=False,
                root_path=self.scene_root_path,
                apply_up_axis_from_stage=self.scene_apply_up_axis,
                verbose=False,
            )

        # --- Load cloth mesh (table is added after so it can auto-size to fit) ---
        if self.cloth_path:
            vertices, mesh_indices = _load_cloth_mesh_from_path(self.cloth_path, self.cloth_usd_prim_path)
        else:
            if Usd is None:
                raise ImportError(
                    "Default cloth asset loading requires pxr/OpenUSD because it reads a USD shirt mesh. "
                    "Pass --cloth-path <mesh.obj> to avoid USD cloth loading, or fix your pxr installation."
                ) from _PXR_IMPORT_ERROR
            usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
            usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
            shirt_mesh = newton.usd.get_mesh(usd_prim)
            vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
            mesh_indices = shirt_mesh.indices

        # Auto-scale: detect mesh units from bounding box
        bbox_extent = _compute_vertex_bbox_extent(vertices)
        self.cloth_bbox_extent_raw = [float(x) for x in bbox_extent]
        if self.cloth_scale_auto:
            inferred_scale, inferred_units, _ = _infer_cloth_scale_from_vertices(vertices)
            self.cloth_scale = inferred_scale
            self.cloth_scale_inferred_units = inferred_units

        # For external meshes, shift vertices to match the built-in shirt's
        # coordinate space so the standard transforms place them over the table.
        if self.cloth_path:
            vertices = _shift_to_shirt_space(vertices)

        # Record raw bbox (after shift, before add_cloth_mesh transform)
        verts_np = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in vertices], dtype=np.float32)
        self.cloth_bbox_min_raw = [float(x) for x in np.min(verts_np, axis=0)]
        self.cloth_bbox_max_raw = [float(x) for x in np.max(verts_np, axis=0)]

        # Compute world-space bbox after placement transforms
        bbox_min_world, bbox_max_world = _estimate_world_bbox_for_placed_cloth(
            vertices, scale=float(self.cloth_scale), yaw_deg=float(self.cloth_yaw_deg), pos=self.cloth_pos
        )

        # Auto-size the table to cover the garment's world-space footprint
        if self.platform_enable and self.cloth_path:
            pad = 0.08  # padding around garment [m]
            cloth_x_half = 0.5 * (bbox_max_world[0] - bbox_min_world[0]) + pad
            cloth_y_half = 0.5 * (bbox_max_world[1] - bbox_min_world[1]) + pad
            cloth_center_y = 0.5 * (bbox_min_world[1] + bbox_max_world[1])
            # Keep table X centered at 0, Y centered on the garment
            self.platform_pos[0] = 0.0
            self.platform_pos[1] = float(cloth_center_y)
            self.platform_size[0] = float(max(self.platform_size[0], 2.0 * cloth_x_half))
            self.platform_size[1] = float(max(self.platform_size[1], 2.0 * cloth_y_half))
            self.platform_top_z = float(self.platform_pos[2] + 0.5 * self.platform_size[2])
        self.cloth_bbox_min_world = [float(x) for x in bbox_min_world]
        self.cloth_bbox_max_world = [float(x) for x in bbox_max_world]

        # --- Add table/platform (after auto-sizing) ---
        if not self.scene_usd_path and self.scene_use_default_table and not self.platform_enable:
            self.scene.add_shape_box(
                -1,
                wp.transform(wp.vec3(0.0, -0.5, 0.1), wp.quat_identity()),
                hx=0.4,
                hy=0.4,
                hz=0.1,
            )
        if self.platform_enable:
            half_extents = 0.5 * self.platform_size
            self.scene.add_shape_box(
                -1,
                wp.transform(
                    wp.vec3(float(self.platform_pos[0]), float(self.platform_pos[1]), float(self.platform_pos[2])),
                    wp.quat_identity(),
                ),
                hx=float(half_extents[0]),
                hy=float(half_extents[1]),
                hz=float(half_extents[2]),
            )
            print(
                "Platform:",
                f"center={self.platform_pos.tolist()}",
                f"size={self.platform_size.tolist()}",
                f"top_z={self.platform_top_z:.4f}",
            )
        print("Cloth world bbox:", f"min={self.cloth_bbox_min_world}", f"max={self.cloth_bbox_max_world}")

        if self.add_cloth:
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(
                    wp.vec3(0.0, 0.0, 1.0),
                    float(np.deg2rad(np.float32(self.cloth_yaw_deg))),
                ),
                pos=self.cloth_pos,
                vel=self.cloth_vel,
                density=float(self.cloth_density),
                scale=float(self.cloth_scale),
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        if (
            self.add_robot
            and self.trajectory_json_path is None
            and self.robot_target_mode in {"bbox-grasp", "bbox-fold"}
        ):
            self._configure_bbox_robot_targets()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.soft_contact_ke
        shape_kd[...] = self.soft_contact_kd
        shape_mu[...] = 1.0

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            # initialize cloth solver
            #   set edge rest angle to zero to disable bending, this is currently a workaround to make SolverVBD stable
            #   TODO: fix SolverVBD's bending issue
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=32,
                particle_edge_contact_buffer_size=64,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.soft_contact_ke,
            )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # create Warp arrays for gravity so we can swap Model.gravity during
        # a simulation running under CUDA graph capture
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)  # used for the robot solver
        # gravity in cm/s^2
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)  # used for the cloth solver

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        if self.add_cloth:
            self.capture()

    def set_up_control(self):
        self.control = self.model.control()

        # we are controlling the velocity
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        # for robot control
        self.delta_q = wp.empty(self.model.joint_count, dtype=float)
        self.joint_q_des = wp.array(self.model.joint_q.numpy(), dtype=float)

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            # TODO verify transform twist
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,  # unit: cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.06
        clamp_open_activation_val = 0.8
        self.gripper_close_activation_val = clamp_close_activation_val
        self.gripper_open_activation_val = clamp_open_activation_val

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position, 4D quaternion), gripper open (1) or closed (0)
                # # top left
                [2.5, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.26, -0.60, 0.26, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.12, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [2, 0.15, -0.33, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.15, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # top left
                [2, -0.28, -0.60, 0.28, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.18, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # # bottom left
                [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, -0.0, -0.21, 0.30, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.0, -0.21, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1.5, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, clamp_close_activation_val],
                [1, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        if self.trajectory_json_path:
            self.robot_key_poses = _load_robot_key_poses_from_json(self.trajectory_json_path)
        self._set_robot_key_poses(self.robot_key_poses)
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                0.22,
            ],
            wp.quat_identity(),
        )

    def _set_robot_key_poses(self, key_poses: np.ndarray) -> None:
        key_poses = np.asarray(key_poses, dtype=np.float32).copy()
        key_poses[:, 1:4] += self.robot_target_pos_offset
        self.robot_key_poses = key_poses
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

    def _configure_bbox_robot_targets(self) -> None:
        if self.cloth_bbox_min_world is None or self.cloth_bbox_max_world is None:
            return
        bbox_min = np.asarray(self.cloth_bbox_min_world, dtype=np.float32)
        bbox_max = np.asarray(self.cloth_bbox_max_world, dtype=np.float32)
        key_poses = _build_bbox_robot_key_poses(
            bbox_min,
            bbox_max,
            np.array([float(self.cloth_pos[0]), float(self.cloth_pos[1]), float(self.cloth_pos[2])], dtype=np.float32),
            self.platform_top_z if self.platform_enable else float(self.cloth_pos[2]),
            self.robot_target_mode,
            self.gripper_open_activation_val,
            self.gripper_close_activation_val,
        )
        self._set_robot_key_poses(key_poses)
        self.robot_target_mode_applied = self.robot_target_mode
        support_z = self.platform_top_z if self.platform_enable else float(self.cloth_pos[2])
        # Park far from the cloth during settle so the garment can fall onto the platform.
        park_x = 0.28
        park_y = -0.62
        park_z = float(np.clip(support_z + 0.30, 0.32, 0.52))
        self.robot_delay_park_target = np.array(
            [park_x, park_y, park_z, 1.0, 0.0, 0.0, 0.0, self.gripper_open_activation_val],
            dtype=np.float32,
        )
        print(
            "Robot targets:",
            f"mode={self.robot_target_mode_applied}",
            f"bbox_min={self.cloth_bbox_min_world}",
            f"bbox_max={self.cloth_bbox_max_world}",
            f"poses={int(self.robot_key_poses.shape[0])}",
        )
        print(
            "Robot park target:",
            f"target={self.robot_delay_park_target.tolist() if self.robot_delay_park_target is not None else None}",
        )

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        """
        Compute the Jacobian of the end effector's velocity related to joint_q

        """

        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(
        self,
        state_in: State,
    ):
        delay_active = self.robot_start_delay > 0.0 and self.sim_time < self.robot_start_delay
        t_mod = (
            self.sim_time
            if self.sim_time < self.robot_key_poses_time[-1]
            else self.sim_time % self.robot_key_poses_time[-1]
        )
        include_rotation = True
        current_interval = np.searchsorted(self.robot_key_poses_time, t_mod)
        if delay_active and self.robot_delay_park_target is not None:
            self.target = self.robot_delay_park_target
        else:
            self.target = self.targets[current_interval]

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        # 2. Compute null-space projector
        #    I is size [num_joints x num_joints]
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()

        # 3. Define a desired "elbow-up" reference posture
        #    (For example, one that keeps joint 2 or 3 above a certain angle.)
        #    Adjust indices and angles to your robot's kinematics.
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]  # e.g., set elbow joint around 1 rad to keep it up

        # 4. Define a null-space velocity term pulling joints toward q_des
        #    K_null is a small gain so it doesn't override main task
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        # 5. Combine primary task and null-space controller
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Apply gripper finger control
        delta_q[-2] = self.target[-1] * 0.04 - q[-2]
        delta_q[-1] = self.target[-1] * 0.04 - q[-1]

        self.target_joint_qd.assign(delta_q)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply interactive viewer forces (no wind is applied by default in this example)
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.36, -0.95, -0.05)
        p_upper = wp.vec3(0.36, 0.05, 0.56)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )

    def write_run_metadata(self, args) -> None:
        metadata_path = getattr(args, "metadata_path", None)
        if metadata_path is None and getattr(args, "viewer", None) == "usd" and getattr(args, "output_path", None):
            metadata_path = f"{args.output_path}.json"
        if not metadata_path:
            return

        metadata = {
            "example": "cloth_dataset_pickup",
            "scene_usd_path": self.scene_usd_path,
            "scene_root_path": self.scene_root_path,
            "cloth_path": self.cloth_path,
            "cloth_usd_prim_path": self.cloth_usd_prim_path,
            "cloth_scale": float(self.cloth_scale),
            "cloth_scale_auto": bool(self.cloth_scale_auto),
            "cloth_scale_inferred_units": self.cloth_scale_inferred_units,
            "cloth_center_mode": self.cloth_center_mode,
            "cloth_center_mode_applied": self.cloth_center_mode_applied,
            "cloth_density": float(self.cloth_density),
            "cloth_bbox_min_raw": self.cloth_bbox_min_raw,
            "cloth_bbox_max_raw": self.cloth_bbox_max_raw,
            "cloth_bbox_extent_raw": self.cloth_bbox_extent_raw,
            "cloth_bbox_min_world": self.cloth_bbox_min_world,
            "cloth_bbox_max_world": self.cloth_bbox_max_world,
            "cloth_pos": [float(self.cloth_pos[0]), float(self.cloth_pos[1]), float(self.cloth_pos[2])],
            "cloth_vel": [float(self.cloth_vel[0]), float(self.cloth_vel[1]), float(self.cloth_vel[2])],
            "cloth_yaw_deg": float(self.cloth_yaw_deg),
            "cloth_particle_radius": float(self.cloth_particle_radius),
            "cloth_body_contact_margin": float(self.cloth_body_contact_margin),
            "sim_substeps": int(self.sim_substeps),
            "iterations": int(self.iterations),
            "platform_enable": bool(self.platform_enable),
            "platform_pos": self.platform_pos.tolist(),
            "platform_size": self.platform_size.tolist(),
            "platform_top_z": float(self.platform_top_z),
            "cloth_place_on_platform": bool(self.cloth_place_on_platform),
            "cloth_platform_clearance": float(self.cloth_platform_clearance),
            "robot_target_mode": self.robot_target_mode,
            "robot_target_mode_applied": self.robot_target_mode_applied,
            "robot_start_delay": float(self.robot_start_delay),
            "robot_base_pos": self.robot_base_pos.tolist(),
            "robot_target_pos_offset": self.robot_target_pos_offset.tolist(),
            "trajectory_json": self.trajectory_json_path,
            "seed": self.seed,
            "num_frames": getattr(args, "num_frames", None),
            "output_path": getattr(args, "output_path", None),
            "particle_count": int(self.model.particle_count),
            "body_count": int(self.model.body_count),
        }
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    import argparse

    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    parser.add_argument("--scene-usd-path", type=str, default=None, help="Path to a furniture/scene USD(.usd/.usda).")
    parser.add_argument(
        "--scene-root-path",
        type=str,
        default="/",
        help="Root prim path to import from the scene USD (used with --scene-usd-path).",
    )
    parser.add_argument(
        "--scene-apply-up-axis",
        action="store_true",
        default=False,
        help="Apply the scene stage up-axis transform when importing the scene USD.",
    )
    parser.add_argument(
        "--scene-use-default-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the built-in table collider when no scene USD is provided.",
    )
    parser.add_argument(
        "--platform-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a simple collision platform/table box (enabled by default, including when a scene USD is loaded).",
    )
    parser.add_argument(
        "--platform-pos",
        type=float,
        nargs=3,
        default=(0.0, -0.5, 0.1),
        metavar=("X", "Y", "Z"),
        help="Collision platform center position [m]. Matches cloth_franka table by default.",
    )
    parser.add_argument(
        "--platform-size",
        type=float,
        nargs=3,
        default=(0.8, 0.8, 0.2),
        metavar=("SX", "SY", "SZ"),
        help="Collision platform full size [m]. Matches cloth_franka table by default.",
    )
    parser.add_argument(
        "--platform-top-z",
        type=float,
        default=None,
        help="Collision platform top surface height [m]. Overrides the Z component of --platform-pos.",
    )
    parser.add_argument(
        "--cloth-place-on-platform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-shift the cloth in +Z so its bbox rests just above the platform surface.",
    )
    parser.add_argument(
        "--cloth-platform-clearance",
        type=float,
        default=0.01,
        help="Vertical clearance [m] above the platform top when auto-placing cloth.",
    )
    parser.add_argument("--cloth-path", type=str, default=None, help="Path to cloth mesh (.obj or USD).")
    parser.add_argument("--cloth-usd-prim-path", type=str, default=None, help="Prim path for cloth mesh inside USD.")
    parser.add_argument("--cloth-scale", type=float, default=0.01, help="Uniform cloth mesh scale factor.")
    parser.add_argument(
        "--cloth-scale-auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Infer cloth units from raw mesh bounding box and override --cloth-scale (heuristic: m/cm/mm).",
    )
    parser.add_argument(
        "--cloth-center-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "bbox-center", "bbox-center-ground"],
        help="Recenter raw cloth mesh before placement. 'auto' uses bbox-center-ground for external cloth and none for built-in shirt.",
    )
    parser.add_argument(
        "--cloth-y-up",
        type=str,
        default="no",
        choices=["auto", "yes", "no"],
        help="Rotate mesh from Y-up (Blender/Maya) to Z-up (Newton). "
        "'auto' detects from bounding box. 'no' keeps the mesh flat (for folding simulations).",
    )
    parser.add_argument("--cloth-density", type=float, default=0.2, help="Cloth areal density.")
    parser.add_argument(
        "--cloth-pos",
        type=float,
        nargs=3,
        default=(0.0, 0.70, 0.28),
        metavar=("X", "Y", "Z"),
        help="Initial cloth position.",
    )
    parser.add_argument(
        "--cloth-vel",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
        help="Initial cloth velocity.",
    )
    parser.add_argument("--cloth-yaw-deg", type=float, default=180.0, help="Initial cloth yaw rotation in degrees.")
    parser.add_argument(
        "--cloth-particle-radius",
        type=float,
        default=0.008,
        help="Cloth particle collision radius [m]; increase for more robust gripper/cloth contact.",
    )
    parser.add_argument(
        "--cloth-body-contact-margin",
        type=float,
        default=0.01,
        help="Soft cloth-body contact margin [m]; increase if the gripper tunnels through the cloth.",
    )
    parser.add_argument("--sim-substeps", type=int, default=15, help="Physics substeps per rendered frame.")
    parser.add_argument("--iterations", type=int, default=5, help="VBD solver iterations per substep.")
    parser.add_argument(
        "--trajectory-json",
        type=str,
        default=None,
        help="Optional path to robot keypose trajectory JSON with shape [N,9].",
    )
    parser.add_argument(
        "--robot-target-mode",
        type=str,
        default="default",
        choices=["default", "bbox-grasp", "bbox-fold"],
        help="Robot target generation mode. bbox-* modes derive pickup/fold targets from the cloth world bounding box.",
    )
    parser.add_argument(
        "--robot-base-pos",
        type=float,
        nargs=3,
        default=(-0.5, -0.5, -0.1),
        metavar=("X", "Y", "Z"),
        help="Franka base position [m]. Matches cloth_franka example by default.",
    )
    parser.add_argument(
        "--robot-start-delay",
        type=float,
        default=2.0,
        help="Delay robot motion [s] so the cloth can fall/settle before the trajectory starts.",
    )
    parser.add_argument(
        "--robot-target-pos-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="World-space offset added to all robot keypose targets to align the pickup motion with the cloth.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for numpy-based setup.")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional sidecar metadata JSON path (default: <output-path>.json for USD viewer).",
    )
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
    example.write_run_metadata(args)
