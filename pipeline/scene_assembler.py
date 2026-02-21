# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble a Newton simulation scene from a declarative SceneConfig.

Mirrors the setup logic from example_cloth_franka.py, extracting it into
a reusable ``SceneAssembler`` that returns an ``AssembledScene`` ready for
stepping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import CollisionPipeline, ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD

from .robot_control import RobotController
from .scene_config import ElementType, SceneConfig


@dataclass
class AssembledScene:
    """All Newton runtime objects for a single scene."""

    model: newton.Model
    state_0: newton.State
    state_1: newton.State
    control: newton.Control
    contacts: newton.Contacts
    robot_solver: SolverFeatherstone
    cloth_solver: SolverVBD | None
    collision_pipeline: CollisionPipeline
    robot_controller: RobotController
    gravity_zero: wp.array
    gravity_earth: wp.array
    has_cloth: bool
    has_robot: bool


class SceneAssembler:
    """Build an ``AssembledScene`` from a ``SceneConfig``."""

    def assemble(self, config: SceneConfig) -> AssembledScene:
        """Convert a declarative config into live Newton objects.

        Returns:
            An ``AssembledScene`` ready for simulation stepping.
        """
        scene = ModelBuilder()
        has_cloth = any(e.element_type == ElementType.CLOTH_USD for e in config.elements)
        has_robot = True  # M1 always includes a Franka

        # --- Robot ---
        franka_builder = ModelBuilder()
        asset_path = newton.utils.download_asset("franka_emika_panda")
        franka_builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                config.franka.position,
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # Apply initial joint angles
        for i, q_val in enumerate(config.franka.initial_joint_q):
            franka_builder.joint_q[i] = q_val

        bodies_per_world = franka_builder.body_count
        endeffector_id = franka_builder.body_count - 3

        scene.add_world(franka_builder)

        # --- Table ---
        env = config.environment
        scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(*env.table_position),
                wp.quat_identity(),
            ),
            hx=env.table_half_extents[0],
            hy=env.table_half_extents[1],
            hz=env.table_half_extents[2],
        )

        # --- Cloth elements ---
        for elem in config.elements:
            if elem.element_type != ElementType.CLOTH_USD:
                continue
            self._add_cloth_element(scene, elem)

        if has_cloth:
            scene.color()

        # --- Ground plane ---
        if env.add_ground_plane:
            scene.add_ground_plane()

        # --- Finalize model ---
        model = scene.finalize(requires_grad=False)

        sim = config.sim
        model.soft_contact_ke = sim.soft_contact_ke
        model.soft_contact_kd = sim.soft_contact_kd
        model.soft_contact_mu = sim.self_contact_friction

        # Set uniform shape materials
        shape_ke = model.shape_material_ke.numpy()
        shape_kd = model.shape_material_kd.numpy()
        shape_mu = model.shape_material_mu.numpy()
        shape_ke[...] = sim.soft_contact_ke
        shape_kd[...] = sim.soft_contact_kd
        shape_mu[...] = 1.0
        model.shape_material_ke = wp.array(
            shape_ke, dtype=model.shape_material_ke.dtype, device=model.shape_material_ke.device
        )
        model.shape_material_kd = wp.array(
            shape_kd, dtype=model.shape_material_kd.dtype, device=model.shape_material_kd.device
        )
        model.shape_material_mu = wp.array(
            shape_mu, dtype=model.shape_material_mu.dtype, device=model.shape_material_mu.device
        )

        # --- States & control ---
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # --- Collision pipeline ---
        collision_pipeline = CollisionPipeline(model, soft_contact_margin=sim.cloth_body_contact_margin)
        contacts = collision_pipeline.contacts()

        # --- Robot solver ---
        robot_solver = SolverFeatherstone(model, update_mass_matrix_interval=sim.sim_substeps)

        # --- Cloth solver ---
        cloth_solver = None
        if has_cloth:
            model.edge_rest_angle.zero_()
            cloth_solver = SolverVBD(
                model,
                iterations=sim.solver_iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_radius=sim.particle_self_contact_radius,
                particle_self_contact_margin=sim.particle_self_contact_margin,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=32,
                particle_edge_contact_buffer_size=64,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=sim.soft_contact_ke,
            )

        # --- Gravity arrays ---
        gravity_zero = wp.zeros(1, dtype=wp.vec3)
        gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        # --- FK init ---
        eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # --- Robot controller ---
        ee_offset = wp.transform(config.franka.endeffector_offset, wp.quat_identity())
        key_poses_np = np.array(config.franka.key_poses, dtype=np.float32) if config.franka.key_poses else None
        robot_controller = RobotController(
            model=model,
            endeffector_id=endeffector_id,
            endeffector_offset=ee_offset,
            bodies_per_world=bodies_per_world,
            key_poses=key_poses_np,
        )

        return AssembledScene(
            model=model,
            state_0=state_0,
            state_1=state_1,
            control=control,
            contacts=contacts,
            robot_solver=robot_solver,
            cloth_solver=cloth_solver,
            collision_pipeline=collision_pipeline,
            robot_controller=robot_controller,
            gravity_zero=gravity_zero,
            gravity_earth=gravity_earth,
            has_cloth=has_cloth,
            has_robot=has_robot,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cloth_element(scene: ModelBuilder, elem) -> None:
        """Load a USD garment mesh and add it as cloth.

        Supports both single-mesh prims (e.g. Newton's built-in shirt) and
        multi-prim Marvelous Designer exports where each pattern panel is a
        separate UsdGeom.Mesh under a parent Xform.
        """
        from pxr import Usd, UsdGeom

        usd_stage = Usd.Stage.Open(elem.asset_path)
        usd_prim = usd_stage.GetPrimAtPath(elem.prim_path)
        if not usd_prim.IsValid():
            raise ValueError(f"USD prim not found: {elem.prim_path} in {elem.asset_path}")

        if usd_prim.IsA(UsdGeom.Mesh):
            # Single mesh prim — original path
            mesh = newton.usd.get_mesh(usd_prim)
            vertices = mesh.vertices
            indices = mesh.indices
        else:
            # Multi-prim (e.g. MD export): traverse descendants for all Mesh prims
            mesh_prims = [p for p in Usd.PrimRange(usd_prim) if p.IsA(UsdGeom.Mesh)]
            if not mesh_prims:
                raise ValueError(
                    f"No UsdGeom.Mesh prims found under {elem.prim_path} in {elem.asset_path}"
                )
            all_vertices = []
            all_indices = []
            index_offset = 0
            for mesh_prim in mesh_prims:
                mesh = newton.usd.get_mesh(mesh_prim)
                all_vertices.append(mesh.vertices)
                all_indices.append(mesh.indices + index_offset)
                index_offset += len(mesh.vertices)
            vertices = np.concatenate(all_vertices, axis=0)
            indices = np.concatenate(all_indices)
            print(f"  Merged {len(mesh_prims)} mesh panels from {elem.prim_path}")

        vert_list = [wp.vec3(v) for v in vertices]

        cp = elem.cloth_params
        scene.add_cloth_mesh(
            vertices=vert_list,
            indices=indices,
            rot=wp.quat_from_axis_angle(wp.vec3(*elem.rotation_axis), elem.rotation_angle),
            pos=wp.vec3(*elem.position),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=cp.density,
            scale=elem.scale,
            tri_ke=cp.tri_ke,
            tri_ka=cp.tri_ka,
            tri_kd=cp.tri_kd,
            edge_ke=cp.bending_ke,
            edge_kd=cp.bending_kd,
            particle_radius=cp.particle_radius,
        )


def simulate_frame(scene: AssembledScene, sim_dt: float, substeps: int) -> None:
    """Advance the simulation by one output frame (multiple substeps).

    Mirrors the dual-solver loop from example_cloth_franka.py:
    1. Rebuild BVH for cloth self-collision.
    2. Per substep:
       a. Clear forces, step robot solver with gravity disabled and particles hidden.
       b. Run collision pipeline.
       c. Step cloth solver.
       d. Swap states.

    Args:
        scene: An assembled scene from ``SceneAssembler.assemble()``.
        sim_dt: Timestep per substep [s].
        substeps: Number of substeps per frame.
    """
    if scene.cloth_solver is not None:
        scene.cloth_solver.rebuild_bvh(scene.state_0)

    for _ in range(substeps):
        scene.state_0.clear_forces()
        scene.state_1.clear_forces()

        if scene.has_robot:
            particle_count = scene.model.particle_count
            scene.model.particle_count = 0
            scene.model.gravity.assign(scene.gravity_zero)
            scene.model.shape_contact_pair_count = 0

            scene.state_0.joint_qd.assign(scene.robot_controller.target_joint_qd)
            scene.robot_solver.step(scene.state_0, scene.state_1, scene.control, None, sim_dt)

            scene.state_0.particle_f.zero_()
            scene.model.particle_count = particle_count
            scene.model.gravity.assign(scene.gravity_earth)

        # Collision detection
        scene.collision_pipeline.collide(scene.state_0, scene.contacts)

        # Cloth solver
        if scene.cloth_solver is not None:
            scene.cloth_solver.step(scene.state_0, scene.state_1, scene.control, scene.contacts, sim_dt)

        scene.state_0, scene.state_1 = scene.state_1, scene.state_0
