# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable Franka Panda velocity controller extracted from example_cloth_franka.py.

Provides Jacobian-based end-effector tracking with null-space posture control
and gripper finger actuation.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton import Model, State, eval_fk
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
    """Compute spatial error between current end-effector pose and target."""
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(
        pos_diff[0],
        pos_diff[1],
        pos_diff[2],
        ang_diff[0],
        ang_diff[1],
        ang_diff[2],
    )


class RobotController:
    """Jacobian-based velocity controller for a Franka Panda arm.

    Tracks waypoints via pseudo-inverse Jacobian control with a null-space
    posture regulator and gripper finger actuation.

    Args:
        model: Newton Model containing the robot.
        endeffector_id: Body index of the end-effector link.
        endeffector_offset: Transform offset from the EE link to the control point [m].
        bodies_per_world: Number of bodies per world (from the Franka builder).
        key_poses: Nx9 array — [duration, x, y, z, qw, qx, qy, qz, gripper].
            If None, uses the default cloth_franka folding trajectory.
    """

    # Default folding trajectory from example_cloth_franka.py
    CLAMP_CLOSE = 0.06
    CLAMP_OPEN = 0.8

    DEFAULT_KEY_POSES = np.array(
        [
            [2.5, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, 0.31, -0.60, 0.23, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, 0.26, -0.60, 0.26, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, 0.12, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [3, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1, -0.06, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, 0.15, -0.33, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [3, 0.15, -0.33, 0.21, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, 0.15, -0.33, 0.28, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [3, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1, -0.02, -0.33, 0.28, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.28, -0.60, 0.28, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.28, -0.60, 0.20, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, -0.18, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [3, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1, 0.05, -0.60, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [3, -0.18, -0.30, 0.205, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [3, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, -0.03, -0.30, 0.31, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.0, -0.21, 0.30, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
            [2, -0.0, -0.21, 0.20, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [2, -0.0, -0.21, 0.35, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1.5, -0.0, -0.30, 0.35, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1.5, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, CLAMP_CLOSE],
            [1, -0.0, -0.40, 0.35, 1, 0.0, 0.0, 0.0, CLAMP_OPEN],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        model: Model,
        endeffector_id: int,
        endeffector_offset: wp.transform,
        bodies_per_world: int,
        key_poses: np.ndarray | None = None,
    ):
        self.model = model
        self.endeffector_id = endeffector_id
        self.endeffector_offset = endeffector_offset
        self.bodies_per_world = bodies_per_world

        # Waypoint setup
        if key_poses is None:
            key_poses = self.DEFAULT_KEY_POSES.copy()
        self.key_poses = key_poses
        self.targets = self.key_poses[:, 1:]
        self.transition_duration = self.key_poses[:, 0]
        self.cumulative_time = np.cumsum(self.key_poses[:, 0])
        self.target = self.targets[0]

        # Jacobian workspace
        out_dim = 6
        in_dim = model.joint_dof_count

        self.Jacobian_one_hots = [
            wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float) for i in range(out_dim)
        ]

        self.target_joint_qd = wp.empty(model.joint_dof_count, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)

        self.temp_state = model.state(requires_grad=True)
        self.initial_pose = model.joint_q.numpy()

        # Build the body_out kernel with baked-in EE parameters
        ee_offset = self.endeffector_offset
        ee_id = self.endeffector_id

        @wp.kernel
        def _compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            mv = transform_twist(wp.static(ee_offset), body_qd[wp.static(ee_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self._compute_body_out_kernel = _compute_body_out

    def compute_body_jacobian(self, joint_q: wp.array, joint_qd: wp.array) -> None:
        """Compute the 6xN Jacobian mapping joint velocities to EE spatial velocity.

        Result is stored in self.J_flat (row-major, shape [6, joint_dof_count]).
        """
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        out_dim = 6

        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state)
            wp.launch(self._compute_body_out_kernel, 1, inputs=[self.temp_state.body_qd], outputs=[self.body_out])

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_joint_qd(self, state_in: State, sim_time: float) -> None:
        """Compute target joint velocities for the current timestep.

        Performs:
        1. Waypoint lookup via cumulative time.
        2. EE pose error computation.
        3. Jacobian pseudo-inverse + null-space posture regulation.
        4. Gripper finger control.

        Result is stored in self.target_joint_qd.
        """
        # Waypoint lookup (loop at end of trajectory)
        t_mod = sim_time if sim_time < self.cumulative_time[-1] else sim_time % self.cumulative_time[-1]
        current_interval = np.searchsorted(self.cumulative_time, t_mod)
        self.target = self.targets[current_interval]

        # EE pose error
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

        # Jacobian
        self.compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        # Null-space projector for posture regulation
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J
        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)

        # Primary + null-space
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Gripper finger control
        delta_q[-2] = self.target[-1] * 0.04 - q[-2]
        delta_q[-1] = self.target[-1] * 0.04 - q[-1]

        self.target_joint_qd.assign(delta_q)
