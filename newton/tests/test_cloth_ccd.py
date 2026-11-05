# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""forKernels cloth self-collision: VBD's CCD safety net and XPBD self-collision actually run."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD, SolverXPBD
from newton.tests.unittest_utils import add_function_test, get_test_devices

GAP = 0.1  # sheet B starts this far above sheet A
MARGIN = 0.01


def _two_sheets(device, b_vel=(0.0, 0.0, 0.0)):
    """Static 8x8 sheet A at z=0 and a free 4x4 sheet B at z=GAP inside A's footprint.

    A is massless (pinned like add_cloth_grid's fix_* edges), so B below z=0 can only mean it went
    through A. B is offset so its vertices land inside A's triangles, never on an edge or diagonal.
    """
    builder = newton.ModelBuilder()
    common = {"rot": wp.quat_identity(), "cell_x": 0.1, "cell_y": 0.1, "particle_radius": 0.005}
    builder.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0), dim_x=8, dim_y=8, mass=0.0, **common)
    n_a = builder.particle_count
    builder.add_cloth_grid(pos=wp.vec3(0.22, 0.26, GAP), vel=wp.vec3(*b_vel), dim_x=4, dim_y=4, mass=0.1, **common)
    builder.color()
    return builder.finalize(device=device), n_a


def _vbd(model, **kwargs):
    return SolverVBD(
        model,
        iterations=5,
        particle_enable_self_contact=True,
        particle_enable_ccd=True,
        particle_self_contact_margin=MARGIN,
        particle_self_contact_gap=MARGIN,
        **kwargs,
    )


def test_ccd_safety_net_stops_tunneling(test, device):
    """B is moved straight through A in one step; the safety net must stop it above A."""
    model, n_a = _two_sheets(device)
    solver = _vbd(model)

    q0 = model.particle_q.numpy()
    q1 = q0.copy()
    q1[n_a:, 2] -= 3.0 * GAP  # ends 0.2 below A: a full tunnel
    solver.particle_q_prev.assign(q0)
    state = model.state()
    state.particle_q.assign(q1)

    solver._run_ccd_safety_net(state)

    q = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    np.testing.assert_allclose(q[:n_a], q0[:n_a])  # A did not move, so it is untouched
    test.assertGreater(q[n_a:, 2].min(), 0.0, "sheet B tunneled through sheet A")
    test.assertLess(q[n_a:, 2].max(), GAP, "sheet B was not moved at all")


def _drop_b(test, device, use_pipeline):
    """B is thrown at A fast enough to cross the gap in one step; it must end above A.

    Runs the CCD pass inside real steps on both detector sources: with a CollisionPipeline the
    pipeline owns the tri-mesh detector, otherwise VBD does. VBD's conservative bound alone also
    holds B here, so test_ccd_safety_net_stops_tunneling is the one that proves the clamp itself.
    """
    model, n_a = _two_sheets(device, b_vel=(0.0, 0.0, -15.0))
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn") if use_pipeline else None
    solver = _vbd(model, collision_pipeline=pipeline)
    s0, s1 = model.state(), model.state()
    for _ in range(5):
        solver.step(s0, s1, None, None, 1.0e-2)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertGreater(q[n_a:, 2].min(), 0.0, "sheet B tunneled through sheet A")


def test_vbd_tri_tri_contact_steps(test, device):
    model, _ = _two_sheets(device, b_vel=(0.0, 0.0, -1.0))
    solver = _vbd(model, particle_enable_tri_tri_contact=True)
    s0, s1 = model.state(), model.state()
    for _ in range(3):
        solver.step(s0, s1, None, None, 1.0e-2)
        s0, s1 = s1, s0
    test.assertTrue(np.isfinite(s0.particle_q.numpy()).all())


def test_xpbd_self_collision_steps(test, device):
    model, n_a = _two_sheets(device, b_vel=(0.0, 0.0, -1.0))
    solver = SolverXPBD(model, iterations=5, enable_self_collision=True, self_collision_radius=MARGIN)
    test.assertIsNotNone(solver._self_collision)
    s0, s1 = model.state(), model.state()
    contacts = model.contacts()
    for _ in range(10):
        solver.step(s0, s1, None, contacts, 1.0e-2)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertGreater(q[n_a:, 2].min(), 0.0, "sheet B passed through sheet A")


class TestClothCCD(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _fn in [
    ("test_ccd_safety_net_stops_tunneling", test_ccd_safety_net_stops_tunneling),
    ("test_ccd_vbd_step_own_detector", lambda test, device: _drop_b(test, device, use_pipeline=False)),
    ("test_ccd_vbd_step_pipeline_detector", lambda test, device: _drop_b(test, device, use_pipeline=True)),
    ("test_vbd_tri_tri_contact_steps", test_vbd_tri_tri_contact_steps),
    ("test_xpbd_self_collision_steps", test_xpbd_self_collision_steps),
]:
    add_function_test(TestClothCCD, _name, _fn, devices=devices)

if __name__ == "__main__":
    unittest.main()
