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


def _sheets(device, a, b, b_vel=(0.0, 0.0, 0.0)):
    """Static sheet A at z=0 and a free sheet B at z=GAP; ``a``/``b`` are (x, y, dim_x, dim_y, cell_x, cell_y).

    A is massless (pinned like add_cloth_grid's fix_* edges), so B below A can only mean it went through.
    """
    builder = newton.ModelBuilder()
    for (x, y, nx, ny, cx, cy), z, vel, mass in ((a, 0.0, (0.0, 0.0, 0.0), 0.0), (b, GAP, b_vel, 0.1)):
        builder.add_cloth_grid(
            pos=wp.vec3(x, y, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(*vel),
            dim_x=nx,
            dim_y=ny,
            cell_x=cx,
            cell_y=cy,
            mass=mass,
            particle_radius=0.005,
        )
        if mass == 0.0:
            n_a = builder.particle_count
    builder.color()
    return builder.finalize(device=device), n_a


def _two_sheets(device, b_vel=(0.0, 0.0, 0.0)):
    """8x8 sheet A and a 4x4 sheet B inside its footprint, offset so B's vertices land inside A's
    triangles, never on an edge or diagonal."""
    return _sheets(device, (0.0, 0.0, 8, 8, 0.1, 0.1), (0.22, 0.26, 4, 4, 0.1, 0.1), b_vel)


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


def _tunnel_b_through_a(model, n_a):
    """Move B straight down 3*GAP in one step (a full tunnel), run only the safety net, return (q0, q)."""
    solver = _vbd(model)
    q0 = model.particle_q.numpy()
    q1 = q0.copy()
    q1[n_a:, 2] -= 3.0 * GAP
    solver.particle_q_prev.assign(q0)
    state = model.state()
    state.particle_q.assign(q1)
    solver._run_ccd_safety_net(state)
    q = state.particle_q.numpy()
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q[:n_a], q0[:n_a])  # A did not move, so clamping it is a no-op
    return q0, q


def test_ccd_safety_net_stops_tunneling(test, device):
    """B's vertices pass through A's triangles: vertex-triangle CCD must stop them above A."""
    model, n_a = _two_sheets(device)
    _, q = _tunnel_b_through_a(model, n_a)
    test.assertGreater(q[n_a:, 2].min(), 0.0, "sheet B tunneled through sheet A")
    test.assertLess(q[n_a:, 2].max(), GAP, "sheet B was not moved at all")


def test_ccd_edge_edge_stops_tunneling(test, device):
    """Two strips cross like a plus sign: no vertex is over the other strip, so only edge-edge CCD sees it."""
    model, n_a = _sheets(device, (0.0, 0.0, 1, 1, 1.0, 0.1), (0.45, -0.5, 1, 1, 0.1, 1.1))
    _, q = _tunnel_b_through_a(model, n_a)
    test.assertGreater(q[n_a:, 2].min(), 0.0, "strip B passed through strip A edge-first")


def _height_of(q0, q, tris, xy):
    """z of the deformed triangle (positions q) whose starting footprint (q0) covers point xy."""
    for tri in tris:
        a, b, c = q0[tri, :2]
        u, v = np.linalg.solve(np.array([b - a, c - a]).T, xy - a)
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            return (1.0 - u - v) * q[tri[0], 2] + u * q[tri[1], 2] + v * q[tri[2], 2]
    raise AssertionError(f"no triangle covers {xy}")


def test_ccd_clamps_the_triangle_side(test, device):
    """A tiny static patch sits inside one of B's triangles, clear of its edges: only A's vertices see a
    hit, so B's triangle has to be clamped through them."""
    model, n_a = _sheets(device, (0.24, 0.29, 1, 1, 0.02, 0.02), (0.0, 0.0, 4, 4, 0.2, 0.2))
    q0, q = _tunnel_b_through_a(model, n_a)
    tris = model.tri_indices.numpy()
    b_tris = tris[tris.min(axis=1) >= n_a]
    z = _height_of(q0, q, b_tris, q0[:n_a, :2].mean(axis=0))
    test.assertGreater(z, 0.0, "B's triangle passed through A's vertices")


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


def test_xpbd_self_collision_lands_without_launch(test, device):
    """B is dropped, then thrown, onto static A under XPBD for 1 s at Newton's usual cloth substep: it
    must come to rest on A, never going through it or rising above where it started. Contact used to
    fling it tens of metres either way."""
    for b_vz in (0.0, -3.0):
        model, n_a = _two_sheets(device, b_vel=(0.0, 0.0, b_vz))
        solver = SolverXPBD(model, iterations=5, enable_self_collision=True, self_collision_radius=MARGIN)
        test.assertIsNotNone(solver._self_collision)
        s0, s1, contacts = model.state(), model.state(), model.contacts()
        z_lo, z_hi = GAP, GAP
        for _ in range(600):
            solver.step(s0, s1, None, contacts, 1.0 / 600.0)
            s0, s1 = s1, s0
            z = s0.particle_q.numpy()[n_a:, 2]
            z_lo, z_hi = min(z_lo, z.min()), max(z_hi, z.max())
        test.assertTrue(np.isfinite(s0.particle_q.numpy()).all())
        test.assertGreater(z_lo, 0.0, f"v0={b_vz}: sheet B passed through sheet A")
        test.assertLessEqual(z_hi, GAP + 1.0e-3, f"v0={b_vz}: sheet B was launched upward")
        test.assertLess(z.max(), 2.0 * MARGIN, f"v0={b_vz}: sheet B did not come to rest on A")


class TestClothCCD(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _fn in [
    ("test_ccd_safety_net_stops_tunneling", test_ccd_safety_net_stops_tunneling),
    ("test_ccd_edge_edge_stops_tunneling", test_ccd_edge_edge_stops_tunneling),
    ("test_ccd_clamps_the_triangle_side", test_ccd_clamps_the_triangle_side),
    ("test_ccd_vbd_step_own_detector", lambda test, device: _drop_b(test, device, use_pipeline=False)),
    ("test_ccd_vbd_step_pipeline_detector", lambda test, device: _drop_b(test, device, use_pipeline=True)),
    ("test_vbd_tri_tri_contact_steps", test_vbd_tri_tri_contact_steps),
    ("test_xpbd_self_collision_lands_without_launch", test_xpbd_self_collision_lands_without_launch),
]:
    add_function_test(TestClothCCD, _name, _fn, devices=devices)

if __name__ == "__main__":
    unittest.main()
