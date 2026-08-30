# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverMuJoCo(sdf_shapes=...) - concave mesh colliders that keep their cavities.

A MuJoCo mesh geom collides as its CONVEX HULL, so a concave collider loses
every cavity it has. Exporting the shape as an SDF geom instead keeps them,
because MuJoCo samples the mesh's own octree.

The fixture is two slabs authored as ONE mesh with a gap between them. It is
chosen because it CANNOT be satisfied both ways: a hull spans the gap, the
mesh does not, and the two outcomes are far apart.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

TOP = 0.6           # top face of both slabs
GAP = 0.4           # the hole a hull spans and a mesh does not
DROP = 1.2          # where the prop starts
SOLID = GAP / 2 + 0.5   # x of a slab's centre


def _two_slabs():
    """Two boxes, top face at TOP, separated by GAP in x, as a single mesh."""
    verts, tris = [], []
    for cx in (-SOLID, +SOLID):
        base = len(verts)
        for z in (TOP - 0.4, TOP):
            for y in (-1.0, 1.0):
                for x in (cx - 0.5, cx + 0.5):
                    verts.append([x, y, z])
        face = [0, 2, 1, 1, 2, 3,  4, 5, 6, 5, 7, 6,  0, 1, 5, 0, 5, 4,
                2, 6, 7, 2, 7, 3,  0, 4, 6, 0, 6, 2,  1, 3, 7, 1, 7, 5]
        tris += [base + i for i in face]
    v = np.array(verts, dtype=np.float32)
    t = np.array(tris, dtype=np.int32)
    # Winding is asserted, not assumed: an inverted collider and a failing
    # collider look identical, and signed volume needs no interior point (the
    # centroid of two SEPARATED slabs lies in the gap, inside neither).
    p, q, r = v[t[0::3]], v[t[1::3]], v[t[2::3]]
    vol = np.einsum("ij,ij->i", p.astype(np.float64), np.cross(q, r)).sum() / 6.0
    assert vol > 0.0, f"fixture is wound inward (signed volume {vol:+.4f})"
    return v, t


class TestSolverMuJoCoSdfMesh(unittest.TestCase):
    def setUp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

    @staticmethod
    def _settle(use_sdf, px, frames=140, substeps=8, fps=60):
        v, t = _two_slabs()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        floor = builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, t), cfg=cfg)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(px, 0.0, DROP), wp.quat_identity())
        )
        builder.add_shape_box(body=body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        model = builder.finalize()

        solver = SolverMuJoCo(model, **({"sdf_shapes": [floor]} if use_sdf else {}))
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / fps / substeps
        for _ in range(frames):
            for _ in range(substeps):
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
        return float(state_0.body_q.numpy()[0][2]), solver

    def test_sdf_shapes_exports_an_sdf_geom(self):
        import mujoco

        _z, solver = self._settle(True, SOLID, frames=1)
        types = np.asarray(solver.mj_model.geom_type)
        self.assertIn(int(mujoco.mjtGeom.mjGEOM_SDF), [int(g) for g in types])
        self.assertTrue(bool(solver.mjw_model.has_sdf_geom))

    def test_a_mesh_geom_spans_a_cavity_it_does_not_have(self):
        """The defect this exists to fix. Not a bug in MuJoCo - a mesh geom
        collides as its convex hull, by design - but it is why a concave
        collider cannot be expressed as one."""
        z, _ = self._settle(False, 0.0)
        self.assertGreater(z, TOP - 0.2, "hull behaviour changed; fixture is stale")

    def test_an_sdf_geom_keeps_the_cavity(self):
        z, _ = self._settle(True, 0.0)
        self.assertLess(z, TOP - 0.2, "prop did not fall through the gap")

    def test_an_sdf_geom_still_collides_with_solid_material(self):
        """THE DISCRIMINATOR, and the reason the test above is not enough.

        An SDF geom that collides with NOTHING AT ALL also 'falls through the
        gap' and is indistinguishable there. This drops the prop over the
        middle of a slab, where it must rest at the same height the mesh geom
        holds it.

        It also guards the mesh-recentring compensation specifically. MuJoCo
        recentres a mesh asset and stores the correction in geom_pos/geom_quat;
        update_geom_properties_kernel composes it back in. Applying that for
        GEOM_TYPE_MESH alone - which is what it did - left the octree in the
        wrong frame and inverted the field: the prop hung in mid air over the
        cavity and fell through the solid part.
        """
        z_sdf, _ = self._settle(True, SOLID)
        z_mesh, _ = self._settle(False, SOLID)
        self.assertGreater(z_sdf, TOP - 0.2, "SDF geom collided with nothing")
        self.assertAlmostEqual(z_sdf, z_mesh, delta=0.02)


if __name__ == "__main__":
    unittest.main()
