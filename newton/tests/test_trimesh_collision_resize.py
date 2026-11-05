# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""forKernels: grow-on-overflow tri-mesh detection (used by XPBD self-collision)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import TriMeshCollisionDetector
from newton.tests.unittest_utils import add_function_test, get_test_devices

QUERY_RADIUS = 0.35  # several grid cells, so each vertex sees many non-adjacent triangles


def _sheet_model(device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=12,
        dim_y=12,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    return builder.finalize(device=device)


def _vertex_triangle_sets(detector):
    info = detector.collision_info
    data = info.vertex_colliding_triangles.numpy()
    offsets = info.vertex_colliding_triangles_offsets.numpy()
    sizes = info.vertex_colliding_triangles_buffer_sizes.numpy()
    counts = info.vertex_colliding_triangles_count.numpy()
    return [
        frozenset(int(data[2 * (offsets[v] + i) + 1]) for i in range(min(counts[v], sizes[v])))
        for v in range(len(counts))
    ]


def test_vertex_triangle_resize_matches_ample_buffers(test, device):
    model = _sheet_model(device)
    ample = TriMeshCollisionDetector(
        model, init_collision_info=True, vertex_collision_buffer_pre_alloc=256, vertex_collision_buffer_max_alloc=256
    )
    small = TriMeshCollisionDetector(
        model, init_collision_info=True, vertex_collision_buffer_pre_alloc=2, vertex_collision_buffer_max_alloc=256
    )
    ample.vertex_triangle_collision_detection(QUERY_RADIUS)
    expected = _vertex_triangle_sets(ample)
    test.assertGreater(max(len(s) for s in expected), 2, "scene must overflow the small buffer")

    small.vertex_triangle_collision_detection(QUERY_RADIUS)
    test.assertNotEqual(_vertex_triangle_sets(small), expected, "fixed small buffers should truncate")

    # Each call doubles once (2 -> 4 -> ... -> 256), as a sim does over successive steps.
    for _ in range(8):
        small.resize_flags.zero_()
        small.vertex_triangle_collision_detection_with_resize(QUERY_RADIUS)

    test.assertEqual(_vertex_triangle_sets(small), expected)
    test.assertGreater(int(small.collision_info.vertex_colliding_triangles_buffer_sizes.numpy().max()), 2)


def test_edge_edge_resize_grows_buffers(test, device):
    model = _sheet_model(device)
    small = TriMeshCollisionDetector(
        model, init_collision_info=True, edge_collision_buffer_pre_alloc=2, edge_collision_buffer_max_alloc=256
    )
    for _ in range(8):
        small.resize_flags.zero_()
        small.edge_edge_collision_detection_with_resize(QUERY_RADIUS)
    info = small.collision_info
    sizes = info.edge_colliding_edges_buffer_sizes.numpy()
    counts = info.edge_colliding_edges_count.numpy()
    test.assertGreater(int(sizes.max()), 2)
    np.testing.assert_array_less(counts - 1, sizes)  # every edge's contacts now fit


class TestTriMeshCollisionResize(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestTriMeshCollisionResize,
    "test_vertex_triangle_resize_matches_ample_buffers",
    test_vertex_triangle_resize_matches_ample_buffers,
    devices=devices,
)
add_function_test(
    TestTriMeshCollisionResize,
    "test_edge_edge_resize_grows_buffers",
    test_edge_edge_resize_grows_buffers,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
