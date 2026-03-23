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

"""XPBD mesh self-collision support.

Provides position-correction kernels for vertex-triangle and edge-edge
self-contacts, integrated with the shared TriMeshCollisionDetector.
"""

import warp as wp

from ...geometry import ParticleFlags
from ...geometry.tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
    set_to_csr,
)
from ...sim import Model


@wp.kernel
def solve_vertex_triangle_self_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    contact_radius: float,
    relaxation: float,
    # outputs
    particle_deltas: wp.array(dtype=wp.vec3),
):
    """XPBD position correction for vertex-triangle self-contacts."""
    v_index = wp.tid()

    flags_v = particle_flags[v_index]
    if flags_v & int(ParticleFlags.ACTIVE) == 0:
        return

    collision_info = collision_info_array[0]
    num_contacts = get_vertex_colliding_triangles_count(collision_info, v_index)

    for c in range(num_contacts):
        tri_index = get_vertex_colliding_triangles(collision_info, v_index, c)
        if tri_index < 0:
            continue

        ti = tri_indices[tri_index, 0]
        tj = tri_indices[tri_index, 1]
        tk = tri_indices[tri_index, 2]

        p = particle_q[v_index]
        a = particle_q[ti]
        b = particle_q[tj]
        c_pos = particle_q[tk]

        # Compute closest point on triangle (barycentric)
        e1 = b - a
        e2 = c_pos - a
        vp = p - a

        d11 = wp.dot(e1, e1)
        d12 = wp.dot(e1, e2)
        d22 = wp.dot(e2, e2)
        dp1 = wp.dot(vp, e1)
        dp2 = wp.dot(vp, e2)

        denom = d11 * d22 - d12 * d12
        if wp.abs(denom) < 1e-20:
            continue

        inv_denom = 1.0 / denom
        u = (d22 * dp1 - d12 * dp2) * inv_denom
        v = (d11 * dp2 - d12 * dp1) * inv_denom

        # Clamp to triangle
        u = wp.clamp(u, 0.0, 1.0)
        v = wp.clamp(v, 0.0, 1.0)
        if u + v > 1.0:
            s = u + v
            u = u / s
            v = v / s

        closest = a + u * e1 + v * e2
        diff = p - closest
        dist = wp.length(diff)

        if dist < contact_radius and dist > 1e-8:
            normal = diff / dist
            err = contact_radius - dist

            # Inverse mass weighting
            w_v = particle_inv_mass[v_index]
            w_a = particle_inv_mass[ti] * (1.0 - u - v) * (1.0 - u - v)
            w_b = particle_inv_mass[tj] * u * u
            w_c = particle_inv_mass[tk] * v * v
            w_sum = w_v + w_a + w_b + w_c

            if w_sum > 1e-20:
                delta = normal * err * relaxation / w_sum

                wp.atomic_add(particle_deltas, v_index, delta * particle_inv_mass[v_index])
                wp.atomic_add(particle_deltas, ti, -delta * particle_inv_mass[ti] * (1.0 - u - v))
                wp.atomic_add(particle_deltas, tj, -delta * particle_inv_mass[tj] * u)
                wp.atomic_add(particle_deltas, tk, -delta * particle_inv_mass[tk] * v)


@wp.kernel
def solve_edge_edge_self_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    contact_radius: float,
    edge_parallel_epsilon: float,
    relaxation: float,
    # outputs
    particle_deltas: wp.array(dtype=wp.vec3),
):
    """XPBD position correction for edge-edge self-contacts."""
    e_index = wp.tid()

    collision_info = collision_info_array[0]
    num_contacts = get_edge_colliding_edges_count(collision_info, e_index)

    e1_v0 = edge_indices[e_index, 2]
    e1_v1 = edge_indices[e_index, 3]

    for c_idx in range(num_contacts):
        e2_index = get_edge_colliding_edges(collision_info, e_index, c_idx)
        if e2_index < 0:
            continue

        e2_v0 = edge_indices[e2_index, 2]
        e2_v1 = edge_indices[e2_index, 3]

        p0 = particle_q[e1_v0]
        p1 = particle_q[e1_v1]
        q0 = particle_q[e2_v0]
        q1 = particle_q[e2_v1]

        st = wp.closest_point_edge_edge(p0, p1, q0, q1, edge_parallel_epsilon)
        s = st[0]
        t = st[1]

        if s < 0.0 or s > 1.0 or t < 0.0 or t > 1.0:
            continue

        c1 = p0 + s * (p1 - p0)
        c2 = q0 + t * (q1 - q0)
        diff = c1 - c2
        dist = wp.length(diff)

        if dist < contact_radius and dist > 1e-8:
            normal = diff / dist
            err = contact_radius - dist

            # Inverse mass weighting using barycentric coords
            w_p0 = particle_inv_mass[e1_v0] * (1.0 - s) * (1.0 - s)
            w_p1 = particle_inv_mass[e1_v1] * s * s
            w_q0 = particle_inv_mass[e2_v0] * (1.0 - t) * (1.0 - t)
            w_q1 = particle_inv_mass[e2_v1] * t * t
            w_sum = w_p0 + w_p1 + w_q0 + w_q1

            if w_sum > 1e-20:
                delta = normal * err * relaxation / w_sum

                wp.atomic_add(particle_deltas, e1_v0, delta * particle_inv_mass[e1_v0] * (1.0 - s))
                wp.atomic_add(particle_deltas, e1_v1, delta * particle_inv_mass[e1_v1] * s)
                wp.atomic_add(particle_deltas, e2_v0, -delta * particle_inv_mass[e2_v0] * (1.0 - t))
                wp.atomic_add(particle_deltas, e2_v1, -delta * particle_inv_mass[e2_v1] * t)


class XPBDSelfCollision:
    """Manages self-collision detection and constraint resolution for XPBD solver."""

    def __init__(
        self,
        model: Model,
        contact_radius: float = 0.01,
        contact_margin: float = 0.02,
        edge_parallel_epsilon: float = 1e-5,
        topological_filter_threshold: int = 2,
        vertex_buffer_size: int = 8,
        edge_buffer_size: int = 8,
    ):
        self.model = model
        self.contact_radius = contact_radius
        self.contact_margin = contact_margin
        self.edge_parallel_epsilon = edge_parallel_epsilon
        self.device = model.device

        # Build topological filtering lists
        vertex_tri_filter_list = None
        vertex_tri_filter_offsets = None
        edge_filter_list = None
        edge_filter_offsets = None

        if topological_filter_threshold > 1 and hasattr(model, "particle_adjacency") and model.particle_adjacency:
            adjacency = model.particle_adjacency
            v_tri_sets = build_vertex_n_ring_tris_collision_filter(
                topological_filter_threshold,
                model.particle_count,
                model.edge_indices.numpy(),
                adjacency.v_adj_edges.numpy(),
                adjacency.v_adj_edges_offsets.numpy(),
                adjacency.v_adj_faces.numpy(),
                adjacency.v_adj_faces_offsets.numpy(),
            )
            if v_tri_sets is not None:
                flat, offs = set_to_csr(v_tri_sets)
                vertex_tri_filter_list = wp.array(flat, dtype=wp.int32, device=self.device)
                vertex_tri_filter_offsets = wp.array(offs, dtype=wp.int32, device=self.device)

            edge_sets = build_edge_n_ring_edge_collision_filter(
                topological_filter_threshold,
                model.edge_indices.numpy(),
                adjacency.v_adj_edges.numpy(),
                adjacency.v_adj_edges_offsets.numpy(),
            )
            if edge_sets is not None:
                flat, offs = set_to_csr(edge_sets)
                edge_filter_list = wp.array(flat, dtype=wp.int32, device=self.device)
                edge_filter_offsets = wp.array(offs, dtype=wp.int32, device=self.device)

        # Create the shared detector
        self.detector = TriMeshCollisionDetector(
            model,
            vertex_collision_buffer_pre_alloc=vertex_buffer_size,
            edge_collision_buffer_pre_alloc=edge_buffer_size,
            vertex_triangle_filtering_list=vertex_tri_filter_list,
            vertex_triangle_filtering_list_offsets=vertex_tri_filter_offsets,
            edge_filtering_list=edge_filter_list,
            edge_filtering_list_offsets=edge_filter_offsets,
            edge_edge_parallel_epsilon=edge_parallel_epsilon,
        )

        self.collision_info_array = wp.array(
            [self.detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
        )

    def detect(self, particle_q: wp.array):
        """Refit BVH and run V-F + E-E detection."""
        self.detector.refit(particle_q)
        self.detector.vertex_triangle_collision_detection_with_resize(self.contact_margin)
        self.detector.edge_edge_collision_detection_with_resize(self.contact_margin)
        # Update collision info reference in case buffers were resized
        self.collision_info_array = wp.array(
            [self.detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
        )

    def solve(
        self,
        particle_q: wp.array,
        particle_inv_mass: wp.array,
        particle_flags: wp.array,
        relaxation: float,
        particle_deltas: wp.array,
    ):
        """Launch V-F and E-E position-correction kernels."""
        model = self.model

        if model.tri_count > 0:
            wp.launch(
                kernel=solve_vertex_triangle_self_contacts,
                dim=model.particle_count,
                inputs=[
                    particle_q,
                    particle_inv_mass,
                    particle_flags,
                    model.tri_indices,
                    self.collision_info_array,
                    self.contact_radius,
                    relaxation,
                ],
                outputs=[particle_deltas],
                device=self.device,
            )

        if model.edge_count > 0:
            wp.launch(
                kernel=solve_edge_edge_self_contacts,
                dim=model.edge_count,
                inputs=[
                    particle_q,
                    particle_inv_mass,
                    particle_flags,
                    model.edge_indices,
                    self.collision_info_array,
                    self.contact_radius,
                    self.edge_parallel_epsilon,
                    relaxation,
                ],
                outputs=[particle_deltas],
                device=self.device,
            )
