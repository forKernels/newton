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
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    contact_radius: float,
    # outputs
    contact_deltas: wp.array[wp.vec3],
    contact_counts: wp.array[wp.int32],
):
    """XPBD vertex-triangle self-contact: keep a vertex ``contact_radius`` off each triangle it projects
    into, on the side it started the step on (``particle_q_prev``).

    The side comes from the start of the step because an unsigned distance cannot tell a vertex that
    crossed the triangle from one that did not, and pushes the crossed one further through. Only
    projections inside the triangle are handled; contacts beside it belong to edge-edge. Corrections
    accumulate into ``contact_deltas`` and ``contact_counts`` for :func:`apply_averaged_self_contact_deltas`.
    """
    v_index = wp.tid()

    flags_v = particle_flags[v_index]
    if flags_v & int(ParticleFlags.ACTIVE) == 0:
        return

    collision_info = collision_info_array[0]
    num_contacts = get_vertex_colliding_triangles_count(collision_info, v_index)
    p = particle_q[v_index]
    p_prev = particle_q_prev[v_index]

    for c in range(num_contacts):
        tri_index = get_vertex_colliding_triangles(collision_info, v_index, c)
        if tri_index < 0:
            continue

        ti = tri_indices[tri_index, 0]
        tj = tri_indices[tri_index, 1]
        tk = tri_indices[tri_index, 2]

        a = particle_q[ti]
        e1 = particle_q[tj] - a
        e2 = particle_q[tk] - a
        n = wp.cross(e1, e2)
        n_len = wp.length(n)
        if n_len < 1.0e-12:
            continue
        n = n / n_len

        # Barycentric coordinates of p's projection onto the triangle's plane
        vp = p - a
        d11 = wp.dot(e1, e1)
        d12 = wp.dot(e1, e2)
        d22 = wp.dot(e2, e2)
        dp1 = wp.dot(vp, e1)
        dp2 = wp.dot(vp, e2)
        inv_denom = 1.0 / (d11 * d22 - d12 * d12)
        u = (d22 * dp1 - d12 * dp2) * inv_denom
        v = (d11 * dp2 - d12 * dp1) * inv_denom
        if u < 0.0 or v < 0.0 or u + v > 1.0:
            continue

        a_prev = particle_q_prev[ti]
        n_prev = wp.cross(particle_q_prev[tj] - a_prev, particle_q_prev[tk] - a_prev)
        if wp.dot(p_prev - a_prev, n_prev) < 0.0:
            n = -n

        dist = wp.dot(vp, n)  # signed: negative once the vertex is past the triangle
        if dist >= contact_radius:
            continue
        err = contact_radius - dist

        w = 1.0 - u - v
        w_sum = (
            particle_inv_mass[v_index]
            + particle_inv_mass[ti] * w * w
            + particle_inv_mass[tj] * u * u
            + particle_inv_mass[tk] * v * v
        )
        if w_sum <= 1.0e-20:
            continue
        delta = n * (err / w_sum)

        wp.atomic_add(contact_deltas, v_index, delta * particle_inv_mass[v_index])
        wp.atomic_add(contact_deltas, ti, -delta * particle_inv_mass[ti] * w)
        wp.atomic_add(contact_deltas, tj, -delta * particle_inv_mass[tj] * u)
        wp.atomic_add(contact_deltas, tk, -delta * particle_inv_mass[tk] * v)
        wp.atomic_add(contact_counts, v_index, 1)
        wp.atomic_add(contact_counts, ti, 1)
        wp.atomic_add(contact_counts, tj, 1)
        wp.atomic_add(contact_counts, tk, 1)


@wp.kernel
def solve_edge_edge_self_contacts(
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    contact_radius: float,
    edge_parallel_epsilon: float,
    # outputs
    contact_deltas: wp.array[wp.vec3],
    contact_counts: wp.array[wp.int32],
):
    """XPBD edge-edge self-contact: keep the closest points ``contact_radius`` apart, along the
    direction they were separated in at the start of the step (see the vertex-triangle kernel)."""
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

        diff = (p0 + s * (p1 - p0)) - (q0 + t * (q1 - q0))
        n = (particle_q_prev[e1_v0] + s * (particle_q_prev[e1_v1] - particle_q_prev[e1_v0])) - (
            particle_q_prev[e2_v0] + t * (particle_q_prev[e2_v1] - particle_q_prev[e2_v0])
        )
        if wp.length(n) < 1.0e-8:
            n = diff
        n_len = wp.length(n)
        if n_len < 1.0e-8:
            continue
        n = n / n_len

        dist = wp.dot(diff, n)  # signed: negative once the edges have passed each other
        if dist >= contact_radius:
            continue
        err = contact_radius - dist

        w_p0 = particle_inv_mass[e1_v0] * (1.0 - s) * (1.0 - s)
        w_p1 = particle_inv_mass[e1_v1] * s * s
        w_q0 = particle_inv_mass[e2_v0] * (1.0 - t) * (1.0 - t)
        w_q1 = particle_inv_mass[e2_v1] * t * t
        w_sum = w_p0 + w_p1 + w_q0 + w_q1
        if w_sum <= 1.0e-20:
            continue
        delta = n * (err / w_sum)

        wp.atomic_add(contact_deltas, e1_v0, delta * particle_inv_mass[e1_v0] * (1.0 - s))
        wp.atomic_add(contact_deltas, e1_v1, delta * particle_inv_mass[e1_v1] * s)
        wp.atomic_add(contact_deltas, e2_v0, -delta * particle_inv_mass[e2_v0] * (1.0 - t))
        wp.atomic_add(contact_deltas, e2_v1, -delta * particle_inv_mass[e2_v1] * t)
        wp.atomic_add(contact_counts, e1_v0, 1)
        wp.atomic_add(contact_counts, e1_v1, 1)
        wp.atomic_add(contact_counts, e2_v0, 1)
        wp.atomic_add(contact_counts, e2_v1, 1)


@wp.kernel
def apply_averaged_self_contact_deltas(
    contact_deltas: wp.array[wp.vec3],
    contact_counts: wp.array[wp.int32],
    relaxation: float,
    # outputs
    particle_deltas: wp.array[wp.vec3],
):
    """Add each particle's self-contact correction averaged over its contacts.

    Summing them overshoots: a vertex near a sheet touches several triangles and edges at once, and XPBD
    turns the excess displacement into velocity, which launched cloth at ~10 m/s on contact.
    """
    i = wp.tid()
    count = contact_counts[i]
    if count > 0:
        particle_deltas[i] = particle_deltas[i] + contact_deltas[i] * (relaxation / float(count))


class XPBDSelfCollision:
    """Manages self-collision detection and constraint resolution for XPBD solver.

    shortcut: detection is discrete, so each step must move particles less than
    ``contact_margin - contact_radius``. Newton's usual 1/600 s cloth substep (Isaac's default) is
    stable; 1/100 s crumples a landing sheet. Add swept detection (geometry/ccd.py) for coarse steps.
    """

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
            init_collision_info=True,  # 1.6.1: the detector only builds its result struct on request
        )

        self.collision_info_array = wp.array(
            [self.detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
        )

        # Per-iteration accumulators, averaged into the solver's deltas
        self.contact_deltas = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.contact_counts = wp.zeros(model.particle_count, dtype=wp.int32, device=self.device)

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
        particle_q_prev: wp.array,
        particle_inv_mass: wp.array,
        particle_flags: wp.array,
        relaxation: float,
        particle_deltas: wp.array,
    ):
        """Launch the V-F and E-E correction kernels and add their averaged result to ``particle_deltas``.

        ``particle_q_prev`` is the start-of-step positions, which decide each contact's side.
        """
        model = self.model
        self.contact_deltas.zero_()
        self.contact_counts.zero_()

        if model.tri_count > 0:
            wp.launch(
                kernel=solve_vertex_triangle_self_contacts,
                dim=model.particle_count,
                inputs=[
                    particle_q,
                    particle_q_prev,
                    particle_inv_mass,
                    particle_flags,
                    model.tri_indices,
                    self.collision_info_array,
                    self.contact_radius,
                ],
                outputs=[self.contact_deltas, self.contact_counts],
                device=self.device,
            )

        if model.edge_count > 0:
            wp.launch(
                kernel=solve_edge_edge_self_contacts,
                dim=model.edge_count,
                inputs=[
                    particle_q,
                    particle_q_prev,
                    particle_inv_mass,
                    particle_flags,
                    model.edge_indices,
                    self.collision_info_array,
                    self.contact_radius,
                    self.edge_parallel_epsilon,
                ],
                outputs=[self.contact_deltas, self.contact_counts],
                device=self.device,
            )

        wp.launch(
            kernel=apply_averaged_self_contact_deltas,
            dim=model.particle_count,
            inputs=[self.contact_deltas, self.contact_counts, relaxation],
            outputs=[particle_deltas],
            device=self.device,
        )
