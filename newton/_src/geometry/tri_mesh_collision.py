# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..utils.mesh import MeshAdjacency

if TYPE_CHECKING:
    # Deferred: this module lives under ``geometry``, which ``sim`` imports at load
    # time. An eager ``sim`` import here would close that cycle. Only used as an
    # annotation, which ``from __future__ import annotations`` keeps lazy.
    from ..sim import Model
from .bvh import compute_bvh_group_roots
from .kernels import (
    compute_edge_aabbs,
    compute_edge_groups,
    compute_tri_aabbs,
    compute_tri_groups,
    edge_colliding_edges_detection_kernel,
    init_triangle_collision_data_kernel,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)


@wp.struct
class TriMeshCollisionInfo:
    # size: 2 x sum(vertex_colliding_triangles_buffer_sizes)
    # every two elements records the vertex index and a triangle index it collides to
    vertex_colliding_triangles: wp.array[wp.int32]
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32]
    vertex_colliding_triangles_count: wp.array[wp.int32]
    vertex_colliding_triangles_min_dist: wp.array[float]

    triangle_colliding_vertices: wp.array[wp.int32]
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32]
    triangle_colliding_vertices_count: wp.array[wp.int32]
    triangle_colliding_vertices_min_dist: wp.array[float]

    # size: 2 x sum(edge_colliding_edges_buffer_sizes)
    # every two elements records the edge index and an edge index it collides to
    edge_colliding_edges: wp.array[wp.int32]
    edge_colliding_edges_offsets: wp.array[wp.int32]
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32]
    edge_colliding_edges_count: wp.array[wp.int32]
    edge_colliding_edges_min_dist: wp.array[float]


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(col_info.vertex_colliding_triangles_count[v], col_info.vertex_colliding_triangles_buffer_sizes[v])


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_triangle_colliding_vertices_count(col_info: TriMeshCollisionInfo, tri: int):
    return wp.min(
        col_info.triangle_colliding_vertices_count[tri], col_info.triangle_colliding_vertices_buffer_sizes[tri]
    )


@wp.func
def get_triangle_colliding_vertices(col_info: TriMeshCollisionInfo, tri: int, i_collision: int):
    offset = col_info.triangle_colliding_vertices_offsets[tri]
    return col_info.triangle_colliding_vertices[offset + i_collision]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(col_info.edge_colliding_edges_count[e], col_info.edge_colliding_edges_buffer_sizes[e])


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


def _as_numpy(arr) -> np.ndarray:
    """Return ``arr`` as NumPy, accepting either a NumPy or a Warp int array."""
    return arr if isinstance(arr, np.ndarray) else arr.numpy()


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract row ``i`` from flat CSR arrays."""
    return vals[offs[i] : offs[i + 1]]


def set_to_csr(
    list_of_sets: list[set[int]], dtype: np.dtype = np.int32, sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-row integer sets to flat CSR values and offsets."""
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])

    flat = np.empty(offsets[-1], dtype=dtype)
    cursor = 0
    for row in list_of_sets:
        values = np.fromiter(sorted(row) if sort else row, count=len(row), dtype=dtype)
        flat[cursor : cursor + len(values)] = values
        cursor += len(values)
    return flat, offsets


def one_ring_vertices(
    vertex: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Return vertices sharing a collision edge with ``vertex``."""
    edge_v0 = edge_indices[:, 2]
    edge_v1 = edge_indices[:, 3]
    edge_rows = _csr_row(v_adj_edges, v_adj_edges_offsets, vertex)
    edge_ids = edge_rows[::2]
    local_slots = edge_rows[1::2]
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    endpoint_edge_ids = edge_ids[np.where(local_slots >= 2)]
    us = edge_v0[endpoint_edge_ids]
    vs = edge_v1[endpoint_edge_ids]
    assert (np.logical_or(us == vertex, vs == vertex)).all()

    neighbors = np.unique(np.concatenate([us, vs]))
    return neighbors[neighbors != vertex]


def leq_n_ring_vertices(
    vertex: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Return vertices within ``n`` edge rings of ``vertex``, including itself."""
    visited = {vertex}
    frontier = {vertex}
    for _ in range(n):
        next_frontier = set()
        for current in frontier:
            for neighbor in one_ring_vertices(current, edge_indices, v_adj_edges, v_adj_edges_offsets):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=np.int32)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    particle_count: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_tris: np.ndarray,
    v_adj_tris_offsets: np.ndarray,
) -> list[set[int]] | None:
    """Build vertex-triangle filters from adjacency within ``n`` edge rings."""
    if n <= 1:
        return None

    vertex_triangle_sets = [set() for _ in range(particle_count)]
    for vertex in range(particle_count):
        if n == 2:
            neighbor_vertices = one_ring_vertices(vertex, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            neighbor_vertices = leq_n_ring_vertices(vertex, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        incident_tris = set(_csr_row(v_adj_tris, v_adj_tris_offsets, vertex)[::2])
        filter_set = vertex_triangle_sets[vertex]
        for neighbor in neighbor_vertices:
            if neighbor != vertex:
                filter_set.update(_csr_row(v_adj_tris, v_adj_tris_offsets, neighbor)[::2])
        filter_set.difference_update(incident_tris)

    return vertex_triangle_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
) -> list[set[int]] | None:
    """Build edge-edge filters from adjacency within ``n`` edge rings."""
    if n <= 1:
        return None

    edge_sets = [set() for _ in range(edge_indices.shape[0])]
    for edge_id in range(edge_indices.shape[0]):
        v0 = edge_indices[edge_id, 2]
        v1 = edge_indices[edge_id, 3]

        if n == 2:
            v0_neighbors = one_ring_vertices(v0, edge_indices, v_adj_edges, v_adj_edges_offsets)
            v1_neighbors = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            v0_neighbors = leq_n_ring_vertices(v0, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            v1_neighbors = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        neighbor_vertices = set(v0_neighbors)
        neighbor_vertices.update(v1_neighbors)

        incident_to_v0 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v0)[::2])
        incident_to_v1 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])

        filter_set = edge_sets[edge_id]
        for neighbor in neighbor_vertices:
            if neighbor != v0 and neighbor != v1:
                edge_rows = _csr_row(v_adj_edges, v_adj_edges_offsets, neighbor)
                adj_edges = edge_rows[::2]
                local_slots = edge_rows[1::2]
                filter_set.update(adj_edges[np.where(local_slots >= 2)])

        filter_set.difference_update(incident_to_v0)
        filter_set.difference_update(incident_to_v1)

    return edge_sets


class TriMeshCollisionDetector:
    """BVH-based triangle mesh self-collision detector.

    Detects vertex-triangle, edge-edge, and triangle-triangle collisions using
    Warp BVH acceleration structures. Results are stored in CSR-format buffers
    accessible via the :class:`TriMeshCollisionInfo` struct.

    This class is solver-agnostic — it handles detection only, not constraint
    resolution. Solvers (VBD, XPBD, etc.) consume the detection results through
    their own force/constraint kernels.
    """

    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_collision_buffer_max_alloc=256,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=8,
        edge_collision_buffer_max_alloc=256,
        edge_filtering_list=None,
        edge_filtering_list_offsets=None,
        topological_contact_filter_threshold: int = 0,
        external_vertex_triangle_filtering_map: dict | None = None,
        external_edge_edge_filtering_map: dict | None = None,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
        collision_detection_block_size=16,
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.vertex_collision_buffer_max_alloc = vertex_collision_buffer_max_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.triangle_collision_buffer_max_alloc = triangle_collision_buffer_max_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.edge_collision_buffer_max_alloc = edge_collision_buffer_max_alloc
        self.triangle_triangle_collision_buffer_pre_alloc = triangle_triangle_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_max_alloc = triangle_triangle_collision_buffer_max_alloc

        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

        self.edge_edge_parallel_epsilon = edge_edge_parallel_epsilon
        # The soft-mesh adjacency comes from the model; ensure its vertex-adjacency CSR is built.
        # init_vertex_adjacency is idempotent (vertex_adjacency_initialized flag), so this is a no-op
        # once the solver has built it.
        if model.soft_mesh_adjacency is None:
            raise ValueError("model.soft_mesh_adjacency is missing; finalize the model with ModelBuilder.")
        self.mesh_adjacency = model.soft_mesh_adjacency.init_vertex_adjacency(model.particle_count)

        self.collision_detection_block_size = collision_detection_block_size

        # Build each filter family independently: generate a side only when the caller did not
        # provide it explicitly and a threshold/external source requests it (so providing one
        # list plus an external map for the other side still generates the missing side).
        need_vertex_triangle = vertex_triangle_filtering_list is None and (
            topological_contact_filter_threshold >= 2 or external_vertex_triangle_filtering_map is not None
        )
        need_edge_edge = edge_filtering_list is None and (
            topological_contact_filter_threshold >= 2 or external_edge_edge_filtering_map is not None
        )
        if (need_vertex_triangle or need_edge_edge) and self.model.tri_count > 0:
            # Extract the shared vertex adjacency once, then build each family with its own builder.
            adjacency = None
            if topological_contact_filter_threshold >= 2 and self.model.edge_indices is not None:
                adjacency = self._extract_filter_adjacency()
            if need_vertex_triangle:
                self._build_vertex_triangle_filter(
                    topological_contact_filter_threshold, external_vertex_triangle_filtering_map, adjacency
                )
            if need_edge_edge:
                self._build_edge_edge_filter(
                    topological_contact_filter_threshold, external_edge_edge_filtering_map, adjacency
                )

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.tri_groups = wp.array(shape=(model.tri_count,), dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )
        wp.launch(
            kernel=compute_tri_groups,
            inputs=[model.tri_indices, model.particle_world, model.world_count, self.tri_groups],
            dim=model.tri_count,
            device=model.device,
        )

        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris, groups=self.tri_groups)
        self.bvh_tris_group_roots = wp.zeros(model.world_count + 1, dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=model.world_count + 1,
            inputs=[self.bvh_tris.id, self.bvh_tris_group_roots],
            device=model.device,
        )

        # collision detections results

        # vertex collision buffers
        self.vertex_colliding_triangles = wp.zeros(
            shape=(2 * model.particle_count * self.vertex_collision_buffer_pre_alloc,),
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_count = wp.array(
            shape=(model.particle_count,), dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.array(
            shape=(model.particle_count,), dtype=float, device=self.device
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.full(
            shape=(model.particle_count,),
            value=self.vertex_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            shape=(model.particle_count + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        if record_triangle_contacting_vertices:
            # triangle collision buffers
            self.triangle_colliding_vertices = wp.zeros(
                shape=(model.tri_count * self.triangle_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(model.tri_count,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_buffer_sizes = wp.full(
                shape=(model.tri_count,),
                value=self.triangle_collision_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            self.triangle_colliding_vertices_offsets = wp.array(
                shape=(model.tri_count + 1,), dtype=wp.int32, device=self.device
            )
            self.compute_collision_buffer_offsets(
                self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
            )
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_count = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_offsets = None

        # this is need regardless of whether we record triangle contacting vertices
        self.triangle_colliding_vertices_min_dist = wp.array(shape=(model.tri_count,), dtype=float, device=self.device)

        # edge collision buffers
        self.edge_colliding_edges = wp.zeros(
            shape=(2 * model.edge_count * self.edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(shape=(model.edge_count,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.full(
            shape=(model.edge_count,),
            value=self.edge_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_colliding_edges_offsets = wp.array(shape=(model.edge_count + 1,), dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        self.edge_colliding_edges_min_dist = wp.array(shape=(model.edge_count,), dtype=float, device=self.device)

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.edge_groups = wp.array(shape=(model.edge_count,), dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )
        wp.launch(
            kernel=compute_edge_groups,
            inputs=[model.edge_indices, model.particle_world, model.world_count, self.edge_groups],
            dim=model.edge_count,
            device=model.device,
        )

        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges, groups=self.edge_groups)
        self.bvh_edges_group_roots = wp.zeros(model.world_count + 1, dtype=wp.int32, device=model.device)
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=model.world_count + 1,
            inputs=[self.bvh_edges.id, self.bvh_edges_group_roots],
            device=model.device,
        )

        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

        self.collision_info = self.get_collision_data()

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def set_collision_filter_list(
        self,
        vertex_triangle_filtering_list,
        vertex_triangle_filtering_list_offsets,
        edge_filtering_list,
        edge_filtering_list_offsets,
    ):
        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets

        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

    def _extract_filter_adjacency(self):
        """Return ``(edge_indices, v_adj_edges, v_adj_edges_offsets, v_adj_tris, v_adj_tris_offsets)`` as
        NumPy for the topological filter builders.

        Reuses the model's vertex-adjacency CSR when it is already populated, otherwise computes it on
        demand. Shared by the vertex-triangle and edge-edge builders so the adjacency is extracted once.
        """
        edge_indices = self.model.edge_indices.numpy()
        adjacency = self.mesh_adjacency
        if (
            adjacency is not None
            and adjacency.v_adj_edges is not None
            and adjacency.v_adj_edges.size > 0
            and adjacency.v_adj_edges_offsets.size > 0
            and adjacency.v_adj_tris_offsets.size > 0
        ):
            source = adjacency
        else:
            source = MeshAdjacency.compute_vertex_adjacency(
                self.model.particle_count,
                edge_indices=self.model.edge_indices,
                tri_indices=self.model.tri_indices,
            )
        return (
            edge_indices,
            _as_numpy(source.v_adj_edges),
            _as_numpy(source.v_adj_edges_offsets),
            _as_numpy(source.v_adj_tris),
            _as_numpy(source.v_adj_tris_offsets),
        )

    def _build_vertex_triangle_filter(
        self,
        topological_contact_filter_threshold: int,
        external_vertex_triangle_filtering_map: dict | None,
        adjacency: tuple | None,
    ) -> None:
        """Build the detector-owned vertex-triangle filter list from the n-ring topology and the optional
        external map. The caller decides whether this side is needed (an explicitly-provided list is left
        untouched); ``adjacency`` is the shared :meth:`_extract_filter_adjacency` result or ``None``.
        """
        filter_sets = None
        if topological_contact_filter_threshold >= 2 and adjacency is not None:
            edge_indices, v_adj_edges, v_adj_edges_offsets, v_adj_tris, v_adj_tris_offsets = adjacency
            filter_sets = build_vertex_n_ring_tris_collision_filter(
                topological_contact_filter_threshold,
                self.model.particle_count,
                edge_indices,
                v_adj_edges,
                v_adj_edges_offsets,
                v_adj_tris,
                v_adj_tris_offsets,
            )
        if external_vertex_triangle_filtering_map is not None:
            if filter_sets is None:
                filter_sets = [set() for _ in range(self.model.particle_count)]
            for vertex_id, filter_set in external_vertex_triangle_filtering_map.items():
                filter_sets[vertex_id].update(filter_set)

        if filter_sets is not None:
            filtering_list, filtering_list_offsets = set_to_csr(filter_sets)
            self.vertex_triangle_filtering_list = wp.array(filtering_list, dtype=wp.int32, device=self.device)
            self.vertex_triangle_filtering_list_offsets = wp.array(
                filtering_list_offsets, dtype=wp.int32, device=self.device
            )

    def _build_edge_edge_filter(
        self,
        topological_contact_filter_threshold: int,
        external_edge_edge_filtering_map: dict | None,
        adjacency: tuple | None,
    ) -> None:
        """Build the detector-owned edge-edge filter list from the n-ring topology and the optional
        external map. The caller decides whether this side is needed (an explicitly-provided list is left
        untouched); ``adjacency`` is the shared :meth:`_extract_filter_adjacency` result or ``None``.
        """
        filter_sets = None
        if topological_contact_filter_threshold >= 2 and adjacency is not None:
            edge_indices, v_adj_edges, v_adj_edges_offsets, _, _ = adjacency
            filter_sets = build_edge_n_ring_edge_collision_filter(
                topological_contact_filter_threshold,
                edge_indices,
                v_adj_edges,
                v_adj_edges_offsets,
            )
        if external_edge_edge_filtering_map is not None:
            if filter_sets is None:
                filter_sets = [set() for _ in range(self.model.edge_count)]
            for edge_id, filter_set in external_edge_edge_filtering_map.items():
                filter_sets[edge_id].update(filter_set)

        if filter_sets is not None:
            filtering_list, filtering_list_offsets = set_to_csr(filter_sets)
            self.edge_filtering_list = wp.array(filtering_list, dtype=wp.int32, device=self.device)
            self.edge_filtering_list_offsets = wp.array(filtering_list_offsets, dtype=wp.int32, device=self.device)

    def get_collision_data(self):
        collision_info = TriMeshCollisionInfo()

        collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        collision_info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        collision_info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        collision_info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        collision_info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
            collision_info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            collision_info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            collision_info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count

        collision_info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        collision_info.edge_colliding_edges = self.edge_colliding_edges
        collision_info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        collision_info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        collision_info.edge_colliding_edges_count = self.edge_colliding_edges_count
        collision_info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist

        return collision_info

    def compute_collision_buffer_offsets(self, buffer_sizes: wp.array[wp.int32], offsets: wp.array[wp.int32]):
        assert offsets.size == buffer_sizes.size + 1
        offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
        offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
        offsets_np[0] = 0

        offsets.assign(offsets_np)

    # Overflow flag indices in self.resize_flags (matches constants in geometry/kernels.py)
    _VERTEX_OVERFLOW_IDX = 0
    _TRI_OVERFLOW_IDX = 1
    _EDGE_OVERFLOW_IDX = 2

    def _check_and_resize_vertex_triangle_buffers(self) -> bool:
        """Check for vertex-triangle buffer overflow and resize if needed.

        Returns:
            True if buffers were resized (caller should re-run detection).
        """
        flags = self.resize_flags.numpy()
        resized = False

        if flags[self._VERTEX_OVERFLOW_IDX]:
            sizes_np = self.vertex_colliding_triangles_buffer_sizes.numpy()
            new_sizes = np.minimum(sizes_np * 2, self.vertex_collision_buffer_max_alloc)
            if not np.array_equal(sizes_np, new_sizes):
                resized = True
                self.vertex_colliding_triangles_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
                self.vertex_colliding_triangles_offsets = wp.array(
                    shape=(new_sizes.shape[0] + 1,), dtype=wp.int32, device=self.device
                )
                self.compute_collision_buffer_offsets(
                    self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
                )
                total = int(np.sum(new_sizes))
                self.vertex_colliding_triangles = wp.zeros(shape=(2 * total,), dtype=wp.int32, device=self.device)

        if flags[self._TRI_OVERFLOW_IDX] and self.record_triangle_contacting_vertices:
            sizes_np = self.triangle_colliding_vertices_buffer_sizes.numpy()
            new_sizes = np.minimum(sizes_np * 2, self.triangle_collision_buffer_max_alloc)
            if not np.array_equal(sizes_np, new_sizes):
                resized = True
                self.triangle_colliding_vertices_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
                self.triangle_colliding_vertices_offsets = wp.array(
                    shape=(new_sizes.shape[0] + 1,), dtype=wp.int32, device=self.device
                )
                self.compute_collision_buffer_offsets(
                    self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
                )
                total = int(np.sum(new_sizes))
                self.triangle_colliding_vertices = wp.zeros(shape=(total,), dtype=wp.int32, device=self.device)

        return resized

    def _check_and_resize_edge_buffers(self) -> bool:
        """Check for edge-edge buffer overflow and resize if needed.

        Returns:
            True if buffers were resized (caller should re-run detection).
        """
        flags = self.resize_flags.numpy()
        if not flags[self._EDGE_OVERFLOW_IDX]:
            return False

        sizes_np = self.edge_colliding_edges_buffer_sizes.numpy()
        new_sizes = np.minimum(sizes_np * 2, self.edge_collision_buffer_max_alloc)
        if np.array_equal(sizes_np, new_sizes):
            return False

        self.edge_colliding_edges_buffer_sizes = wp.array(new_sizes, dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_offsets = wp.array(
            shape=(new_sizes.shape[0] + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        total = int(np.sum(new_sizes))
        self.edge_colliding_edges = wp.zeros(shape=(2 * total,), dtype=wp.int32, device=self.device)
        return True

    def vertex_triangle_collision_detection_with_resize(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        """Run vertex-triangle detection, automatically resizing buffers on overflow.

        Runs detection once, checks overflow flags, and if buffers overflowed,
        doubles their sizes (up to max_alloc) and re-runs detection.
        """
        self.vertex_triangle_collision_detection(max_query_radius, min_query_radius, min_distance_filtering_ref_pos)
        if self._check_and_resize_vertex_triangle_buffers():
            self.resize_flags.zero_()
            self.collision_info = self.get_collision_data()
            self.vertex_triangle_collision_detection(max_query_radius, min_query_radius, min_distance_filtering_ref_pos)

    def edge_edge_collision_detection_with_resize(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        """Run edge-edge detection, automatically resizing buffers on overflow.

        Runs detection once, checks overflow flags, and if buffers overflowed,
        doubles their sizes (up to max_alloc) and re-runs detection.
        """
        self.edge_edge_collision_detection(max_query_radius, min_query_radius, min_distance_filtering_ref_pos)
        if self._check_and_resize_edge_buffers():
            self.resize_flags.zero_()
            self.collision_info = self.get_collision_data()
            self.edge_edge_collision_detection(max_query_radius, min_query_radius, min_distance_filtering_ref_pos)

    def rebuild(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[
                self.vertex_positions,
                self.model.tri_indices,
            ],
            outputs=[self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.rebuild()
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=self.model.world_count + 1,
            inputs=[self.bvh_tris.id, self.bvh_tris_group_roots],
            device=self.model.device,
        )

        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.rebuild()
        wp.launch(
            kernel=compute_bvh_group_roots,
            dim=self.model.world_count + 1,
            inputs=[self.bvh_edges.id, self.bvh_edges_group_roots],
            device=self.model.device,
        )

    def refit(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        self.refit_triangles()
        self.refit_edges()

    def refit_triangles(self):
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.refit()

    def vertex_triangle_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.vertex_colliding_triangles.fill_(-1)

        if self.record_triangle_contacting_vertices:
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[
                    max_query_radius,
                ],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(max_query_radius)

        wp.launch(
            kernel=vertex_triangle_collision_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_tris.id,
                self.bvh_tris_group_roots,
                self.vertex_positions,
                self.model.tri_indices,
                self.model.particle_world,
                self.model.world_count,
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.triangle_colliding_vertices_offsets,
                self.triangle_colliding_vertices_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.triangle_colliding_vertices,
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def edge_edge_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.edge_colliding_edges.fill_(-1)
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_edges.id,
                self.bvh_edges_group_roots,
                self.vertex_positions,
                self.model.edge_indices,
                self.model.particle_world,
                self.model.world_count,
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def triangle_triangle_intersection_detection(self):
        if self.triangle_intersecting_triangles is None:
            self.triangle_intersecting_triangles = wp.zeros(
                shape=(self.model.tri_count * self.triangle_triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )

        if self.triangle_intersecting_triangles_count is None:
            self.triangle_intersecting_triangles_count = wp.array(
                shape=(self.model.tri_count,), dtype=wp.int32, device=self.device
            )

        if self.triangle_intersecting_triangles_offsets is None:
            buffer_sizes = np.full((self.model.tri_count,), self.triangle_triangle_collision_buffer_pre_alloc)
            offsets = np.zeros((self.model.tri_count + 1,), dtype=np.int32)
            offsets[1:] = np.cumsum(buffer_sizes)

            self.triangle_intersecting_triangles_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=triangle_triangle_collision_detection_kernel,
            inputs=[
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.triangle_intersecting_triangles_offsets,
            ],
            outputs=[
                self.triangle_intersecting_triangles,
                self.triangle_intersecting_triangles_count,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )


# --- Topological filtering helpers (CPU/numpy) ---


def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract CSR row `i` from the flattened adjacency arrays."""
    return vals[offs[i] : offs[i + 1]]


def set_to_csr(
    list_of_sets: list[set[int]], dtype: np.dtype = np.int32, sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of integer sets into CSR (Compressed Sparse Row) structure.

    Args:
        list_of_sets: Iterable where each entry is a set of ints.
        dtype: Output dtype for the flattened arrays.
        sort: Whether to sort each row when writing into ``flat``.

    Returns:
        A tuple ``(flat, offsets)`` representing the CSR values and offsets.
    """
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.empty(offsets[-1], dtype=dtype)
    idx = 0
    for s in list_of_sets:
        if sort:
            arr = np.fromiter(sorted(s), count=len(s), dtype=dtype)
        else:
            arr = np.fromiter(s, count=len(s), dtype=dtype)

        flat[idx : idx + len(arr)] = arr
        idx += len(arr)
    return flat, offsets


def one_ring_vertices(
    v: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Find immediate neighboring vertices that share an edge with vertex `v`.

    Args:
        v: Vertex index whose neighborhood is queried.
        edge_indices: Array of shape [num_edges, 4] storing edge endpoint indices.
        v_adj_edges: Flattened CSR adjacency array listing edge ids and local order.
        v_adj_edges_offsets: CSR offsets indexing into `v_adj_edges`.

    Returns:
        Sorted array of neighboring vertex indices, excluding `v`.
    """
    e_u = edge_indices[:, 2]
    e_v = edge_indices[:, 3]
    # preserve only the adjacent edge information, remove the order information
    inc_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[::2]
    inc_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[1::2]
    if inc_edges.size == 0:
        return np.empty(0)
    us = e_u[inc_edges[np.where(inc_edges_order >= 2)]]
    vs = e_v[inc_edges[np.where(inc_edges_order >= 2)]]

    assert (np.logical_or(us == v, vs == v)).all()
    nbrs = np.unique(np.concatenate([us, vs]))
    return nbrs[nbrs != v]


def leq_n_ring_vertices(
    v: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """Find all vertices within n-ring distance of vertex v using BFS.

    Args:
        v: Starting vertex index.
        edge_indices: Edge connectivity array.
        n: Maximum ring distance.
        v_adj_edges: CSR values for vertex-edge adjacency.
        v_adj_edges_offsets: CSR offsets for vertex-edge adjacency.

    Returns:
        Array of all vertices within n-ring distance, including v itself.
    """
    visited = {v}
    frontier = {v}
    for _ in range(n):
        next_frontier = set()
        for u in frontier:
            for w in one_ring_vertices(u, edge_indices, v_adj_edges, v_adj_edges_offsets):  # iterable of neighbors of u
                if w not in visited:
                    visited.add(w)
                    next_frontier.add(w)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=int)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    num_vertices: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_faces: np.ndarray,
    v_adj_faces_offsets: np.ndarray,
):
    """For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.

    Excludes triangles incident to v itself (dist 0).

    Returns:
        v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex.
    """

    if n <= 1:
        return None, None

    v_nei_tri_sets = [set() for _ in range(num_vertices)]

    for v in range(num_vertices):
        # distance-1 vertices

        if n == 2:
            ring_n_minus_1 = one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        ring_1_tri_set = set(_csr_row(v_adj_faces, v_adj_faces_offsets, v)[::2])

        nei_tri_set = v_nei_tri_sets[v]
        for w in ring_n_minus_1:
            if w != v:
                # preserve only the adjacent edge information, remove the order information
                nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])

        nei_tri_set.difference_update(ring_1_tri_set)

    return v_nei_tri_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
):
    """For each edge e, return edges within n-ring distance.

    Excludes edges incident to the edge's own vertices.

    Returns:
        List of sets, one per edge.
    """

    if n <= 1:
        return None, None

    edge_nei_edge_sets = [set() for _ in range(edge_indices.shape[0])]

    for e_idx in range(edge_indices.shape[0]):
        # distance-1 vertices
        v1 = edge_indices[e_idx, 2]
        v2 = edge_indices[e_idx, 3]

        if n == 2:
            ring_n_minus_1_v1 = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = one_ring_vertices(v2, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1_v1 = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = leq_n_ring_vertices(v2, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        all_neighbors = set(ring_n_minus_1_v1)
        all_neighbors.update(ring_n_minus_1_v2)

        ring_1_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])
        ring_2_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v2)[::2])

        nei_edge_set = edge_nei_edge_sets[e_idx]
        for w in all_neighbors:
            if w != v1 and w != v2:
                # preserve only the adjacent edge information, remove the order information
                # nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])
                adj_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[::2]
                adj_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[1::2]
                adj_collision_edges = adj_edges[np.where(adj_edges_order >= 2)]
                nei_edge_set.update(adj_collision_edges)

        nei_edge_set.difference_update(ring_1_edge_set)
        nei_edge_set.difference_update(ring_2_edge_set)

    return edge_nei_edge_sets
