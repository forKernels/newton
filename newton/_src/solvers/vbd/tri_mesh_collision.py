# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

# Re-export from shared geometry module for backward compatibility.
from ...geometry.tri_mesh_collision import (  # noqa: F401
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_edge_collision_buffer_edge_index,
    get_triangle_colliding_vertices,
    get_triangle_colliding_vertices_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
    get_vertex_collision_buffer_vertex_index,
    one_ring_vertices,
    set_to_csr,
)
