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

"""Continuous Collision Detection (CCD) primitives for cloth self-collision.

Implements Bridson/Provot cubic-equation CCD for vertex-triangle and
edge-edge swept collisions. Uses coplanarity tests (det = 0 gives a cubic
in t) followed by barycentric/parameter validation at each root.

References:
    - R. Bridson, R. Fedkiw, J. Anderson: "Robust Treatment of Collisions,
      Contact and Friction for Cloth Animation" (2002), Section 4.
    - X. Provot: "Collision and self-collision handling in cloth model
      dedicated to design garments" (1997).
"""

import warp as wp

# Sentinel value for "no collision found"
CCD_NO_COLLISION = wp.constant(2.0)


@wp.func
def _solve_quadratic_roots_01(a: float, b: float, c: float, roots: wp.vec4) -> wp.vec4:
    """Solve a*t^2 + b*t + c = 0 for roots in [0,1]. Appends to existing roots vec4.

    roots.w stores the current count of valid roots (as float).
    """
    count = int(roots[3])
    if wp.abs(a) < 1e-20:
        # Linear
        if wp.abs(b) > 1e-20:
            t = -c / b
            if 0.0 <= t <= 1.0 and count < 3:
                roots[count] = t
                count += 1
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sq = wp.sqrt(disc)
            inv_2a = 0.5 / a
            t1 = (-b - sq) * inv_2a
            t2 = (-b + sq) * inv_2a
            if 0.0 <= t1 <= 1.0 and count < 3:
                roots[count] = t1
                count += 1
            if 0.0 <= t2 <= 1.0 and count < 3 and wp.abs(t2 - t1) > 1e-12:
                roots[count] = t2
                count += 1
    roots[3] = float(count)
    return roots


@wp.func
def solve_cubic_roots_01(a: float, b: float, c: float, d: float) -> wp.vec4:
    """Solve a*t^3 + b*t^2 + c*t + d = 0, return roots in [0,1].

    Returns:
        vec4 where [0..2] are roots (CCD_NO_COLLISION sentinel for unused slots),
        and [3] is the count of valid roots.
    """
    roots = wp.vec4(CCD_NO_COLLISION, CCD_NO_COLLISION, CCD_NO_COLLISION, 0.0)

    if wp.abs(a) < 1e-20:
        # Degenerate to quadratic
        return _solve_quadratic_roots_01(b, c, d, roots)

    # Normalize: t^3 + p*t^2 + q*t + r = 0
    inv_a = 1.0 / a
    p = b * inv_a
    q = c * inv_a
    r = d * inv_a

    # Depressed cubic substitution: t = u - p/3
    # u^3 + au + b = 0  where a_dep = q - p^2/3, b_dep = r - p*q/3 + 2*p^3/27
    p_over_3 = p / 3.0
    a_dep = q - p * p_over_3
    b_dep = r - p_over_3 * q + 2.0 * p_over_3 * p_over_3 * p_over_3

    # Discriminant for depressed cubic
    disc = b_dep * b_dep / 4.0 + a_dep * a_dep * a_dep / 27.0

    count = 0

    if disc > 1e-20:
        # One real root (Cardano)
        sqrt_disc = wp.sqrt(disc)
        half_b = -b_dep / 2.0
        s_arg = half_b + sqrt_disc
        t_arg = half_b - sqrt_disc
        # Cube root with sign preservation
        s = wp.sign(s_arg) * wp.pow(wp.abs(s_arg), 1.0 / 3.0)
        t_val = wp.sign(t_arg) * wp.pow(wp.abs(t_arg), 1.0 / 3.0)
        root = s + t_val - p_over_3
        if 0.0 <= root <= 1.0:
            roots[count] = root
            count += 1
    elif disc < -1e-20:
        # Three real roots (trigonometric method)
        sqrt_neg_a3 = wp.sqrt(-a_dep / 3.0)
        cos_arg = -b_dep / (2.0 * sqrt_neg_a3 * sqrt_neg_a3 * sqrt_neg_a3)
        cos_arg = wp.clamp(cos_arg, -1.0, 1.0)
        theta = wp.acos(cos_arg) / 3.0

        root0 = 2.0 * sqrt_neg_a3 * wp.cos(theta) - p_over_3
        root1 = 2.0 * sqrt_neg_a3 * wp.cos(theta - 2.0 * 3.14159265358979323846 / 3.0) - p_over_3
        root2 = 2.0 * sqrt_neg_a3 * wp.cos(theta - 4.0 * 3.14159265358979323846 / 3.0) - p_over_3

        if 0.0 <= root0 <= 1.0 and count < 3:
            roots[count] = root0
            count += 1
        if 0.0 <= root1 <= 1.0 and count < 3:
            roots[count] = root1
            count += 1
        if 0.0 <= root2 <= 1.0 and count < 3:
            roots[count] = root2
            count += 1
    else:
        # Discriminant ~ 0: repeated roots
        if wp.abs(a_dep) > 1e-20:
            root0 = 3.0 * b_dep / a_dep - p_over_3
            root1 = -3.0 * b_dep / (2.0 * a_dep) - p_over_3
            if 0.0 <= root0 <= 1.0 and count < 3:
                roots[count] = root0
                count += 1
            if 0.0 <= root1 <= 1.0 and count < 3 and wp.abs(root1 - root0) > 1e-12:
                roots[count] = root1
                count += 1
        else:
            # Triple root at -p/3
            root0 = -p_over_3
            if 0.0 <= root0 <= 1.0:
                roots[count] = root0
                count += 1

    roots[3] = float(count)
    return roots


@wp.func
def _triple_product(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Scalar triple product: dot(a, cross(b, c))."""
    return wp.dot(a, wp.cross(b, c))


@wp.func
def ccd_vertex_triangle(
    v0: wp.vec3,
    v1: wp.vec3,
    a0: wp.vec3,
    a1: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    c0: wp.vec3,
    c1: wp.vec3,
    thickness: float,
) -> float:
    """Find earliest time-of-impact for a vertex sweeping toward a triangle.

    Uses the Bridson/Provot cubic coplanarity test:
    det(v(t)-a(t), b(t)-a(t), c(t)-a(t)) = 0 yields a cubic in t.

    Args:
        v0, v1: Vertex position at t=0 and t=1.
        a0, a1: Triangle vertex A at t=0 and t=1.
        b0, b1: Triangle vertex B at t=0 and t=1.
        c0, c1: Triangle vertex C at t=0 and t=1.
        thickness: Combined contact thickness [m].

    Returns:
        Time of impact in [0,1], or CCD_NO_COLLISION (2.0) if no collision.
    """
    # Positions and displacements relative to triangle vertex A
    x1 = v0 - a0
    x2 = b0 - a0
    x3 = c0 - a0

    d1 = (v1 - a1) - (v0 - a0)
    d2 = (b1 - a1) - (b0 - a0)
    d3 = (c1 - a1) - (c0 - a0)

    # Cubic coefficients from triple product expansion
    # f(t) = coeff_a*t^3 + coeff_b*t^2 + coeff_c*t + coeff_d
    coeff_d = _triple_product(x1, x2, x3)
    coeff_c = _triple_product(d1, x2, x3) + _triple_product(x1, d2, x3) + _triple_product(x1, x2, d3)
    coeff_b = _triple_product(d1, d2, x3) + _triple_product(d1, x2, d3) + _triple_product(x1, d2, d3)
    coeff_a = _triple_product(d1, d2, d3)

    roots = solve_cubic_roots_01(coeff_a, coeff_b, coeff_c, coeff_d)
    num_roots = int(roots[3])

    min_toi = CCD_NO_COLLISION

    for i in range(3):
        if i >= num_roots:
            break
        t = roots[i]

        # Interpolate positions at time t
        vt = v0 + t * (v1 - v0)
        at = a0 + t * (a1 - a0)
        bt = b0 + t * (b1 - b0)
        ct = c0 + t * (c1 - c0)

        # Compute closest point on triangle to vertex (barycentric)
        e1 = bt - at
        e2 = ct - at
        vp = vt - at

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
        w = 1.0 - u - v

        # Check if point is inside triangle (with small tolerance)
        tol = -1e-4
        if u >= tol and v >= tol and w >= tol:
            # Compute distance
            closest = at + u * e1 + v * e2
            dist = wp.length(vt - closest)
            if dist < thickness and t < min_toi:
                min_toi = t

    return min_toi


@wp.func
def ccd_edge_edge(
    a0: wp.vec3,
    a1: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    c0: wp.vec3,
    c1: wp.vec3,
    d0: wp.vec3,
    d1: wp.vec3,
    thickness: float,
) -> float:
    """Find earliest time-of-impact for two sweeping edges.

    Edge 1: a→b, Edge 2: c→d. Uses coplanarity condition
    det(b(t)-a(t), d(t)-c(t), a(t)-c(t)) = 0.

    Args:
        a0, a1: Edge 1 start vertex at t=0 and t=1.
        b0, b1: Edge 1 end vertex at t=0 and t=1.
        c0, c1: Edge 2 start vertex at t=0 and t=1.
        d0, d1: Edge 2 end vertex at t=0 and t=1.
        thickness: Combined contact thickness [m].

    Returns:
        Time of impact in [0,1], or CCD_NO_COLLISION (2.0) if no collision.
    """
    # Edge directions and separation at t=0
    x1 = b0 - a0
    x2 = d0 - c0
    x3 = a0 - c0

    d1_v = (b1 - a1) - (b0 - a0)
    d2_v = (d1 - c1) - (d0 - c0)
    d3_v = (a1 - c1) - (a0 - c0)

    # Cubic coefficients
    coeff_d = _triple_product(x1, x2, x3)
    coeff_c = _triple_product(d1_v, x2, x3) + _triple_product(x1, d2_v, x3) + _triple_product(x1, x2, d3_v)
    coeff_b = _triple_product(d1_v, d2_v, x3) + _triple_product(d1_v, x2, d3_v) + _triple_product(x1, d2_v, d3_v)
    coeff_a = _triple_product(d1_v, d2_v, d3_v)

    roots = solve_cubic_roots_01(coeff_a, coeff_b, coeff_c, coeff_d)
    num_roots = int(roots[3])

    min_toi = CCD_NO_COLLISION

    for i in range(3):
        if i >= num_roots:
            break
        t = roots[i]

        # Interpolate positions at time t
        at = a0 + t * (a1 - a0)
        bt = b0 + t * (b1 - b0)
        ct = c0 + t * (c1 - c0)
        dt_v = d0 + t * (d1 - d0)

        # Find closest points on the two edges
        e1 = bt - at
        e2 = dt_v - ct
        r = at - ct

        a_val = wp.dot(e1, e1)
        e_val = wp.dot(e2, e2)
        f_val = wp.dot(e2, r)

        if a_val < 1e-20 or e_val < 1e-20:
            continue

        b_val = wp.dot(e1, e2)
        c_val = wp.dot(e1, r)

        denom = a_val * e_val - b_val * b_val
        if wp.abs(denom) < 1e-20:
            continue

        s = (b_val * f_val - c_val * e_val) / denom
        t_param = (a_val * f_val - b_val * c_val) / denom

        # Check parameters are within edge bounds
        if 0.0 <= s <= 1.0 and 0.0 <= t_param <= 1.0:
            p1 = at + s * e1
            p2 = ct + t_param * e2
            dist = wp.length(p1 - p2)
            if dist < thickness and t < min_toi:
                min_toi = t

    return min_toi


@wp.kernel
def compute_tri_aabbs_swept(
    pos_t0: wp.array(dtype=wp.vec3),
    pos_t1: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    thickness: float,
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    """Compute AABBs enclosing the swept volume of each triangle from t0 to t1.

    The AABB is the union of the triangle at t=0 and t=1, expanded by thickness.
    """
    tid = wp.tid()
    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]

    # Positions at t=0
    pi0 = pos_t0[i]
    pj0 = pos_t0[j]
    pk0 = pos_t0[k]

    # Positions at t=1
    pi1 = pos_t1[i]
    pj1 = pos_t1[j]
    pk1 = pos_t1[k]

    lo = wp.min(wp.min(wp.min(wp.min(wp.min(pi0, pj0), pk0), pi1), pj1), pk1)
    hi = wp.max(wp.max(wp.max(wp.max(wp.max(pi0, pj0), pk0), pi1), pj1), pk1)

    expand = wp.vec3(thickness, thickness, thickness)
    lower_bounds[tid] = lo - expand
    upper_bounds[tid] = hi + expand


@wp.kernel
def compute_edge_aabbs_swept(
    pos_t0: wp.array(dtype=wp.vec3),
    pos_t1: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    thickness: float,
    # outputs
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    """Compute AABBs enclosing the swept volume of each edge from t0 to t1."""
    eid = wp.tid()
    i = edge_indices[eid, 2]
    j = edge_indices[eid, 3]

    pi0 = pos_t0[i]
    pj0 = pos_t0[j]
    pi1 = pos_t1[i]
    pj1 = pos_t1[j]

    lo = wp.min(wp.min(wp.min(pi0, pj0), pi1), pj1)
    hi = wp.max(wp.max(wp.max(pi0, pj0), pi1), pj1)

    expand = wp.vec3(thickness, thickness, thickness)
    lower_bounds[eid] = lo - expand
    upper_bounds[eid] = hi + expand


@wp.kernel
def vertex_triangle_ccd_kernel(
    pos_t0: wp.array(dtype=wp.vec3),
    pos_t1: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    thickness: float,
    bvh_id: wp.uint64,
    filtering_list: wp.array(dtype=wp.int32),
    filtering_list_offsets: wp.array(dtype=wp.int32),
    # outputs
    toi_per_vertex: wp.array(dtype=float),
    contact_tri_per_vertex: wp.array(dtype=wp.int32),
):
    """Per-vertex CCD: query swept BVH, run cubic CCD against candidates, store min TOI."""
    v_index = wp.tid()

    v0 = pos_t0[v_index]
    v1 = pos_t1[v_index]

    # Query BVH with the swept AABB of this vertex
    lo = wp.min(v0, v1) - wp.vec3(thickness, thickness, thickness)
    hi = wp.max(v0, v1) + wp.vec3(thickness, thickness, thickness)

    min_toi = CCD_NO_COLLISION
    min_tri = -1

    query = wp.bvh_query_aabb(bvh_id, lo, hi)
    tri_index = int(0)
    while wp.bvh_query_next(query, tri_index):
        # Skip if vertex is part of this triangle
        ti = tri_indices[tri_index, 0]
        tj = tri_indices[tri_index, 1]
        tk = tri_indices[tri_index, 2]
        if ti == v_index or tj == v_index or tk == v_index:
            continue

        # Topological filtering
        if filtering_list_offsets is not None:
            filter_start = filtering_list_offsets[v_index]
            filter_end = filtering_list_offsets[v_index + 1]
            skip = False
            for f_idx in range(filter_start, filter_end):
                if filtering_list[f_idx] == tri_index:
                    skip = True
                    break
            if skip:
                continue

        toi = ccd_vertex_triangle(
            v0,
            v1,
            pos_t0[ti],
            pos_t1[ti],
            pos_t0[tj],
            pos_t1[tj],
            pos_t0[tk],
            pos_t1[tk],
            thickness,
        )
        if toi < min_toi:
            min_toi = toi
            min_tri = tri_index

    toi_per_vertex[v_index] = min_toi
    contact_tri_per_vertex[v_index] = min_tri


@wp.kernel
def edge_edge_ccd_kernel(
    pos_t0: wp.array(dtype=wp.vec3),
    pos_t1: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    thickness: float,
    bvh_id: wp.uint64,
    filtering_list: wp.array(dtype=wp.int32),
    filtering_list_offsets: wp.array(dtype=wp.int32),
    # outputs
    toi_per_edge: wp.array(dtype=float),
    contact_edge_per_edge: wp.array(dtype=wp.int32),
):
    """Per-edge CCD: query swept BVH, run cubic CCD against candidate edges, store min TOI."""
    e_index = wp.tid()

    e1_i = edge_indices[e_index, 2]
    e1_j = edge_indices[e_index, 3]

    a0 = pos_t0[e1_i]
    a1 = pos_t1[e1_i]
    b0 = pos_t0[e1_j]
    b1 = pos_t1[e1_j]

    # Query BVH with the swept AABB of this edge
    lo = wp.min(wp.min(wp.min(a0, a1), b0), b1) - wp.vec3(thickness, thickness, thickness)
    hi = wp.max(wp.max(wp.max(a0, a1), b0), b1) + wp.vec3(thickness, thickness, thickness)

    min_toi = CCD_NO_COLLISION
    min_edge = -1

    query = wp.bvh_query_aabb(bvh_id, lo, hi)
    e2_index = int(0)
    while wp.bvh_query_next(query, e2_index):
        if e2_index <= e_index:
            continue  # Avoid duplicate pairs

        e2_i = edge_indices[e2_index, 2]
        e2_j = edge_indices[e2_index, 3]

        # Skip edges sharing a vertex
        if e1_i == e2_i or e1_i == e2_j or e1_j == e2_i or e1_j == e2_j:
            continue

        # Topological filtering
        if filtering_list_offsets is not None:
            filter_start = filtering_list_offsets[e_index]
            filter_end = filtering_list_offsets[e_index + 1]
            skip = False
            for f_idx in range(filter_start, filter_end):
                if filtering_list[f_idx] == e2_index:
                    skip = True
                    break
            if skip:
                continue

        toi = ccd_edge_edge(
            a0,
            a1,
            b0,
            b1,
            pos_t0[e2_i],
            pos_t1[e2_i],
            pos_t0[e2_j],
            pos_t1[e2_j],
            thickness,
        )
        if toi < min_toi:
            min_toi = toi
            min_edge = e2_index

    toi_per_edge[e_index] = min_toi
    contact_edge_per_edge[e_index] = min_edge


@wp.kernel
def clamp_positions_to_toi(
    pos_t0: wp.array(dtype=wp.vec3),
    pos_t1: wp.array(dtype=wp.vec3),
    toi_vertices: wp.array(dtype=float),
    toi_edges: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    safety_factor: float,
    # outputs
    pos_out: wp.array(dtype=wp.vec3),
):
    """Clamp particle positions to prevent tunneling past the earliest TOI.

    For each particle, find the minimum TOI across its vertex CCD result
    and any edge CCD results involving it, then interpolate position.
    """
    vid = wp.tid()

    min_toi = toi_vertices[vid]

    # Clamp only if we found a collision
    if min_toi < CCD_NO_COLLISION:
        clamped_t = min_toi * safety_factor
        pos_out[vid] = pos_t0[vid] + clamped_t * (pos_t1[vid] - pos_t0[vid])
