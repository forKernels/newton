"""
Spring Force Assist for Newton VBD Cloth Simulation
=====================================================
Injects mass-spring forces as external forces into state.particle_f to
provide a "gravity assist" for stuck particles in Newton's VBD solver.

The VBD solver has a known bug where ~10% of particles on imported garments
produce zero displacement. This module computes stretching and bending spring
forces on the GPU via Warp kernels and writes them into the force accumulator
between clear_forces() and solver.step().

Usage:
    spring_assist = SpringForceAssist(model, ks=500.0, kb=25.0)
    # In sim loop:
    state.clear_forces()
    spring_assist.apply(state)
    solver.step(state, ...)
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import warp as wp


# ── Warp kernels ─────────────────────────────────────────────────────────────

@wp.kernel
def _compute_spring_forces(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    forces: wp.array(dtype=wp.vec3),
):
    """Compute spring force for one spring and scatter via atomic_add."""
    sid = wp.tid()

    pair = spring_indices[sid]
    i = pair[0]
    j = pair[1]

    xi = positions[i]
    xj = positions[j]

    diff = xi - xj
    dist = wp.length(diff)

    if dist < 1.0e-8:
        return

    direction = diff / dist

    rl = rest_lengths[sid]
    ks = stiffness[sid]
    kd = damping[sid]

    # Spring force: F = k * (1 - rest_len / dist) * diff
    f_spring = direction * (ks * (dist - rl))

    # Damping force: F_damp = -kd * dot(v_rel, dir) * dir
    vi = velocities[i]
    vj = velocities[j]
    v_rel = vi - vj
    f_damp = direction * (-kd * wp.dot(v_rel, direction))

    f_total = f_spring + f_damp

    # Newton's 3rd law: equal and opposite
    wp.atomic_sub(forces, i, f_total)
    wp.atomic_add(forces, j, f_total)


@wp.kernel
def _apply_forces_all(
    spring_forces: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    boost_factor: float,
):
    """Add spring forces (scaled by boost_factor) into particle_f for all particles."""
    i = wp.tid()
    particle_f[i] = particle_f[i] + spring_forces[i] * boost_factor


@wp.kernel
def _apply_forces_masked(
    spring_forces: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.int32),
    boost_factor: float,
):
    """Add spring forces only where mask[i] != 0 (stuck particles)."""
    i = wp.tid()
    if mask[i] != 0:
        particle_f[i] = particle_f[i] + spring_forces[i] * boost_factor


@wp.kernel
def _compute_displacement(
    current_q: wp.array(dtype=wp.vec3),
    prev_q: wp.array(dtype=wp.vec3),
    displacement: wp.array(dtype=float),
):
    """Compute per-particle displacement magnitude between two states."""
    i = wp.tid()
    displacement[i] = wp.length(current_q[i] - prev_q[i])


# ── SpringForceAssist class ─────────────────────────────────────────────────


class SpringForceAssist:
    """Mass-spring force assist for Newton's VBD cloth solver.

    Builds a spring network from mesh triangle topology and computes
    spring forces on the GPU. Forces are injected into state.particle_f
    between clear_forces() and solver.step().

    Args:
        model: Finalized Newton model with cloth particles.
        ks: Stretching spring stiffness (1-ring edges).
        kb: Bending spring stiffness (2-ring neighbors).
        kd_stretch: Damping coefficient for stretching springs.
        kd_bend: Damping coefficient for bending springs.
        boost_factor: Multiplier applied to all spring forces.
        stuck_threshold: Displacement threshold [m] below which a particle
            is considered stuck (per substep).
        stuck_only: If True, only apply forces to particles detected as stuck.
        verbose: Print spring network statistics.
    """

    def __init__(
        self,
        model,
        ks: float = 500.0,
        kb: float = 25.0,
        kd_stretch: float = 5.0,
        kd_bend: float = 1.0,
        boost_factor: float = 1.0,
        stuck_threshold: float = 1e-5,
        stuck_only: bool = False,
        verbose: bool = True,
    ):
        self.model = model
        self.n_particles = model.particle_count
        self.boost_factor = boost_factor
        self.stuck_threshold = stuck_threshold
        self.stuck_only = stuck_only
        self.verbose = verbose

        # Build spring network from triangle topology
        stretch_pairs, bend_pairs = self._build_spring_network(model)

        if verbose:
            print(f"  SpringForceAssist: {len(stretch_pairs)} stretch + "
                  f"{len(bend_pairs)} bend = {len(stretch_pairs) + len(bend_pairs)} springs")

        # Get initial positions for rest lengths
        rest_positions = model.particle_q.numpy()

        # Compute rest lengths and build flat arrays
        all_indices = []
        all_rest_lengths = []
        all_stiffness = []
        all_damping = []

        for i, j in stretch_pairs:
            all_indices.append((i, j))
            all_rest_lengths.append(np.linalg.norm(rest_positions[i] - rest_positions[j]))
            all_stiffness.append(ks)
            all_damping.append(kd_stretch)

        for i, j in bend_pairs:
            all_indices.append((i, j))
            all_rest_lengths.append(np.linalg.norm(rest_positions[i] - rest_positions[j]))
            all_stiffness.append(kb)
            all_damping.append(kd_bend)

        self.n_springs = len(all_indices)

        if self.n_springs == 0:
            if verbose:
                print("  SpringForceAssist: WARNING — no springs built (no triangles?)")
            self._empty = True
            return

        self._empty = False

        # Upload to GPU as Warp arrays
        idx_np = np.array(all_indices, dtype=np.int32)
        self.spring_indices = wp.array(idx_np, dtype=wp.vec2i)
        self.rest_lengths = wp.array(np.array(all_rest_lengths, dtype=np.float32), dtype=float)
        self.stiffness = wp.array(np.array(all_stiffness, dtype=np.float32), dtype=float)
        self.damping = wp.array(np.array(all_damping, dtype=np.float32), dtype=float)

        # Force accumulator (zeroed each call)
        self.spring_forces = wp.zeros(self.n_particles, dtype=wp.vec3)

        # Stuck detection state
        self.prev_q = wp.zeros(self.n_particles, dtype=wp.vec3)
        self.displacement = wp.zeros(self.n_particles, dtype=float)
        self.stuck_mask = wp.zeros(self.n_particles, dtype=wp.int32)
        self._frame_count = 0

        # Stats
        self._n_stretch = len(stretch_pairs)
        self._n_bend = len(bend_pairs)
        self._total_stuck = 0
        self._total_calls = 0

    @staticmethod
    def _build_spring_network(model):
        """Extract 1-ring (stretching) and 2-ring (bending) springs from mesh triangles.

        Returns:
            stretch_pairs: List of (i, j) tuples for 1-ring edges.
            bend_pairs: List of (i, j) tuples for 2-ring neighbors (excluding 1-ring).
        """
        tri_indices = model.tri_indices.numpy()  # shape (N, 3)

        # 1-ring: unique edges from triangles
        adjacency = defaultdict(set)
        edge_set = set()

        for tri in tri_indices:
            v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in [(v0, v1), (v1, v2), (v0, v2)]:
                edge = (min(a, b), max(a, b))
                if edge not in edge_set:
                    edge_set.add(edge)
                adjacency[a].add(b)
                adjacency[b].add(a)

        stretch_pairs = list(edge_set)

        # 2-ring: BFS depth=2 from each vertex, excluding self and 1-ring neighbors
        bend_set = set()
        for v in adjacency:
            ring1 = adjacency[v]
            for n1 in ring1:
                for n2 in adjacency.get(n1, set()):
                    if n2 != v and n2 not in ring1:
                        edge = (min(v, n2), max(v, n2))
                        bend_set.add(edge)

        bend_pairs = list(bend_set)

        return stretch_pairs, bend_pairs

    def apply(self, state, dt: float = 0.0):
        """Compute and inject spring forces into state.particle_f.

        Call between state.clear_forces() and solver.step().

        Args:
            state: Newton simulation state (must have particle_q, particle_qd, particle_f).
            dt: Substep dt (unused currently, reserved for adaptive schemes).
        """
        if self._empty:
            return

        # Zero the force accumulator
        self.spring_forces.zero_()

        # Compute spring forces on GPU
        wp.launch(
            _compute_spring_forces,
            dim=self.n_springs,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.spring_indices,
                self.rest_lengths,
                self.stiffness,
                self.damping,
                self.spring_forces,
            ],
        )

        if self.stuck_only:
            # Update stuck mask from displacement tracking
            self._update_stuck_mask(state)

            wp.launch(
                _apply_forces_masked,
                dim=self.n_particles,
                inputs=[
                    self.spring_forces,
                    state.particle_f,
                    self.stuck_mask,
                    self.boost_factor,
                ],
            )
        else:
            wp.launch(
                _apply_forces_all,
                dim=self.n_particles,
                inputs=[
                    self.spring_forces,
                    state.particle_f,
                    self.boost_factor,
                ],
            )

        self._frame_count += 1

    def _update_stuck_mask(self, state):
        """Detect stuck particles by comparing current and previous positions."""
        if self._frame_count == 0:
            # First call: copy positions, mark all as stuck (conservative)
            wp.copy(self.prev_q, state.particle_q)
            self.stuck_mask.fill_(1)
            return

        # Compute displacement since last call
        wp.launch(
            _compute_displacement,
            dim=self.n_particles,
            inputs=[
                state.particle_q,
                self.prev_q,
                self.displacement,
            ],
        )

        # Transfer to CPU for mask update (small overhead, simple logic)
        disp_np = self.displacement.numpy()
        mask_np = (disp_np < self.stuck_threshold).astype(np.int32)
        self.stuck_mask.assign(mask_np)

        # Track stats
        n_stuck = int(np.sum(mask_np))
        self._total_stuck += n_stuck
        self._total_calls += 1

        # Save current positions for next comparison
        wp.copy(self.prev_q, state.particle_q)

    def get_stats(self) -> dict:
        """Return diagnostic stats."""
        avg_stuck = (self._total_stuck / self._total_calls) if self._total_calls > 0 else 0
        return {
            "n_springs": self.n_springs,
            "n_particles": self.n_particles,
            "total_calls": self._total_calls,
            "avg_stuck": avg_stuck,
            "boost_factor": self.boost_factor,
            "stuck_only": self.stuck_only,
        }

    def log_stats(self):
        """Print diagnostic stats."""
        stats = self.get_stats()
        print(f"  SpringForceAssist stats:")
        print(f"    Springs: {stats['n_springs']} "
              f"({self._n_stretch} stretch, {self._n_bend} bend)")
        print(f"    Boost factor: {stats['boost_factor']}")
        if stats['stuck_only']:
            print(f"    Avg stuck particles: {stats['avg_stuck']:.0f} / {stats['n_particles']}")
        print(f"    Total apply() calls: {stats['total_calls']}")
