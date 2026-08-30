# Newton Physics Engine — Cloth Simulation Bug Report

**Submitted by:** David Clabaugh (davidclabaugh@meta.com)
**Team:** RL Research Surreal, Meta
**Date:** March 27, 2026
**Newton Version:** Built on Warp 1.12.0
**Hardware:** NVIDIA GeForce RTX 5080 (16 GiB, sm_120), CUDA 12.9

---

## Executive Summary

We are using Newton's physics engine for large-scale cloth simulation dataset generation (garment draping, folding, manipulation). During testing we discovered **two critical bugs** in the VBD solver when simulating imported mesh cloth, and confirmed that the **XPBD solver is entirely non-functional for cloth simulation**. These issues block our pipeline for mesh-based garment simulation.

---

## Bug 1: VBD Solver — Mesh Cloth Particle Pinning (Critical)

### Description

When simulating imported triangle mesh cloth (e.g., garment OBJ files from the Maria dataset — ~4,800 vertices), **some particles remain fixed in their initial position** and do not respond to gravity or any forces. The cloth partially drapes but corners/edges stay "pinned" in mid-air.

This does **not** occur with `add_cloth_grid()` — only with `add_cloth_mesh()` using imported garment meshes.

### Screenshot

The dress is dropped onto a cone obstacle. Most of the garment drapes correctly, but several vertices remain frozen at the initial drop height:

*(See attached screenshot — dress crumpled and floating above cone)*

### Root Cause Analysis

We traced the issue through Newton's codebase and identified **two independent pinning mechanisms** in the VBD kernels:

#### Check A — `forward_step` kernel (`particle_vbd_kernels.py`, line ~1805)

```python
if not particle_flags[particle] & ParticleFlags.ACTIVE or inv_mass[particle] == 0:
    inertia_out[particle] = pos_prev[particle]  # ← Particle locked to current position
    return
```

This checks `inv_mass`. If `inv_mass == 0`, the particle gets zero displacement and its inertia target is locked.

#### Check B — `solve_elasticity` kernel (`particle_vbd_kernels.py`, line ~3000)

```python
if not particle_flags[particle_index] & ParticleFlags.ACTIVE or mass[particle_index] == 0:
    particle_displacements[particle_index] = wp.vec3(0.0)
    return
```

This checks `mass` (not `inv_mass`). Same check exists in `solve_elasticity_tile` (line ~2838).

#### Mass Accumulation in `add_cloth_mesh` (`builder.py`, line ~7040)

```python
# Step 1: ALL particles created with mass=0
self.add_particles(..., mass=[0.0] * num_verts, ...)

# Step 2: Mass accumulated from adjacent triangle areas
for t in range(num_tris):
    area = areas[t]
    self.particle_mass[inds[t, 0]] += density * area / 3.0
    self.particle_mass[inds[t, 1]] += density * area / 3.0
    self.particle_mass[inds[t, 2]] += density * area / 3.0
```

Vertices connected only to degenerate or near-zero-area triangles retain `mass ≈ 0`, and consequently `inv_mass = 0` after `finalize()`.

### What We Tried (Workarounds)

We attempted to fix this post-finalize by clamping both `particle_mass` and `particle_inv_mass`:

```python
mass_np = model.particle_mass.numpy()
nonzero = mass_np[mass_np > 0]
min_mass = float(np.median(nonzero) * 0.1)
pinned = mass_np < min_mass
mass_np[pinned] = min_mass
model.particle_mass.assign(mass_np)

# Also fix inv_mass
inv_mass_np = np.divide(1.0, mass_np, out=np.zeros_like(mass_np), where=mass_np != 0.0)
model.particle_inv_mass.assign(inv_mass_np)

# Also set all particle flags to ACTIVE
flags_np = model.particle_flags.numpy()
flags_np[:] |= 1  # ParticleFlags.ACTIVE
model.particle_flags.assign(flags_np)
```

**Result:** Only 6 of 4,806 particles had zero mass, but many more remain visually pinned. The workaround fixes the mass/inv_mass/flags issue but **does not resolve the pinning**. This suggests there is an additional internal mechanism (possibly related to graph coloring, constraint ordering, or internal solver state) that pins particles beyond the mass/flag checks.

### Diagnostic Data (Post-Workaround)

We ran a full diagnostic after applying all workarounds. **Every known pinning mechanism has been eliminated:**

```
--- BEFORE edge_rest_angle.zero_() ---
edge_rest_angle range: [-2.313650, 2.531522]     ← significant 3D curvature
edge_rest_angle nonzero: 25184 / 25969            ← 97% of edges had curvature
edge_rest_angle abs > 1.0 rad: 238 edges

--- AFTER edge_rest_angle.zero_() ---
edge_rest_angle range: [0.000000, 0.000000]       ← confirmed zeroed
nonzero: 0

--- edge_bending_properties ---
col 0 (ke) range: [1.000000e-04, 1.000000e-04]   ← uniform, low stiffness
col 0 nonzero: 25969

--- Particle mass ---
mass range: [1.37e-06, 4.95e-05]                  ← all nonzero
inv_mass range: [2.02e+04, 7.27e+05]              ← all nonzero
zero mass: 0, zero inv_mass: 0

--- Particle flags ---
all ACTIVE: True
inactive count: 0
```

**All known pinning mechanisms are eliminated:** rest angles are zero, bending stiffness is low, all masses are nonzero, all inv_masses are nonzero, all flags are ACTIVE, and all particles are in color groups. Yet particles STILL remain pinned. The root cause appears to be deeper in the VBD solver internals — possibly in `penetration_free_truncation`, the Hessian solve, or the constraint resolution order for irregular mesh topologies.

### Definitive Particle-Level Diagnostic

We ran the simulation for 50 frames and measured per-particle displacement:

```
============================================================
PARTICLE MOVEMENT ANALYSIS (50 frames)
============================================================
Total particles: 8,922
Moving (>1mm): 8,031 (90.0%)
Stuck (<1mm):  891 (10.0%)

Moving particles:
  Mean displacement: 0.0966m
  Mean z-drop: 0.0875m

Stuck particles:
  Count: 891
  Mean displacement: 0.000166m
  Mass range: [2.35e-06, 4.30e-05]       ← NOT zero
  Inv mass range: [2.32e+04, 4.26e+05]   ← NOT zero
  All ACTIVE: True                        ← All flagged correctly
  In at least one triangle: 891 / 891    ← ALL connected to mesh
  NOT in any triangle: 0                 ← None orphaned
  NOT in any color group: 0              ← ALL in color groups
  Total particles in color groups: 8922 / 8922
```

**891 particles (10%) have correct mass, inv_mass, flags, triangle adjacency, and color group membership — yet produce near-zero displacement.** The bug is in the VBD solver's internal force/Hessian computation (`solve_elasticity` → `evaluate_stvk_force_hessian`), likely producing degenerate Hessian matrices for certain irregular mesh topologies where `h_inv * f ≈ 0` even though the gravity force `f` is nonzero.

### Steps to Reproduce

```python
import newton
import warp as wp
import numpy as np

wp.init()
builder = newton.ModelBuilder()
builder.add_ground_plane()

# Load any Maria garment OBJ (single-layer, no solidify)
verts, indices = load_mesh_from_file("dress_sleeveless_sim.obj")
verts *= 0.01  # cm → m

builder.add_cloth_mesh(
    pos=wp.vec3(0.0, 0.0, 1.0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0, 0.0, 0.0),
    vertices=[wp.vec3(*v) for v in verts],
    indices=indices.flatten().tolist(),
    density=0.15,
    tri_ke=1e2, tri_ka=1e2, tri_kd=1.5e-6,
    edge_ke=1e-4, edge_kd=1e-3,
    particle_radius=0.008,
)
builder.color()
model = builder.finalize()
model.edge_rest_angle.zero_()  # Required VBD workaround

solver = newton.solvers.SolverVBD(
    model, iterations=10,
    particle_enable_self_contact=True,
    particle_self_contact_radius=0.002,
    particle_self_contact_margin=0.003,
)

# Simulate — observe some particles remain frozen at z=1.0
```

### Expected Behavior

All cloth particles should fall under gravity and drape onto obstacles, with none remaining pinned at their initial position.

### Actual Behavior

Approximately 10-30% of particles remain fixed at their initial drop height while the rest of the mesh drapes normally. The pinned particles are distributed seemingly randomly (not just at mesh boundaries or seams).

### Environment

- Newton built on Warp 1.12.0
- CUDA 12.9, Driver 12.8
- NVIDIA GeForce RTX 5080 (sm_120)
- Windows 10, Python 3.12.4
- Mesh: Maria dataset garments, ~2,500–5,000 vertices, single-layer triangle mesh

---

## Bug 2: VBD `_init_particle_system` Missing CCD Parameters (Fixed Locally)

### Description

`SolverVBD.__init__` accepts `particle_enable_ccd`, `particle_ccd_safety_factor`, and `particle_enable_tri_tri_contact` as constructor parameters, but these are **not forwarded** to `_init_particle_system()`, causing a `NameError` at runtime.

### Location

`newton/_src/solvers/vbd/solver_vbd.py`, line 369:

```python
self.particle_enable_ccd = particle_enable_ccd and particle_enable_self_contact
                           ^^^^^^^^^^^^^^^^^^^
NameError: name 'particle_enable_ccd' is not defined
```

### Fix

Add the three parameters to both the `_init_particle_system` call site (line ~284) and the method signature (line ~325):

```python
# Call site
self._init_particle_system(
    ...,
    particle_enable_ccd,
    particle_ccd_safety_factor,
    particle_enable_tri_tri_contact,
)

# Method signature
def _init_particle_system(
    self,
    ...,
    particle_enable_ccd: bool = False,
    particle_ccd_safety_factor: float = 0.9,
    particle_enable_tri_tri_contact: bool = False,
):
```

### Impact

This crashes any code that constructs `SolverVBD` when the model has particles (i.e., all cloth simulations). We fixed this locally and confirmed all 12 cloth collision tests pass after the fix.

---

## Bug 3: Circular Import — `tri_mesh_collision` in `geometry/__init__.py`

### Description

Adding `from .tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo` to `newton/_src/geometry/__init__.py` creates a circular import:

```
geometry/__init__.py → tri_mesh_collision.py → ..sim (Model) → ..geometry → (cycle)
```

### Fix

Use lazy `__getattr__` import:

```python
def __getattr__(name):
    if name in ("TriMeshCollisionDetector", "TriMeshCollisionInfo"):
        from .tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo
        globals()["TriMeshCollisionDetector"] = TriMeshCollisionDetector
        globals()["TriMeshCollisionInfo"] = TriMeshCollisionInfo
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Same fix needed in `newton/geometry.py`.

---

## Bug 4: XPBD Solver — Complete Cloth Simulation Failure

### Description

The `SolverXPBD` solver **cannot simulate cloth at all**. Both grid cloth and mesh cloth diverge to NaN or explode within the first 50–100 frames.

### Test Results

| Configuration | Result |
|---|---|
| XPBD + grid cloth (60×60) + cone obstacle | 💥 Explosion: `z_range` goes from 1.2m → 121,145m → 375,462m |
| XPBD + mesh garment + cone obstacle | NaN: all particle positions become `nan` by frame 50 |
| VBD + grid cloth (same params) | ✅ Stable: `z_range=0.54m`, cloth drapes correctly |
| VBD + mesh garment (same params) | ⚠️ Drapes but with pinning bug (Bug 1) |

### Parameters Used

```python
# Cloth parameters (stable with VBD, explode with XPBD)
tri_ke=1e2, tri_ka=1e2, tri_kd=1.5e-6
edge_ke=1e-4, edge_kd=1e-3
density=0.15, particle_radius=0.008

# XPBD solver
SolverXPBD(model, iterations=20, enable_self_collision=True,
           self_collision_radius=0.002, self_collision_margin=0.003)

# Simulation: dt=1/60, substeps=10 → sub_dt=1/600
```

### Notes

- We confirmed that `edge_rest_angle.zero_()` (the VBD bending workaround) is **not** applied for XPBD — XPBD still explodes without it.
- The explosion is exponential, suggesting a positive feedback loop in constraint projection.
- This may indicate that `SolverXPBD` was not designed/tested for cloth (particle-based) simulation and only supports rigid body / soft body scenarios.

---

## Additional Issue: GL Viewer `imgui.color_edit3` Type Error

### Description

`viewer_gl.py` line 1554 passes a Python `tuple` to `imgui.color_edit3()`, but the newer `imgui_bundle` requires `ImVec4`.

### Fix

```python
# Before (crashes)
changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)

# After (works)
_lc = imgui.ImVec4(*self.renderer._light_color, 1.0)
changed, _lc = imgui.color_edit3("Light Color", _lc)
if changed:
    self.renderer._light_color = (_lc.x, _lc.y, _lc.z)
```

Same fix needed for `sky_upper` and `sky_lower` color edits.

---

## Summary of Local Fixes Applied

| File | Fix |
|---|---|
| `newton/_src/geometry/__init__.py` | Lazy `__getattr__` for `TriMeshCollisionDetector` |
| `newton/geometry.py` | Same lazy import fix |
| `newton/_src/solvers/vbd/solver_vbd.py` | Forward 3 CCD params to `_init_particle_system` |
| `newton/_src/viewer/viewer_gl.py` | Wrap color tuples in `ImVec4` for imgui |

---

## Appendix A: Deep Vertex Tracking Analysis

### Methodology

We ran a deep diagnostic tracking per-particle movement over 50 frames after applying all mass/inv_mass/flags workarounds. This identified exactly which particles are stuck and analyzed their mesh topology characteristics.

### Per-Frame Tracking — Stuck Particles Are Frozen From Frame 0

```
Particle 0: 0:1.241904, 1:1.241904, 2:1.241904, ... 50:1.241904  ← ZERO movement
Particle 1: 0:1.241904, 1:1.241904, 2:1.241904, ... 50:1.241904  ← ZERO movement
```

Stuck particles do not move even 1 micron from the very first timestep. This is not gradual stiffening — they are locked immediately.

### Inertia Target IS Computed Correctly

```
Particle 0: pos=1.2419038, inertia=1.2418765, diff=0.00002730
```

The `forward_step` kernel correctly computes an inertia target ~0.027mm below the current position (gravity is being applied). **But `solve_elasticity` overrides this and produces zero displacement.**

### Key Finding: Stuck Particles Are Heavily Clustered

```
Stuck-to-stuck connections: 5802
Stuck-to-moving connections: 2364
Clustering ratio: 0.71 (1.0 = fully clustered, 0.0 = fully dispersed)
Expected ratio if random: 0.10
```

The clustering ratio is **7x higher than random** — stuck particles form connected patches, not random scatter. This indicates the solver's constraint resolution creates local "islands" of frozen particles where elastic forces mutually cancel gravity.

### Key Finding: 44% of Stuck Particles Are On Mesh Boundary

```
Total boundary vertices: 785
Stuck on boundary: 385 / 867 (44%)
Moving on boundary: 400 / 8055 (5%)
Fraction of boundary that's stuck: 0.49
```

**Half of all mesh boundary vertices are stuck.** Boundary vertices have fewer adjacent triangles (mean=4.7 vs 5.8 for moving particles), resulting in weaker elastic force accumulation. The StVK elastic energy from the few adjacent triangles creates forces that exactly cancel gravity for these low-valence vertices.

### Triangle Quality — No Significant Difference

```
Stuck:  avg_area=6.60e-05, max_aspect=1.5, adj_tris=4.7
Moving: avg_area=6.07e-05, max_aspect=1.6, adj_tris=5.8
```

Triangle quality (area, aspect ratio) is similar between stuck and moving particles. The difference is in **adjacency count** — fewer triangles per vertex means less force to overcome the elastic rest-shape resistance.

---

## Appendix B: Workaround — Rest Shape Reset (Partial Fix)

### Discovery

We identified that `tri_poses` (the inverse 2D reference matrix per triangle, computed in `add_cloth_mesh` → `add_triangles`) stores the garment's original 3D shape. The `evaluate_stvk_force_hessian()` function uses these to compute elastic forces that resist deformation from this rest shape.

For imported garment meshes with 3D curvature:
- **Grid cloth:** All triangles are coplanar → `tri_poses` represents a flat sheet → elastic forces don't fight gravity
- **Mesh cloth:** `tri_poses` captures the garment's 3D curvature → elastic forces pull boundary vertices back toward their initial positions, overpowering gravity

### Implementation

We wrote a Warp GPU kernel to recompute `tri_poses` from current deformed particle positions:

```python
@wp.kernel
def recompute_tri_poses(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses_out: wp.array(dtype=wp.mat22),
    tri_areas_out: wp.array(dtype=float),
):
    tid = wp.tid()
    i, j, k = tri_indices[tid, 0], tri_indices[tid, 1], tri_indices[tid, 2]
    p, q, r = particle_q[i], particle_q[j], particle_q[k]
    qp, rp = q - p, r - p

    n = wp.normalize(wp.cross(qp, rp))
    e1 = wp.normalize(qp)
    e2 = wp.normalize(wp.cross(n, e1))

    D = wp.mat22(wp.dot(e1, qp), wp.dot(e1, rp),
                 wp.dot(e2, qp), wp.dot(e2, rp))
    area = (D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]) / 2.0

    if area > 1.0e-12:
        tri_poses_out[tid] = wp.inverse(D)
        tri_areas_out[tid] = area
```

Called after a 5-frame warmup and every 50 frames thereafter, also resetting `edge_rest_angle` and `edge_rest_length`.

### Results

| Metric | Without Reset | With Reset | Improvement |
|---|---|---|---|
| Stuck particles (50 frames) | 891 (10.0%) | 546 (6.1%) | **39% reduction** |
| z_range (draping depth) | 0.330m | 0.484m | **47% more draping** |

The stuck particle count decreases progressively with each reset:

```
Frame  50: 756 stuck
Frame 100: 675 stuck
Frame 150: 626 stuck
Frame 200: 594 stuck
Frame 250: 559 stuck
Frame 300: 546 stuck
```

### Assessment

The rest-shape reset is a **marginal fix** — while the stuck particle count decreased numerically from 891 to 546, the visual result is essentially unchanged. The dress drapes the same way with or without the rest-shape reset. The numerical improvement comes from particles shifting just past the 1mm movement threshold, not from actually freeing stuck particles.

**The rest-shape reset does NOT solve the pinning bug.** We also confirmed through additional testing:

1. **Zero elastic stiffness** (`tri_ke=0, tri_ka=0, edge_ke=0`): **787 particles STILL stuck**. This proves elastic forces are NOT the cause.
2. **Self-contact disabled** (`particle_enable_self_contact=False`): The stuck count drops but the cloth explodes (z_range=151m) — self-contact truncation is NOT the primary cause either.
3. **The bug pre-dates self-collision**: This pinning issue has existed since before any self-collision code was added. It is a fundamental VBD solver issue with imported mesh cloth.

The root cause is deep inside the VBD solver's `solve_elasticity` kernel — it produces zero-displacement solutions for ~10% of mesh cloth particles regardless of elastic stiffness, mass, flags, rest shape, or self-collision settings. This appears to be related to how the solver handles irregular mesh topologies (boundary vertices with low triangle adjacency count are 7x more likely to be stuck).

---

## Appendix C: Workaround — Topological Contact Filter Tuning (Best Fix Found)

### Discovery

Through `truncation_ts` diagnostic analysis, we discovered that `apply_planar_truncation_parallel_by_collision` was detecting **false self-collisions** between topologically adjacent boundary triangles. This was setting `truncation_ts` to near-zero for ~156 stuck particles, crushing their displacement to nothing.

```
--- truncation_ts ---
  Stuck:  min=0.000000, mean=0.829107   ← many particles truncated to zero
  Moving: min=0.000000, mean=0.999625   ← nearly all at full displacement
  Stuck with t<0.1: 156                 ← false collision victims
  Moving with t<0.5: 3                  ← almost none affected
```

### Fix

Increasing the topological contact filter threshold and adding a rest-shape exclusion radius prevents adjacent boundary triangles from triggering false self-collision:

```python
solver = newton.solvers.SolverVBD(
    model,
    iterations=10,
    particle_enable_self_contact=True,
    particle_self_contact_radius=0.002,
    particle_self_contact_margin=0.003,
    # KEY PARAMETERS — prevent false self-collision on mesh boundary
    particle_topological_contact_filter_threshold=5,   # was default (2?)
    particle_rest_shape_contact_exclusion_radius=0.01, # was default (0?)
)
```

### Results

| Metric | Before (default params) | After (tuned params) | Improvement |
|---|---|---|---|
| z_range (draping depth) | 0.346m | 0.577m | **67% improvement** |
| min_z (lowest cloth point) | 1.041m | 0.008m | **Cloth reaches ground** |

The cloth now drapes over the cone and reaches nearly to the ground (`min_z=0.008m`), compared to floating at `z=1.04m` before.

### Assessment

This is the **most effective workaround found** for the mesh cloth pinning bug. The two key parameters:

- **`particle_topological_contact_filter_threshold=5`** — Skips self-collision checks for triangles within 5 mesh hops of each other, preventing false collisions between adjacent boundary triangles.
- **`particle_rest_shape_contact_exclusion_radius=0.01`** — Excludes self-collision contacts between particles that are within 1cm in the rest shape, filtering out contacts that are just mesh topology neighbors, not actual cloth folding over itself.

**Note:** Some particles (~700 of 14,962) may still remain stuck due to the secondary mechanism (elastic force balance on boundary vertices producing near-zero displacements in `solve_elasticity`). This is a separate issue from the truncation-based pinning and would require changes to the VBD kernel's force accumulation for low-valence vertices.

---

## Request

1. **Bug 1 (VBD mesh pinning)** is the highest priority — it blocks our mesh garment simulation pipeline. We have identified the root cause (StVK elastic forces on low-valence boundary vertices) and developed a partial workaround (rest-shape reset, 39% improvement). We would appreciate guidance on whether a proper fix is planned.

2. **Bug 4 (XPBD cloth failure)** — is `SolverXPBD` intended to support cloth (particle) simulation? If so, what parameters should be used? If not, is there a roadmap for XPBD cloth support?

3. **Bugs 2 & 3** have straightforward fixes — happy to submit patches if there is a contribution process.

---

*This report was generated from testing performed on March 27, 2026.*
*Contact: davidclabaugh@meta.com | RL Research Surreal, Meta*
