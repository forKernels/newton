# USD Export Enhancement Plan — Blender-Ready Animated Scenes

## Context

Newton's `ViewerUSD` (`--viewer usd`) already records simulation data to USD files with per-frame
rigid body transforms and deformable mesh vertex positions. However, the output lacks materials,
textures, UV coordinates, and merges all deformable meshes into a single prim — making it
unsuitable for direct use in Blender. This plan enhances `ViewerUSD` to produce Blender-ready
USD files with full visual fidelity.

## Scope

**Adding:**
1. UV coordinate export (currently a TODO that silently drops UVs)
2. `UsdPreviewSurface` material export (diffuse color, roughness, metallic)
3. Texture image file export (referenced from materials)
4. Per-body deformable mesh splitting (separate USD prims instead of one monolithic mesh)
5. MPM sand particles as instanced spheres via `UsdGeom.PointInstancer`

**Unchanged:** Rigid body instancing with per-frame transforms, static geometry, CLI interface,
normals (Blender auto-computes).

**Out of scope:** Fluid simulation, camera export, skeleton hierarchies.

---

## Step 1: Implement UV coordinate export in `ViewerUSD.log_mesh()`

**File:** `newton/_src/viewer/viewer_usd.py`, lines 255-258

Replace the TODO block with actual UV primvar writing:

```python
if uvs is not None:
    uvs_np = uvs.numpy().astype(np.float32)
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    uv_primvar = primvars_api.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex if len(uvs_np) == len(points_np) else UsdGeom.Tokens.faceVarying,
    )
    uv_primvar.Set(uvs_np)
```

UVs are set once with topology (not time-sampled). The interpolation mode is detected from array
length: per-vertex (`len == num_verts`) or per-face-vertex (`len == num_face_indices`).

This is inside the `if name not in self._meshes:` block (creation-only), since UVs don't change
per frame.

**Import needed:** `Sdf` is already imported.

---

## Step 2: Add material and texture export to `ViewerUSD`

**File:** `newton/_src/viewer/viewer_usd.py`

### 2a. Add new instance state

In `__init__`, add:
```python
self._materials = {}        # material_key -> USD material path
self._texture_dir = None    # textures/ directory path (created lazily)
```

### 2b. New method: `_create_material()`

Creates a `UsdShadeMaterial` with a `UsdPreviewSurface` shader. Parameters: `color` (vec3),
`roughness` (float), `metallic` (float), `texture` (path/data or None).

Material hierarchy under `/root/materials/material_N/`:
- `UsdShade.Material`
- `UsdShade.Shader` (id=`UsdPreviewSurface`) with inputs: `diffuseColor`, `roughness`, `metallic`
- If texture: `UsdShade.Shader` (id=`UsdUVTexture`) connected to diffuseColor, and
  `UsdShade.Shader` (id=`UsdPrimvarReader_float2`) reading the `"st"` primvar

Deduplication: key = `(tuple(color), roughness, metallic, texture_hash)`. Return cached path
if already created.

**Import needed:** `from pxr import UsdShade` (add to the existing pxr import block at line 26).

### 2c. New method: `_export_texture()`

Given texture data (file path string or numpy image array), writes/copies the texture file to
a `textures/` subdirectory next to the USD output file. Returns the relative path for USD
`@textures/texture_N.png@` reference.

- If `texture` is a file path string: copy the file
- If `texture` is numpy array or image data: write as PNG using standard library or numpy

### 2d. Bind materials to instanced rigid body prims

In `log_instances()` (line 297-318), after creating each instance prim, bind the material:

The `materials` parameter is already a `wp.array(dtype=wp.vec4)` containing
`(roughness, metallic, checker, texture_enable)` per instance. Colors come from the `colors`
parameter. We need to also pass texture/mesh data.

**Approach:** Store material info per geometry prototype during `_populate_shapes()`. Add a new
dictionary `self._mesh_materials` that maps mesh prototype name -> `(color, roughness, metallic,
texture)`. When `log_instances()` creates a new instance prim, look up the prototype's material
and call `_create_material()`, then bind it via `UsdShade.MaterialBindingAPI`.

To thread this data through:
- In `ViewerBase._populate_shapes()` (viewer.py ~line 968-976), when `_populate_geometry()` is
  called for MESH/CONVEX_MESH, also store the mesh's material info. Add a new method
  `log_material_info()` on ViewerBase (no-op default) that ViewerUSD overrides to store the
  mapping.
- ViewerUSD's `log_material_info(mesh_name, color, roughness, metallic, texture)` stores in
  `self._mesh_materials[mesh_name]`.
- In `_populate_shapes()`, after calling `_populate_geometry()`, call
  `self.log_material_info(mesh_name, geo_src.color, geo_src.roughness, geo_src.metallic, geo_src.texture)`
  for MESH/CONVEX_MESH types.

For primitive shapes (sphere, box, capsule, etc.) that don't have textures, create a simple
material from the `displayColor` + roughness/metallic values.

In `log_instances()`, when creating a new instance prim:
```python
# look up material for this mesh prototype
if mesh in self._mesh_materials:
    mat_info = self._mesh_materials[mesh]
    mat_path = self._create_material(...)
    UsdShade.MaterialBindingAPI.Apply(instance).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)))
```

---

## Step 3: Split deformable meshes into separate USD prims

**File:** `newton/_src/viewer/viewer.py`, method `_log_triangles()` (line 1357)

### Problem

Currently `_log_triangles()` dumps all particles + all triangle indices into one
`/model/triangles` mesh. Newton has **no** `particle_shape` or `tri_shape` mapping — particles
and shapes are independent systems.

### Solution: Connected component analysis

At `set_model()` time (in ViewerBase), compute connected components from `model.tri_indices`
to identify separate deformable meshes. Store the result for per-frame use.

**New method in ViewerBase: `_build_deformable_mesh_groups()`**

Called once from `set_model()` after `_populate_shapes()`. Uses union-find on vertex indices
referenced by `tri_indices` to find connected components:

```python
def _build_deformable_mesh_groups(self):
    """Group triangles into connected components for separate USD mesh export."""
    if not self.model.tri_count:
        self._deformable_groups = []
        return

    tri_indices = self.model.tri_indices.numpy().reshape(-1, 3)

    # Union-find on particle indices
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[a] = b

    for tri in tri_indices:
        union(tri[0], tri[1])
        union(tri[0], tri[2])

    # Group triangles by component
    from collections import defaultdict
    groups = defaultdict(list)
    for ti, tri in enumerate(tri_indices):
        groups[find(tri[0])].append(ti)

    # For each component, build local index mapping and store
    self._deformable_groups = []
    for group_id, (root, tri_list) in enumerate(sorted(groups.items())):
        group_tris = tri_indices[tri_list]
        global_verts = np.unique(group_tris)
        global_to_local = {g: l for l, g in enumerate(global_verts)}
        local_tris = np.vectorize(global_to_local.get)(group_tris).flatten()

        self._deformable_groups.append({
            "name": f"/model/deformable/mesh_{group_id}",
            "global_vertex_indices": global_verts,     # for extracting from state.particle_q
            "local_tri_indices": local_tris,           # local face indices (0-based)
        })
```

### Modified `_log_triangles()`

Replace the single `log_mesh()` call with per-group calls:

```python
def _log_triangles(self, state):
    if not self._deformable_groups:
        return
    for group in self._deformable_groups:
        # Extract this group's particle positions
        positions = state.particle_q.numpy()[group["global_vertex_indices"]]
        points = wp.array(positions, dtype=wp.vec3, device=self.device)
        indices = wp.array(group["local_tri_indices"], dtype=wp.int32, device=self.device)
        self.log_mesh(
            group["name"],
            points,
            indices,
            hidden=not self.show_triangles,
            backface_culling=False,
        )
```

**Performance note:** The per-frame numpy gather (`particle_q[indices]`) is negligible compared
to simulation cost. The `wp.array()` creation per frame could be optimized with pre-allocated
buffers if needed.

---

## Step 4: MPM sand particles as instanced spheres

**File:** `newton/_src/viewer/viewer.py`, method `_log_particles()` (line 1367)

### Determine which particles are "loose" (not part of any surface)

After `_build_deformable_mesh_groups()`, collect all particle indices that appear in any
deformable group's `global_vertex_indices`. Particles NOT in this set are "loose" particles
(MPM sand, granular, etc.).

Store as `self._loose_particle_indices` (numpy array) and `self._has_loose_particles` (bool).

### Modified `_log_particles()`

Split into two paths:
1. **Surface particles** (in deformable groups): already rendered by `_log_triangles()`, skip
2. **Loose particles**: render via a new `log_particle_spheres()` call on ViewerUSD

```python
def _log_particles(self, state):
    if not self._has_loose_particles:
        return
    loose_positions = state.particle_q.numpy()[self._loose_particle_indices]
    loose_radii = self.model.particle_radius.numpy()[self._loose_particle_indices]
    # ... call viewer-specific method
```

### New method in ViewerUSD: `log_particle_spheres()`

Creates a `UsdGeom.PointInstancer` with a `UsdGeom.Sphere` prototype:

```python
def log_particle_spheres(self, name, positions, radii, colors):
    if name not in self._instancers:
        # Create sphere prototype
        instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))
        sphere = UsdGeom.Sphere.Define(self.stage, self._get_path(name) + "/proto_sphere")
        sphere.GetRadiusAttr().Set(1.0)
        instancer.CreatePrototypesRel().SetTargets([sphere.GetPath()])
        # ... setup ids, protoIndices, displayColor primvar
        self._instancers[name] = instancer

    # Per frame: set positions, scales (= radii), orientations (identity)
    instancer.GetPositionsAttr().Set(positions, self._frame_index)
    instancer.GetScalesAttr().Set(radii_as_scales, self._frame_index)
```

**For non-USD viewers:** The existing `log_points()` path continues to work unchanged. The
split only affects ViewerUSD's rendering of loose particles. We can add a `log_particle_spheres()`
to ViewerBase as a default that calls `log_points()`, and override it only in ViewerUSD.

---

## Step 5: Material binding for deformable meshes

For deformable meshes, we need to pass color/material data from the model to the viewer.

Currently, cloth/soft body meshes in Newton don't store per-mesh appearance data through the
model — they only have particle positions and triangle connectivity. The `Mesh` object used
during `add_cloth_mesh()` is not stored on the model.

**Pragmatic approach:** Apply a default material per deformable group, using the shape color map
(the same Paul Tol palette used for rigid shapes). Each group gets a distinct color. If future
work adds per-particle or per-mesh appearance data, materials can be upgraded then.

---

## Files Modified

| File | Changes |
|------|---------|
| `newton/_src/viewer/viewer_usd.py` | UV export in `log_mesh()`, new `_create_material()`, `_export_texture()`, `log_particle_spheres()`, material binding in `log_instances()`, new pxr imports (`UsdShade`) |
| `newton/_src/viewer/viewer.py` | New `_build_deformable_mesh_groups()`, modified `_log_triangles()` for per-group export, modified `_log_particles()` for loose particle detection, new `log_material_info()` hook, new `log_particle_spheres()` default |

No new files. No public API changes.

---

## Verification

1. **Run a rigid body example with USD export and open in Blender:**
   ```
   uv run -m newton.examples basic_shapes --viewer usd --output-path test_rigid.usd --num-frames 50
   ```
   Verify: animated transforms, materials with colors on each shape, textures on URDF meshes

2. **Run a cloth example with USD export and open in Blender:**
   ```
   uv run -m newton.examples cloth_bending --viewer usd --output-path test_cloth.usd --num-frames 50
   ```
   Verify: separate mesh prims per cloth piece, animated vertex positions, distinct colors

3. **Run an example with both rigid + deformable (cloth_h1, cloth_franka):**
   ```
   uv run -m newton.examples cloth_franka --viewer usd --output-path test_mixed.usd --num-frames 50
   ```
   Verify: robot links have materials + textures, cloth is separate prim with animation

4. **Run an MPM example:**
   ```
   uv run -m newton.examples mpm_granular --viewer usd --output-path test_mpm.usd --num-frames 50
   ```
   Verify: sand particles render as small spheres in Blender

5. **Run existing tests to check for regressions:**
   ```
   uv run --extra dev -m newton.tests.test_examples
   ```

6. **Run pre-commit hooks:**
   ```
   uvx pre-commit run -a
   ```
