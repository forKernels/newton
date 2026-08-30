"""
Newton Simulation Utilities
=============================
Shared infrastructure for all Newton-based simulation pipelines.

Provides:
  - DTC prop discovery and loading (GLB → Newton Mesh → rigid body shapes)
  - Maria garment discovery and loading (OBJ → cloth mesh)
  - Scene building helpers (ground, table, prop placement)
  - Config loading from sim_config.json
  - USD export helpers
  - Job manifest generation for 1M+ sim runs

All paths come from sim_config.json for portability across machines.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp

import newton

# =============================================================================
# Config
# =============================================================================

_CONFIG_FILE = Path(__file__).resolve().parent / "sim_config.json"


def load_config(config_path: str | Path | None = None, env: str | None = None) -> dict:
    """Load sim_config.json, resolving the active environment.

    Environment resolution order:
      1. ``env`` argument (e.g. from --env CLI flag)
      2. ``SIM_ENV`` environment variable
      3. Auto-detect from machine username

    The ``environments`` dict in the config maps machine names to path overrides.
    Top-level keys (blender_preset_dir, dtc_unique_only, ...) are shared across
    all environments.
    """
    p = Path(config_path) if config_path else _CONFIG_FILE
    if not p.exists():
        return {}

    with open(p) as f:
        raw = json.load(f)

    envs = raw.pop("environments", None)
    # Strip comment keys
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    if envs is None:
        return raw  # legacy flat config

    # Resolve which environment to use
    env_name = env or os.environ.get("SIM_ENV")
    if not env_name:
        # Auto-detect from username
        username = os.environ.get("USERNAME", os.environ.get("USER", "")).lower()
        if username == "discokid":
            env_name = "disco_machine"
        elif username == "davidclabaugh":
            # Could be either work machine — default to work_home
            env_name = "work_home"
        else:
            env_name = list(envs.keys())[0] if envs else None

    if env_name and env_name in envs:
        env_paths = envs[env_name]
        raw.update(env_paths)
        raw["_active_env"] = env_name
        print(f"  Config: environment={env_name}")
    else:
        available = list(envs.keys())
        print(f"  WARNING: Unknown environment '{env_name}'. Available: {available}")
        print(f"  Set SIM_ENV or pass --env. Falling back to first: {available[0]}")
        raw.update(envs[available[0]])
        raw["_active_env"] = available[0]

    return raw


# =============================================================================
# DTC Prop Discovery
# =============================================================================


def _group_by_shape(names: list[str]) -> dict[str, list[str]]:
    """Group DTC folder names by physical shape (skip color variants)."""
    groups = defaultdict(list)
    for name in sorted(names):
        clean = re.sub(r"_TU$", "", name)
        asin = re.match(r"^(.+?)_(B[A-Z0-9]{9,}|TS\d+)_(.+)$", clean)
        if asin:
            groups[f"{asin.group(1)}_{asin.group(2)}"].append(name)
            continue
        book = re.match(r"^(Book_\d+)_(.+)$", clean)
        if book:
            groups[book.group(1)].append(name)
            continue
        toy = re.match(r"^(\w+_Toy)_([A-F0-9]+)_(.+)$", clean)
        if toy:
            groups[f"{toy.group(1)}_{toy.group(2)}"].append(name)
            continue
        groups[clean].append(name)
    return dict(groups)


def discover_dtc_props(
    dtc_dir: str | Path,
    unique_only: bool = True,
) -> list[dict]:
    """Discover DTC sim-ready props.

    Returns list of dicts:
      name, path (to GLB), metadata (dict with mass, friction, dims, etc.)
    """
    dtc_root = Path(dtc_dir)
    if not dtc_root.is_dir():
        print(f"  WARNING: DTC dir not found: {dtc_dir}")
        return []

    all_folders = sorted([d for d in dtc_root.iterdir() if d.is_dir() and (d / "metadata.json").exists()])

    if unique_only:
        groups = _group_by_shape([d.name for d in all_folders])
        unique_names = set()
        for variants in groups.values():
            unique_names.add(sorted(variants, key=lambda x: (len(x), x))[0])
        folders = [d for d in all_folders if d.name in unique_names]
    else:
        folders = all_folders

    props = []
    for d in folders:
        glb = d / "visual_textured.glb"
        if not glb.exists():
            glb = d / "visual.glb"
        if not glb.exists():
            continue

        with open(d / "metadata.json") as f:
            meta = json.load(f)

        props.append(
            {
                "name": d.name,
                "path": glb,
                "collision_path": d / "collision.glb" if (d / "collision.glb").exists() else None,
                "metadata": meta,
            }
        )

    return props


# =============================================================================
# USDA Props Discovery (Props_USDA layout: flat dir with .usda + .physics.json)
# =============================================================================


def discover_usda_props(props_dir: str | Path) -> list[dict]:
    """Discover USDA-format props with .physics.json sidecars.

    Expected layout::

        Props_USDA/
            Bowl_01.usda
            Bowl_01.physics.json
            Avocado_basket.usda
            Avocado_basket.physics.json
            textures/

    Returns list of dicts compatible with the DTC prop format:
      name, path (to .usda), collision_path (None), metadata (physics dict).
    """
    root = Path(props_dir)
    if not root.is_dir():
        print(f"  WARNING: USDA props dir not found: {props_dir}")
        return []

    props = []
    for usda_file in sorted(root.glob("*.usda")):
        physics_json = usda_file.with_suffix(".physics.json")
        if not physics_json.exists():
            continue

        with open(physics_json) as f:
            physics_data = json.load(f)

        # physics.json has {"objects": [{"name": ..., "physics": {...}}]}
        obj_list = physics_data.get("objects", [])
        if not obj_list:
            continue

        phys = obj_list[0].get("physics", {})

        # Remap to DTC-compatible metadata keys
        meta = {
            "mass_kg": phys.get("mass_kg", 0.5),
            "friction_static": phys.get("friction_static", 0.5),
            "restitution": phys.get("restitution", 0.2),
            "dimensions_m": phys.get("dimensions_m", [0.1, 0.1, 0.1]),
            "collision_primitive": phys.get("collision_primitive"),
            "collision_strategy": "primitive" if phys.get("collision_primitive") else "hull",
        }

        props.append(
            {
                "name": usda_file.stem,
                "path": usda_file,
                "collision_path": None,
                "metadata": meta,
                "format": "usda",
            }
        )

    return props


def discover_usda_scenes(scenes_dir: str | Path) -> list[dict]:
    """Discover USDA-format scene files (furniture) with .physics.json sidecars.

    Expected layout::

        Scenes_USDA/
            Bed_01.usda
            Bed_01.physics.json
            Chair/
                Chair_01.usda
                Chair_01.physics.json
            Table/
                Table_01.usda
                Table_01.physics.json
            Sofa/
                ...

    Returns list of dicts:
      name, path (to .usda), metadata (physics dict), category.
    """
    root = Path(scenes_dir)
    if not root.is_dir():
        print(f"  WARNING: Scenes dir not found: {scenes_dir}")
        return []

    scenes = []

    def _scan_dir(d: Path, category: str = ""):
        for usda_file in sorted(d.glob("*.usda")):
            physics_json = usda_file.with_suffix(".physics.json")
            meta = {}
            if physics_json.exists():
                with open(physics_json) as f:
                    physics_data = json.load(f)
                obj_list = physics_data.get("objects", [])
                if obj_list:
                    meta = obj_list[0].get("physics", {})

            scenes.append(
                {
                    "name": usda_file.stem,
                    "path": usda_file,
                    "category": category or usda_file.stem.split("_")[0],
                    "metadata": meta,
                }
            )

    # Scan top-level files
    _scan_dir(root)

    # Scan subdirectories (Chair/, Table/, Sofa/, etc.)
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name != "textures":
            _scan_dir(subdir, category=subdir.name)

    return scenes


def discover_all_props(config: dict) -> list[dict]:
    """Discover props from both DTC (GLB) and USDA sources.

    Checks config keys: dtc_dir, props_usda_dir, props_dir.
    Returns combined list.
    """
    all_props = []

    # DTC GLB props
    dtc_dir = config.get("dtc_dir", "")
    if dtc_dir and Path(dtc_dir).is_dir():
        dtc_props = discover_dtc_props(
            dtc_dir,
            unique_only=config.get("dtc_unique_only", True),
        )
        all_props.extend(dtc_props)

    # USDA props
    usda_dir = config.get("props_usda_dir", config.get("props_dir", ""))
    if usda_dir and Path(usda_dir).is_dir():
        usda_props = discover_usda_props(usda_dir)
        all_props.extend(usda_props)

    return all_props


# =============================================================================
# Garment Discovery
# =============================================================================


def discover_garments(garment_dir: str | Path) -> list[dict]:
    """Discover Maria dataset garments.

    Returns list of dicts: name, category, path (to OBJ).
    Prefers _clean.obj (already in meters), falls back to _sim.obj.
    """
    root = Path(garment_dir)
    if not root.is_dir():
        print(f"  WARNING: Garment dir not found: {garment_dir}")
        return []

    garments = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith(("_", ".")):
            continue

        category = category_dir.name
        for item_dir in sorted(category_dir.iterdir()):
            if not item_dir.is_dir():
                continue

            # Prefer clean mesh, fall back to sim mesh
            clean = list(item_dir.glob("*_clean.obj"))
            sim = list(item_dir.glob("*_sim.obj"))
            obj_path = clean[0] if clean else (sim[0] if sim else None)

            if obj_path:
                garments.append(
                    {
                        "name": item_dir.name,
                        "category": category,
                        "path": obj_path,
                        "is_clean": bool(clean),
                    }
                )

    return garments


# =============================================================================
# Mesh Loading (GLB/OBJ → Newton)
# =============================================================================

try:
    import trimesh as _trimesh

    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


def load_mesh_from_file(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh file and return (vertices [N,3], indices [M,3]).

    Supports GLB, OBJ, STL via trimesh.
    """
    if not _HAS_TRIMESH:
        raise ImportError("trimesh is required: pip install trimesh[easy]")

    scene = _trimesh.load(str(filepath), force="scene")
    if isinstance(scene, _trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, _trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes in {filepath}")
        mesh = _trimesh.util.concatenate(meshes)
    elif isinstance(scene, _trimesh.Trimesh):
        mesh = scene
    else:
        raise ValueError(f"Unexpected type from {filepath}: {type(scene)}")

    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def load_garment_mesh(
    garment: dict,
    scale: float = 1.0,
    stitch: bool = True,
    stitch_threshold: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    """Load garment OBJ mesh, applying scale if raw (not clean).

    Args:
        garment: Dict from discover_garments() with 'path' and 'is_clean' keys.
        scale: Additional scale factor.
        stitch: If True, merge near-duplicate vertices to stitch disconnected panels.
        stitch_threshold: Distance threshold [meters] for vertex merging.
    """
    verts, indices = load_mesh_from_file(garment["path"])

    if not garment["is_clean"]:
        # Raw mesh: cm → m, flip Y (rotate 180 around Y)
        verts *= 0.01
        verts[:, 0] *= -1  # flip X (180 around Y)
        verts[:, 2] *= -1  # flip Z (180 around Y)

        # Center XY, floor at Z=0
        center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2
        center_y = (verts[:, 1].min() + verts[:, 1].max()) / 2
        min_z = verts[:, 2].min()
        verts[:, 0] -= center_x
        verts[:, 1] -= center_y
        verts[:, 2] -= min_z

    if scale != 1.0:
        verts *= scale

    # Stitch disconnected panels by merging near-duplicate vertices at seams
    if stitch:
        verts, indices = _stitch_mesh(verts, indices, stitch_threshold)

    return verts, indices


def _stitch_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    merge_threshold: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge near-duplicate vertices to stitch disconnected garment panels.

    Maria garment _sim.obj meshes are exported with separate panels (front, back,
    sleeves) that share boundary vertices at the seams but aren't topologically
    connected. This merges vertices within a threshold distance to create a single
    connected mesh suitable for cloth simulation.
    """
    from scipy.spatial import cKDTree

    n_verts = len(verts)
    tree = cKDTree(verts)
    pairs = tree.query_pairs(merge_threshold)

    if not pairs:
        return verts, faces

    # Union-find for merging
    parent = list(range(n_verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    # Build old→new vertex mapping
    roots = {}
    new_idx = 0
    old_to_new = np.zeros(n_verts, dtype=np.int32)
    for v in range(n_verts):
        root = find(v)
        if root not in roots:
            roots[root] = new_idx
            new_idx += 1
        old_to_new[v] = roots[root]

    n_merged = n_verts - new_idx
    if n_merged > 0:
        print(f"  stitch: Merged {n_merged} vertices ({n_verts} → {new_idx})")

    # Average positions of merged vertices
    new_verts = np.zeros((new_idx, 3), dtype=np.float32)
    counts = np.zeros(new_idx, dtype=np.int32)
    for v in range(n_verts):
        nv = old_to_new[v]
        new_verts[nv] += verts[v]
        counts[nv] += 1
    new_verts /= counts[:, np.newaxis]

    # Remap face indices and remove degenerate triangles
    new_faces = old_to_new[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    n_degenerate = np.sum(~valid)
    if n_degenerate > 0:
        print(f"  stitch: Removed {n_degenerate} degenerate triangles")
        new_faces = new_faces[valid]

    return new_verts, new_faces


# =============================================================================
# Scene Building Helpers
# =============================================================================


def add_ground_plane(builder: newton.ModelBuilder):
    """Add a ground plane at Z=0."""
    builder.add_ground_plane()


def add_table(
    builder: newton.ModelBuilder,
    position: tuple[float, float, float] = (0.0, 0.0, 0.4),
    size: tuple[float, float, float] = (0.4, 0.4, 0.02),
):
    """Add a static table (box shape) to the scene.

    Args:
        position: Center of table top (x, y, z).
        size: Half-extents (hx, hy, hz).
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.6
    cfg.restitution = 0.1

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            p=wp.vec3(*position),
            q=wp.quat_identity(),
        ),
        hx=size[0],
        hy=size[1],
        hz=size[2],
        cfg=cfg,
    )


def add_dtc_prop_as_rigid_body(
    builder: newton.ModelBuilder,
    prop: dict,
    position: tuple[float, float, float] = (0.0, 0.0, 0.5),
    rotation: tuple[float, float, float, float] | None = None,
    static: bool = False,
) -> int:
    """Add a DTC prop with separate visual + collision shapes (game-engine style).

    Visual mesh: full-detail, VISIBLE only (no collision).
    Collision shape: simplified box/hull, COLLIDE only (not visible).

    Args:
        builder: Newton ModelBuilder.
        prop: Dict from discover_dtc_props().
        position: World position (x, y, z).
        rotation: Quaternion (x, y, z, w) or None for identity.
        static: If True, body is fixed (kinematic).

    Returns:
        Body index.
    """
    meta = prop["metadata"]

    # Collision config from DTC metadata
    collision_cfg = newton.ModelBuilder.ShapeConfig()
    collision_cfg.density = meta.get("mass_kg", 0.3) * 1000.0
    collision_cfg.mu = meta.get("friction_static", 0.5)
    collision_cfg.restitution = meta.get("restitution", 0.15)
    collision_cfg.is_visible = False
    collision_cfg.has_shape_collision = True
    collision_cfg.has_particle_collision = True

    # Visual config: visible only, no collision
    visual_cfg = newton.ModelBuilder.ShapeConfig()
    visual_cfg.is_visible = True
    visual_cfg.has_shape_collision = False
    visual_cfg.has_particle_collision = False

    q = wp.quat(*rotation) if rotation else wp.quat_identity()

    # DTC meshes are Y-up, Newton is Z-up — rotate 90 around X
    y_to_z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
    q = q * y_to_z

    xform = wp.transform(p=wp.vec3(*position), q=q)

    collision_prim = meta.get("collision_primitive")

    if static:
        body_idx = -1
    else:
        body_idx = builder.add_body(xform=xform)

    body_xform = xform if not static else None

    # --- 1. Add visual mesh (full detail, VISIBLE only) ---
    visual_mesh_path = prop["path"]  # always the full visual mesh
    verts = None
    try:
        verts, faces = load_mesh_from_file(visual_mesh_path)
        visual_mesh = newton.Mesh(
            vertices=verts,
            indices=faces.flatten().astype(np.int32),
        )
        builder.add_shape_mesh(
            body=body_idx,
            xform=body_xform if static else None,
            mesh=visual_mesh,
            cfg=visual_cfg,
        )
    except Exception as e:
        print(f"    WARNING: Failed to load visual mesh for {prop['name']}: {e}")

    # --- 2. Add collision shape (simplified, COLLIDE only, not visible) ---
    if collision_prim and collision_prim["type"] == "sphere":
        center = collision_prim["center"]
        radius = collision_prim["radius"]
        shape_xform = (
            wp.transform(
                p=wp.vec3(center[0], center[1], center[2]),
                q=wp.quat_identity(),
            )
            if not static
            else xform
        )
        builder.add_shape_sphere(
            body=body_idx,
            xform=shape_xform,
            radius=radius,
            cfg=collision_cfg,
        )
    elif collision_prim and collision_prim["type"] == "box":
        center = collision_prim["center"]
        half = collision_prim["half_extents"]
        shape_xform = (
            wp.transform(
                p=wp.vec3(center[0], center[1], center[2]),
                q=wp.quat_identity(),
            )
            if not static
            else xform
        )
        builder.add_shape_box(
            body=body_idx,
            xform=shape_xform,
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=collision_cfg,
        )
    else:
        # Fallback: bounding box collision from mesh extents or metadata
        try:
            if verts is not None:
                bbox_min = np.min(verts, axis=0)
                bbox_max = np.max(verts, axis=0)
                half_extents = (bbox_max - bbox_min) / 2.0
                center = (bbox_min + bbox_max) / 2.0
                builder.add_shape_box(
                    body=body_idx,
                    xform=wp.transform(
                        p=wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                        q=wp.quat_identity(),
                    )
                    if not static
                    else None,
                    hx=float(half_extents[0]),
                    hy=float(half_extents[1]),
                    hz=float(half_extents[2]),
                    cfg=collision_cfg,
                )
            else:
                raise ValueError("No visual mesh loaded")
        except Exception:
            dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
            builder.add_shape_box(
                body=body_idx,
                hx=dims[0] / 2,
                hy=dims[1] / 2,
                hz=dims[2] / 2,
                cfg=collision_cfg,
            )

    return body_idx


def _detect_y_up_mesh(verts: np.ndarray) -> bool:
    """Detect if a mesh is Y-up by checking if Y extent > Z extent * 1.5."""
    extent = np.max(verts, axis=0) - np.min(verts, axis=0)
    return float(extent[1]) > float(extent[2]) * 1.5 and int(np.argmax(extent)) == 1


def _rotate_verts_y_to_z(verts: np.ndarray) -> np.ndarray:
    """Rotate vertices from Y-up to Z-up: (x, y, z) -> (x, -z, y)."""
    out = np.empty_like(verts)
    out[:, 0] = verts[:, 0]
    out[:, 1] = -verts[:, 2]
    out[:, 2] = verts[:, 1]
    return out


def _lay_flat(verts: np.ndarray) -> np.ndarray:
    """Tip garment from standing upright to lying flat on XY plane."""
    return _rotate_verts_y_to_z(verts)


def _center_and_floor(verts: np.ndarray) -> np.ndarray:
    """Center XY at origin and floor Z at 0."""
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    verts = verts.copy()
    verts[:, 0] -= center[0]
    verts[:, 1] -= center[1]
    verts[:, 2] -= bbox_min[2]
    return verts


def add_garment_as_cloth(
    builder: newton.ModelBuilder,
    garment: dict,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    scale: float = 1.0,
    density: float = 0.2,
    tri_ke: float = 1e2,
    tri_ka: float = 1e2,
    tri_kd: float = 1.5e-6,
    edge_ke: float = 1e-4,
    edge_kd: float = 1e-3,
    particle_radius: float = 0.008,
    lay_flat: bool = True,
    **kwargs,
):
    """Add a Maria garment as cloth particles.

    Uses parameters validated against Newton's cloth_dataset_pickup example.

    Args:
        builder: Newton ModelBuilder.
        garment: Dict from discover_garments().
        position: Drop position (x, y, z).
        scale: Mesh scale factor (applied by add_cloth_mesh, not pre-baked).
        density: Cloth area density [kg/m²].
        tri_ke/ka/kd: Stretch stiffness / area / damping.
        edge_ke/kd: Bending stiffness / damping.
        particle_radius: Collision radius per particle.
        lay_flat: If True, tip standing garment to lie flat.
    """
    verts, indices = load_garment_mesh(garment, scale=1.0)  # scale via add_cloth_mesh

    # Auto-detect and fix Y-up meshes
    if _detect_y_up_mesh(verts):
        verts = _rotate_verts_y_to_z(verts)

    # Lay flat so garment lies on XY plane instead of standing upright
    if lay_flat:
        verts = _lay_flat(verts)

    # Center XY, floor Z at 0
    verts = _center_and_floor(verts)

    # Convert to wp.vec3 list for add_cloth_mesh
    vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]

    builder.add_cloth_mesh(
        pos=wp.vec3(*position),
        rot=wp.quat_identity(),
        scale=scale,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=indices.flatten().tolist(),
        density=density,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
        particle_radius=particle_radius,
    )


def add_cloth_grid(
    builder: newton.ModelBuilder,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    size: float = 1.0,
    resolution: int = 30,
    density: float = 0.2,
    tri_ke: float = 1e3,
    tri_ka: float = 1e3,
    tri_kd: float = 1e-1,
    edge_ke: float = 0.01,
    edge_kd: float = 1e-2,
    fix_left: bool = False,
    fix_right: bool = False,
    fix_top: bool = False,
    fix_bottom: bool = False,
):
    """Add a procedural cloth grid."""
    cell = size / resolution
    builder.add_cloth_grid(
        pos=wp.vec3(*position),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=cell,
        cell_y=cell,
        mass=density * cell * cell,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
        fix_left=fix_left,
        fix_right=fix_right,
        fix_top=fix_top,
        fix_bottom=fix_bottom,
    )


# =============================================================================
# Cloth Presets (unified — derived from clean_cloth_dataset.FABRICS)
# =============================================================================

# Import the single source of truth for fabric properties.  The newton-specific
# sub-dict is extracted so callers can keep using CLOTH_PRESETS["cotton"] etc.
# If the import fails (e.g. running outside the disco dir), fall back to a
# self-contained copy of the five most common presets.

try:
    from fabric_library import FABRICS as _FABRICS

    CLOTH_PRESETS: dict[str, dict] = {}
    for _name, _props in _FABRICS.items():
        _n = _props.get("newton", {})
        CLOTH_PRESETS[_name] = {
            "density": _n.get("density", _props.get("density", 0.15)),
            "tri_ke": _n.get("tri_ke", 1e2),
            "tri_ka": _n.get("tri_ka", 1e2),
            "tri_kd": _n.get("tri_kd", 1.5e-6),
            "edge_ke": _n.get("edge_ke", 1e-4),
            "edge_kd": _n.get("edge_kd", 1e-3),
            "particle_radius": _n.get("particle_radius", 0.008),
            "self_contact_radius": _n.get("self_contact_radius", 0.002),
            "contact_mu": _n.get("contact_mu", 0.6),
        }
except ImportError:
    # Fallback: standalone Newton presets calibrated to cloth_dataset_pickup example.
    # tri_ke/ka ~1e2, tri_kd ~1e-6, edge_ke ~1e-4, edge_kd ~1e-3 are the proven
    # VBD-stable baseline. Heavier fabrics scale up slightly but stay well below
    # the previous 1e3+ values that caused mesh-ball collapse.
    CLOTH_PRESETS = {
        "silk": {
            "density": 0.08,
            "tri_ke": 5e1,
            "tri_ka": 1e0,
            "tri_kd": 1e-6,
            "edge_ke": 5e-5,
            "edge_kd": 5e-4,
            "particle_radius": 0.008,
        },
        "cotton": {
            "density": 0.15,
            "tri_ke": 1e2,
            "tri_ka": 2e0,
            "tri_kd": 1.5e-6,
            "edge_ke": 1e-4,
            "edge_kd": 1e-3,
            "particle_radius": 0.008,
        },
        "denim": {
            "density": 0.35,
            "tri_ke": 2e2,
            "tri_ka": 5e0,
            "tri_kd": 3e-6,
            "edge_ke": 5e-4,
            "edge_kd": 3e-3,
            "particle_radius": 0.008,
        },
        "leather": {
            "density": 0.50,
            "tri_ke": 4e2,
            "tri_ka": 4e2,
            "tri_kd": 5e-6,
            "edge_ke": 1e-3,
            "edge_kd": 5e-3,
            "particle_radius": 0.008,
        },
        "rubber": {
            "density": 0.80,
            "tri_ke": 8e2,
            "tri_ka": 8e2,
            "tri_kd": 1e-5,
            "edge_ke": 5e-3,
            "edge_kd": 1e-2,
            "particle_radius": 0.008,
        },
    }


# =============================================================================
# Simulation Runner
# =============================================================================


def finalize_cloth_model(
    builder: newton.ModelBuilder,
    soft_contact_ke: float = 100.0,
    soft_contact_kd: float = 2e-3,
    soft_contact_mu: float = 0.8,
    solver_type: str = "vbd",
) -> newton.Model:
    """Finalize a cloth model with proven contact parameters.

    Applies the critical edge_rest_angle.zero_() workaround required
    for VBD solver stability (from Newton's cloth_dataset_pickup example).
    This is ONLY applied for VBD — XPBD handles bending differently.
    """
    model = builder.finalize()
    model.soft_contact_ke = soft_contact_ke
    model.soft_contact_kd = soft_contact_kd
    model.soft_contact_mu = soft_contact_mu

    # CRITICAL: Zero out edge rest angles — workaround for VBD bending instability.
    # Without this, bending elements fight each other and collapse the mesh into a ball.
    # Only for VBD — XPBD handles bending via constraint projection, not energy minimization.
    if solver_type == "vbd":
        model.edge_rest_angle.zero_()

    return model


def build_solver(
    model: newton.Model,
    solver_type: str = "vbd",
    iterations: int = 10,
    self_contact: bool = True,
    collision_margin: float = 0.01,
):
    """Create a solver for the given model.

    Args:
        solver_type: "vbd", "xpbd", "style3d", "mujoco"
        iterations: Solver iterations per substep.
        self_contact: Enable particle self-contact (cloth).
        collision_margin: Soft contact margin for collision pipeline.
    """
    if solver_type == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            particle_enable_self_contact=self_contact,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.003,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
            particle_topological_contact_filter_threshold=5,
            particle_rest_shape_contact_exclusion_radius=0.01,
            rigid_contact_k_start=model.soft_contact_ke,
        )
    elif solver_type == "xpbd":
        return newton.solvers.SolverXPBD(
            model,
            iterations=iterations,
            enable_self_collision=self_contact,
            self_collision_radius=0.002,
            self_collision_margin=0.003,
        )
    elif solver_type == "style3d":
        return newton.solvers.SolverStyle3D(model, iterations=iterations)
    elif solver_type == "mujoco":
        return newton.solvers.SolverMuJoCo(model)
    else:
        raise ValueError(f"Unknown solver: {solver_type}")


def run_simulation(
    model: newton.Model,
    solver,
    num_frames: int = 150,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
    collision_margin: float = 0.01,
) -> newton.State:
    """Run a simulation loop and return the final state.

    Args:
        model: Finalized Newton model.
        solver: Solver instance.
        num_frames: Total frames to simulate.
        substeps: Substeps per frame.
        dt: Frame time step (divided by substeps internally).
        viewer: Optional viewer for recording (ViewerUSD, ViewerNull, etc.)
        collision_margin: Soft contact margin for collision pipeline.

    Returns:
        Final simulation state.
    """
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Use explicit collision pipeline with margin (matches cloth_dataset_pickup)
    collision_pipeline = newton.CollisionPipeline(
        model,
        soft_contact_margin=collision_margin,
    )
    contacts = collision_pipeline.contacts()

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    sub_dt = dt / substeps

    for frame in range(num_frames):
        for sub in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


# =============================================================================
# Output / Export
# =============================================================================


def _ensure_openusd_env():
    """Auto-detect OPENUSD_ROOT if not set."""
    if os.environ.get("OPENUSD_ROOT"):
        return
    candidates = [
        "C:/_tools/OpenUSD/25.08",
        "C:/Program Files/OpenUSD",
    ]
    for c in candidates:
        if Path(c).is_dir():
            os.environ["OPENUSD_ROOT"] = c
            return


def create_viewer(
    output_path: str | Path | None = None,
    viewer_type: str = "null",
):
    """Create a Newton viewer for recording or display.

    Args:
        output_path: Path for USD/file output. Required for "usd" and "file".
        viewer_type: "null" (headless), "usd" (USD export), "file" (binary), "gl" (interactive).
    """
    if viewer_type == "usd":
        _ensure_openusd_env()
        return newton.viewer.ViewerUSD(str(output_path))
    elif viewer_type == "null":
        return newton.viewer.ViewerNull()
    elif viewer_type == "file":
        return newton.viewer.ViewerFile(str(output_path))
    elif viewer_type == "gl":
        return newton.viewer.ViewerGL()
    else:
        raise ValueError(f"Unknown viewer type: {viewer_type}")


def create_blend_file(usda_path: str | Path, blend_path: str | Path, blender_exe: str = "blender"):
    """Create a .blend file that imports the given USDA.

    Runs Blender in background mode with a small Python script that
    imports the USD stage and saves.
    """
    usda_path = Path(usda_path).resolve()
    blend_path = Path(blend_path).resolve()

    if not usda_path.exists():
        print(f"  WARNING: USDA not found at {usda_path}, skipping .blend creation")
        return

    import_script = f'''
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=r"{usda_path}")
bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")
'''
    try:
        result = subprocess.run(
            [blender_exe, "--background", "--python-expr", import_script],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        if result.returncode == 0:
            print(f"  Blender file: {blend_path}")
        else:
            print(f"  WARNING: Blender export failed: {result.stderr[-300:]}")
    except FileNotFoundError:
        print(f"  WARNING: Blender not found at '{blender_exe}', skipping .blend creation")
    except subprocess.TimeoutExpired:
        print("  WARNING: Blender timed out creating .blend file")


def write_sim_metadata(
    output_path: str | Path,
    sim_type: str,
    seed: int,
    **kwargs,
):
    """Write simulation metadata JSON alongside the output."""
    meta = {
        "sim_type": sim_type,
        "seed": seed,
        **kwargs,
    }
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Test Mode Defaults
# =============================================================================

TEST_FRAMES = 10  # quick smoke-test: just enough to verify the full pipeline


# =============================================================================
# Spring Force Assist
# =============================================================================


def create_spring_assist(
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
    """Create a SpringForceAssist from a finalized Newton model.

    Builds a mass-spring network (stretching + bending) from the model's
    triangle topology and returns an object whose ``apply(state)`` method
    injects spring forces into ``state.particle_f``.

    Call ``apply()`` between ``state.clear_forces()`` and ``solver.step()``
    in the simulation loop.

    Args:
        model: Finalized Newton model (must have particle_q, tri_indices).
        ks: Stretching spring stiffness (1-ring edges).
        kb: Bending spring stiffness (2-ring neighbors).
        kd_stretch: Damping for stretching springs.
        kd_bend: Damping for bending springs.
        boost_factor: Multiplier on all spring forces.
        stuck_threshold: Displacement [m] below which a particle is "stuck".
        stuck_only: If True, only inject forces for stuck particles.
        verbose: Print spring network statistics.

    Returns:
        SpringForceAssist instance (or None if no particles).
    """
    from spring_force_assist import SpringForceAssist

    if model.particle_count == 0:
        if verbose:
            print("  create_spring_assist: No particles, skipping.")
        return None

    return SpringForceAssist(
        model,
        ks=ks,
        kb=kb,
        kd_stretch=kd_stretch,
        kd_bend=kd_bend,
        boost_factor=boost_factor,
        stuck_threshold=stuck_threshold,
        stuck_only=stuck_only,
        verbose=verbose,
    )
