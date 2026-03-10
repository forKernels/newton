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
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

import newton

# =============================================================================
# Config
# =============================================================================

_CONFIG_FILE = Path(__file__).resolve().parent / "sim_config.json"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load sim_config.json. Returns empty dict if not found."""
    p = Path(config_path) if config_path else _CONFIG_FILE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


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

    all_folders = sorted([
        d for d in dtc_root.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

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

        props.append({
            "name": d.name,
            "path": glb,
            "collision_path": d / "collision.glb" if (d / "collision.glb").exists() else None,
            "metadata": meta,
        })

    return props


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
                garments.append({
                    "name": item_dir.name,
                    "category": category,
                    "path": obj_path,
                    "is_clean": bool(clean),
                })

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
) -> tuple[np.ndarray, np.ndarray]:
    """Load garment OBJ mesh, applying scale if raw (not clean)."""
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

    return verts, indices


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
    """Add a DTC prop to the scene as a rigid body.

    Loads the collision mesh (or visual mesh as convex hull) and applies
    physics params from DTC metadata.

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

    # Physics config from DTC metadata
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = meta.get("mass_kg", 0.3) * 1000.0  # rough density estimate
    cfg.mu = meta.get("friction_static", 0.5)
    cfg.restitution = meta.get("restitution", 0.15)

    q = wp.quat(*rotation) if rotation else wp.quat_identity()

    # DTC meshes are Y-up, Newton is Z-up — rotate 90 around X
    y_to_z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
    q = q * y_to_z

    xform = wp.transform(p=wp.vec3(*position), q=q)

    # Load mesh
    collision_strategy = meta.get("collision_strategy", "hull")
    collision_prim = meta.get("collision_primitive")

    if static:
        body_idx = -1
    else:
        body_idx = builder.add_body(xform=xform)

    body_xform = xform if not static else None

    # Use primitive collision if available
    if collision_prim and collision_prim["type"] == "sphere":
        center = collision_prim["center"]
        radius = collision_prim["radius"]
        shape_xform = wp.transform(
            p=wp.vec3(center[0], center[1], center[2]),
            q=wp.quat_identity(),
        ) if not static else xform
        builder.add_shape_sphere(
            body=body_idx,
            xform=shape_xform,
            radius=radius,
            cfg=cfg,
        )
    elif collision_prim and collision_prim["type"] == "box":
        center = collision_prim["center"]
        half = collision_prim["half_extents"]
        shape_xform = wp.transform(
            p=wp.vec3(center[0], center[1], center[2]),
            q=wp.quat_identity(),
        ) if not static else xform
        builder.add_shape_box(
            body=body_idx,
            xform=shape_xform,
            hx=half[0], hy=half[1], hz=half[2],
            cfg=cfg,
        )
    else:
        # Load mesh for convex hull collision
        mesh_path = prop["collision_path"] or prop["path"]
        try:
            verts, faces = load_mesh_from_file(mesh_path)
            mesh = newton.Mesh(
                vertices=wp.array(verts, dtype=wp.vec3),
                indices=wp.array(faces.flatten(), dtype=wp.int32),
            )
            builder.add_shape_mesh(
                body=body_idx,
                xform=body_xform if static else None,
                mesh=mesh,
                cfg=cfg,
            )
        except Exception as e:
            print(f"    WARNING: Failed to load mesh for {prop['name']}: {e}")
            # Fallback: bounding box from dimensions
            dims = meta.get("dimensions_m", [0.1, 0.1, 0.1])
            builder.add_shape_box(
                body=body_idx,
                hx=dims[0] / 2, hy=dims[1] / 2, hz=dims[2] / 2,
                cfg=cfg,
            )

    return body_idx


def add_garment_as_cloth(
    builder: newton.ModelBuilder,
    garment: dict,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    scale: float = 1.0,
    density: float = 0.2,
    tri_ke: float = 1e3,
    tri_ka: float = 1e3,
    tri_kd: float = 1e-1,
    edge_ke: float = 0.01,
    edge_kd: float = 1e-2,
    particle_radius: float = 0.005,
):
    """Add a Maria garment as cloth particles.

    Args:
        builder: Newton ModelBuilder.
        garment: Dict from discover_garments().
        position: Drop position (x, y, z).
        scale: Mesh scale factor.
        density: Cloth area density [kg/m²].
        tri_ke/ka/kd: Stretch stiffness / area / damping.
        edge_ke/kd: Bending stiffness / damping.
        particle_radius: Collision radius per particle.
    """
    verts, indices = load_garment_mesh(garment, scale=scale)

    builder.add_cloth_mesh(
        pos=wp.vec3(*position),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
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
# Cloth Presets (maps to fabric types)
# =============================================================================

CLOTH_PRESETS = {
    "silk": {
        "density": 0.08,
        "tri_ke": 5e2, "tri_ka": 5e2, "tri_kd": 5e-2,
        "edge_ke": 1e-4, "edge_kd": 1e-3,
    },
    "cotton": {
        "density": 0.15,
        "tri_ke": 1e3, "tri_ka": 1e3, "tri_kd": 1e-1,
        "edge_ke": 1e-2, "edge_kd": 1e-2,
    },
    "denim": {
        "density": 0.35,
        "tri_ke": 5e3, "tri_ka": 5e3, "tri_kd": 5e-1,
        "edge_ke": 5e-2, "edge_kd": 5e-2,
    },
    "leather": {
        "density": 0.50,
        "tri_ke": 1e4, "tri_ka": 1e4, "tri_kd": 1.0,
        "edge_ke": 1e-1, "edge_kd": 1e-1,
    },
    "rubber": {
        "density": 0.80,
        "tri_ke": 2e4, "tri_ka": 2e4, "tri_kd": 2.0,
        "edge_ke": 5e-1, "edge_kd": 5e-1,
    },
}


# =============================================================================
# Simulation Runner
# =============================================================================


def build_solver(
    model: newton.Model,
    solver_type: str = "vbd",
    iterations: int = 10,
    self_contact: bool = True,
):
    """Create a solver for the given model.

    Args:
        solver_type: "vbd", "xpbd", "style3d", "mujoco"
        iterations: Solver iterations per substep.
        self_contact: Enable particle self-contact (cloth).
    """
    if solver_type == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=iterations,
            particle_enable_self_contact=self_contact,
        )
    elif solver_type == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=iterations)
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
) -> newton.State:
    """Run a simulation loop and return the final state.

    Args:
        model: Finalized Newton model.
        solver: Solver instance.
        num_frames: Total frames to simulate.
        substeps: Substeps per frame.
        dt: Time step per substep.
        viewer: Optional viewer for recording (ViewerUSD, ViewerNull, etc.)

    Returns:
        Final simulation state.
    """
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Evaluate initial forward kinematics

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    sub_dt = dt / substeps

    for frame in range(num_frames):
        for sub in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

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
            capture_output=True, text=True, errors="replace", timeout=120,
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
