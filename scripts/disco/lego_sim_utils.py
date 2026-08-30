"""
Lego Simulation Utilities (Blender)
====================================
Shared helpers for Lego rigid body simulation scripts.

Provides: model discovery, brick separation, rigid body setup,
ground plane creation, mass estimation, and material presets.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# =============================================================================
# Material presets  --  density [kg/m³], friction, restitution (bounciness)
# =============================================================================

LEGO_MATERIALS = {
    "ABS": {"density": 1050.0, "friction": 0.5, "restitution": 0.3},
    "Metal": {"density": 7800.0, "friction": 0.4, "restitution": 0.2},
    "Rubber": {"density": 1200.0, "friction": 0.8, "restitution": 0.6},
    "Wood": {"density": 600.0, "friction": 0.5, "restitution": 0.2},
    "Glass": {"density": 2500.0, "friction": 0.3, "restitution": 0.1},
}

LEGO_MATERIAL_NAMES = list(LEGO_MATERIALS.keys())

# Rigid body world defaults
GROUND_SIZE = 10.0  # metres
RBD_SUBSTEPS = 10
RBD_SOLVER_ITERATIONS = 10


# =============================================================================
# Discovery
# =============================================================================


def find_lego_models(model_dir: str, specific: str | None = None) -> list[dict]:
    """Find Lego model .blend files.

    Returns list of dicts with keys: model_id, blend_path.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if specific:
        p = root / specific
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return [{"model_id": p.stem, "blend_path": str(p)}]

    blends = sorted(root.glob("*.blend"))
    if not blends:
        raise FileNotFoundError(f"No .blend files found in: {model_dir}")
    return [{"model_id": b.stem, "blend_path": str(b)} for b in blends]


# =============================================================================
# Import and brick separation
# =============================================================================


def import_lego_model(blend_path: str) -> list[bpy.types.Object]:
    """Append all objects from a Lego .blend file into the current scene.

    Returns list of mesh objects.
    """
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    mesh_objects = []
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None:
            scene.collection.objects.link(obj)
            if obj.type == "MESH":
                mesh_objects.append(obj)

    return mesh_objects


def apply_transforms(objects: list[bpy.types.Object]):
    """Apply rotation and scale transforms to mesh objects.

    Preserves location so relative brick positions stay intact.
    This ensures collision shapes and mass estimates are correct.
    """
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


def separate_bricks(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """Separate mesh objects into individual bricks by loose parts.

    Objects that are already a single connected mesh remain unchanged.
    Multi-island meshes are split into separate objects.
    Returns the final list of individual brick objects.
    """
    all_bricks = []

    for obj in objects:
        if obj.type != "MESH":
            continue

        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Separate by loose parts (no-op for single-island meshes)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")

        # Collect all selected objects (original + new splits)
        for sel in bpy.context.selected_objects:
            if sel.type == "MESH":
                all_bricks.append(sel)

    return all_bricks


# =============================================================================
# Bounding box helpers
# =============================================================================


def model_bbox(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    """Compute combined world-space bounding box of all mesh objects.

    Returns (bb_min, bb_max).
    """
    all_min = Vector((float("inf"), float("inf"), float("inf")))
    all_max = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            all_min.x = min(all_min.x, world_co.x)
            all_min.y = min(all_min.y, world_co.y)
            all_min.z = min(all_min.z, world_co.z)
            all_max.x = max(all_max.x, world_co.x)
            all_max.y = max(all_max.y, world_co.y)
            all_max.z = max(all_max.z, world_co.z)

    return all_min, all_max


def model_center(objects: list[bpy.types.Object]) -> Vector:
    """Compute the center of the combined bounding box."""
    bb_min, bb_max = model_bbox(objects)
    return (bb_min + bb_max) / 2.0


def model_height(objects: list[bpy.types.Object]) -> float:
    """Compute the Z extent (height) of the combined bounding box [m]."""
    bb_min, bb_max = model_bbox(objects)
    return bb_max.z - bb_min.z


# =============================================================================
# Group transform
# =============================================================================


def move_bricks(
    bricks: list[bpy.types.Object],
    offset: Vector,
    rotation_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Translate and rotate all bricks as a group around their combined center.

    Args:
        bricks: Brick objects to transform.
        offset: Translation to apply before rotation.
        rotation_euler: (rx, ry, rz) in radians, applied around group center
            after translation.
    """
    if not bricks:
        return

    # Translate
    for brick in bricks:
        brick.location += offset

    # Rotate around group center
    rx, ry, rz = rotation_euler
    if abs(rx) < 1e-6 and abs(ry) < 1e-6 and abs(rz) < 1e-6:
        return

    bb_min, bb_max = model_bbox(bricks)
    center = (bb_min + bb_max) / 2.0

    rot_mat = Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")

    for brick in bricks:
        # Rotate position around center
        rel = brick.location - center
        brick.location = rot_mat @ rel + center
        # Rotate brick's own orientation
        orig_mat = brick.rotation_euler.to_matrix().to_4x4()
        brick.rotation_euler = (rot_mat @ orig_mat).to_euler()


# =============================================================================
# Rigid body setup
# =============================================================================


def estimate_brick_mass(obj: bpy.types.Object, density: float = 1050.0) -> float:
    """Estimate mass from bounding box volume and material density [kg].

    Default density is ABS plastic (1050 kg/m³).
    Returns at least 1 gram to avoid zero-mass issues.
    """
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    volume = (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs))
    return max(volume * density, 0.001)


def add_rigid_body_active(
    obj: bpy.types.Object,
    mass: float = 0.01,
    friction: float = 0.5,
    restitution: float = 0.3,
    collision_shape: str = "CONVEX_HULL",
):
    """Add an active (simulated) rigid body to a mesh object."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rb = obj.rigid_body
    rb.mass = mass
    rb.friction = friction
    rb.restitution = restitution
    rb.collision_shape = collision_shape
    obj.select_set(False)


def add_rigid_body_passive(
    obj: bpy.types.Object,
    friction: float = 0.5,
    restitution: float = 0.2,
    collision_shape: str = "MESH",
):
    """Add a passive (static) rigid body to a mesh object."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    rb = obj.rigid_body
    rb.friction = friction
    rb.restitution = restitution
    rb.collision_shape = collision_shape
    obj.select_set(False)


def setup_ground_plane(
    size: float = GROUND_SIZE,
    location: tuple = (0, 0, 0),
    friction: float = 0.5,
    restitution: float = 0.2,
) -> bpy.types.Object:
    """Create a ground plane with passive rigid body."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    ground = bpy.context.active_object
    ground.name = "Ground"
    add_rigid_body_passive(ground, friction=friction, restitution=restitution)
    return ground


def ensure_rigid_body_world(frame_count: int = 250):
    """Ensure a rigid body world exists and configure simulation parameters."""
    scene = bpy.context.scene
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()

    rbw = scene.rigidbody_world
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = frame_count
    rbw.substeps_per_frame = RBD_SUBSTEPS
    rbw.solver_iterations = RBD_SOLVER_ITERATIONS


# =============================================================================
# Material resolution
# =============================================================================


def resolve_lego_materials(material_arg: str) -> list[str]:
    """Convert a --material arg into a list of material preset names."""
    if material_arg.lower() == "all":
        return list(LEGO_MATERIAL_NAMES)
    if material_arg not in LEGO_MATERIALS:
        raise ValueError(f"Unknown material '{material_arg}'. Choose from: {LEGO_MATERIAL_NAMES}")
    return [material_arg]
