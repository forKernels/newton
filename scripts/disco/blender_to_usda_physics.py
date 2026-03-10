"""
Blender to USDA with Physics Data
==================================
Batch processes .blend files and exports USDA with embedded UsdPhysics schema.
Auto-estimates physics properties (mass, friction, collision) from mesh geometry.

Usage:
    # Process all .blend files in a folder
    blender --background --python scripts/disco/blender_to_usda_physics.py -- --input-dir "path/to/blend_files" --output-dir "path/to/usda_output"

    # Process a single .blend file
    blender --background --python scripts/disco/blender_to_usda_physics.py -- --input "path/to/model.blend" --output "output.usda"

    # Dry run to see what would be processed
    blender --background --python scripts/disco/blender_to_usda_physics.py -- --input-dir "path/to/blend_files" --dry-run

    # With custom default density (kg/m³)
    blender --background --python scripts/disco/blender_to_usda_physics.py -- --input-dir "path/to/blend_files" --output-dir "output" --density 500
"""

import argparse
import json
import sys
from pathlib import Path

# Blender imports
try:
    import bpy
    import bmesh
    from mathutils import Vector
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    print("WARNING: Not running inside Blender. Run with: blender --background --python <script>")


# =============================================================================
# Material-based Physics Estimation
# =============================================================================

MATERIAL_PHYSICS = {
    # Metals
    "metal": {"density": 7800, "friction": 0.4, "restitution": 0.3},
    "steel": {"density": 7850, "friction": 0.4, "restitution": 0.3},
    "iron": {"density": 7870, "friction": 0.4, "restitution": 0.25},
    "aluminum": {"density": 2700, "friction": 0.35, "restitution": 0.35},
    "copper": {"density": 8960, "friction": 0.4, "restitution": 0.3},
    "brass": {"density": 8500, "friction": 0.35, "restitution": 0.3},
    # Wood
    "wood": {"density": 600, "friction": 0.5, "restitution": 0.2},
    "oak": {"density": 750, "friction": 0.5, "restitution": 0.2},
    "pine": {"density": 500, "friction": 0.5, "restitution": 0.2},
    "plywood": {"density": 550, "friction": 0.45, "restitution": 0.15},
    # Plastics
    "plastic": {"density": 1050, "friction": 0.35, "restitution": 0.4},
    "rubber": {"density": 1100, "friction": 0.9, "restitution": 0.7},
    "silicone": {"density": 1100, "friction": 0.8, "restitution": 0.6},
    "pvc": {"density": 1400, "friction": 0.4, "restitution": 0.3},
    "nylon": {"density": 1150, "friction": 0.3, "restitution": 0.35},
    # Glass/Ceramic
    "glass": {"density": 2500, "friction": 0.2, "restitution": 0.5},
    "ceramic": {"density": 2400, "friction": 0.4, "restitution": 0.2},
    "porcelain": {"density": 2400, "friction": 0.35, "restitution": 0.2},
    # Stone/Concrete
    "stone": {"density": 2600, "friction": 0.6, "restitution": 0.15},
    "concrete": {"density": 2400, "friction": 0.7, "restitution": 0.1},
    "marble": {"density": 2700, "friction": 0.4, "restitution": 0.2},
    "granite": {"density": 2750, "friction": 0.5, "restitution": 0.15},
    # Fabric/Soft
    "fabric": {"density": 300, "friction": 0.6, "restitution": 0.1},
    "cloth": {"density": 300, "friction": 0.6, "restitution": 0.1},
    "leather": {"density": 900, "friction": 0.5, "restitution": 0.15},
    "foam": {"density": 50, "friction": 0.7, "restitution": 0.3},
    # Paper/Cardboard
    "paper": {"density": 700, "friction": 0.5, "restitution": 0.1},
    "cardboard": {"density": 200, "friction": 0.5, "restitution": 0.1},
    # Default
    "default": {"density": 1000, "friction": 0.5, "restitution": 0.2},
}


def guess_physics_from_name(name: str) -> dict:
    """Guess physics properties from object/material name."""
    name_lower = name.lower()
    for mat_key, props in MATERIAL_PHYSICS.items():
        if mat_key in name_lower:
            return props.copy()
    return MATERIAL_PHYSICS["default"].copy()


# =============================================================================
# Blender Geometry Analysis
# =============================================================================

def get_mesh_volume(obj) -> float:
    """Calculate mesh volume in cubic meters."""
    if not HAS_BLENDER or obj.type != 'MESH':
        return 0.001  # default 1 liter

    # Apply transforms to get world-space volume
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    volume = bm.calc_volume()
    bm.free()

    return abs(volume)


def get_bounding_box(obj) -> tuple:
    """Get world-space bounding box dimensions and center."""
    if not HAS_BLENDER or obj.type != 'MESH':
        return ([0.1, 0.1, 0.1], [0, 0, 0])

    # Get world-space bounding box corners
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    xs = [c.x for c in bbox_corners]
    ys = [c.y for c in bbox_corners]
    zs = [c.z for c in bbox_corners]

    dims = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
    center = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2]

    return dims, center


def choose_collision_type(dims: list) -> str:
    """Choose best collision primitive based on dimensions."""
    x, y, z = dims

    # Check if roughly spherical
    avg = (x + y + z) / 3
    if all(abs(d - avg) / avg < 0.2 for d in dims):
        return "sphere"

    # Check if roughly cylindrical (one axis much longer)
    sorted_dims = sorted(dims)
    if sorted_dims[2] > sorted_dims[1] * 2:
        return "capsule"

    # Default to box
    return "box"


def estimate_physics(obj, default_density: float = 1000) -> dict:
    """Estimate physics properties from mesh geometry and material names."""
    if not HAS_BLENDER:
        return {
            "mass_kg": 1.0,
            "friction_static": 0.5,
            "friction_dynamic": 0.4,
            "restitution": 0.2,
            "dimensions_m": [0.1, 0.1, 0.1],
            "collision_primitive": {"type": "box", "half_extents": [0.05, 0.05, 0.05], "center": [0, 0, 0]},
        }

    # Get geometry info
    volume = get_mesh_volume(obj)
    dims, center = get_bounding_box(obj)

    # Guess physics from names
    physics = guess_physics_from_name(obj.name)

    # Also check material names
    if obj.data.materials:
        for mat in obj.data.materials:
            if mat:
                mat_physics = guess_physics_from_name(mat.name)
                if mat_physics != MATERIAL_PHYSICS["default"]:
                    physics = mat_physics
                    break

    # Calculate mass from volume and density
    density = physics.get("density", default_density)
    mass = volume * density

    # Clamp to reasonable range
    mass = max(0.01, min(mass, 1000))

    # Choose collision type
    collision_type = choose_collision_type(dims)

    if collision_type == "sphere":
        radius = max(dims) / 2
        collision = {"type": "sphere", "radius": radius, "center": center}
    elif collision_type == "capsule":
        # Use box for now (USD Physics doesn't have capsule as primitive)
        collision = {"type": "box", "half_extents": [d/2 for d in dims], "center": center}
    else:
        collision = {"type": "box", "half_extents": [d/2 for d in dims], "center": center}

    return {
        "mass_kg": round(mass, 4),
        "friction_static": physics["friction"],
        "friction_dynamic": physics["friction"] * 0.8,
        "restitution": physics["restitution"],
        "dimensions_m": dims,
        "collision_primitive": collision,
        "estimated_material": next((k for k, v in MATERIAL_PHYSICS.items() if v == physics), "default"),
    }


# =============================================================================
# Blender Scene Operations
# =============================================================================

def clear_scene():
    """Remove all objects from the scene."""
    if not HAS_BLENDER:
        return
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def load_blend_file(filepath: Path):
    """Load a .blend file, keeping existing transforms."""
    if not HAS_BLENDER:
        return []

    # Open the file directly
    bpy.ops.wm.open_mainfile(filepath=str(filepath))

    # Return all mesh objects
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


def export_usda_with_physics(
    output_path: Path,
    objects_physics: list,
    asset_name: str,
):
    """Export scene to USDA with embedded UsdPhysics data."""
    if not HAS_BLENDER:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First export using Blender's USD exporter (geometry + materials)
    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        export_animation=False,
        export_hair=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        use_instancing=True,
        evaluation_mode='RENDER',
    )

    # Now read the USDA and inject physics data
    inject_physics_into_usda(output_path, objects_physics)


def inject_physics_into_usda(usda_path: Path, objects_physics: list):
    """Inject UsdPhysics schema into existing USDA file."""
    with open(usda_path, 'r') as f:
        content = f.read()

    # Build physics material definitions
    physics_materials = []
    for obj_name, physics in objects_physics:
        safe_name = obj_name.replace("-", "_").replace(" ", "_").replace(".", "_")
        mat_def = f'''
def Material "{safe_name}_PhysMat" (
    prepend apiSchemas = ["PhysicsMaterialAPI"]
)
{{
    float physics:staticFriction = {physics['friction_static']}
    float physics:dynamicFriction = {physics['friction_dynamic']}
    float physics:restitution = {physics['restitution']}
}}'''
        physics_materials.append(mat_def)

    # Add PhysicsMaterials scope before final closing
    materials_scope = f'''
def Scope "PhysicsMaterials"
{{
    {"".join(physics_materials)}
}}
'''

    # Find where to inject (before the last closing brace if it's a proper USD)
    # For simplicity, append to end
    content = content.rstrip()
    if content.endswith('}'):
        content = content[:-1] + materials_scope + '\n}'
    else:
        content += '\n' + materials_scope

    # Write back
    with open(usda_path, 'w') as f:
        f.write(content)

    # Also write a companion JSON with physics data for Newton
    json_path = usda_path.with_suffix('.physics.json')
    physics_data = {
        "objects": [
            {"name": name, "physics": physics}
            for name, physics in objects_physics
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(physics_data, f, indent=2)


# =============================================================================
# Main Processing
# =============================================================================

def process_blend_file(
    input_path: Path,
    output_path: Path,
    default_density: float = 1000,
):
    """Process a single .blend file to USDA with physics."""
    print(f"Processing: {input_path.name}")

    if not HAS_BLENDER:
        print("  ERROR: Must run inside Blender")
        return None

    # Load the blend file
    mesh_objects = load_blend_file(input_path)

    if not mesh_objects:
        print("  WARNING: No mesh objects found")
        return None

    # Estimate physics for each object
    objects_physics = []
    for obj in mesh_objects:
        physics = estimate_physics(obj, default_density)
        objects_physics.append((obj.name, physics))
        print(f"  {obj.name}: mass={physics['mass_kg']:.3f}kg, "
              f"material={physics.get('estimated_material', 'default')}, "
              f"collision={physics['collision_primitive']['type']}")

    # Export to USDA with physics
    asset_name = input_path.stem.replace("-", "_").replace(" ", "_")
    export_usda_with_physics(output_path, objects_physics, asset_name)

    print(f"  -> {output_path}")
    print(f"  -> {output_path.with_suffix('.physics.json')}")

    return output_path


def process_batch(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    default_density: float = 1000,
):
    """Process all .blend files in a directory."""
    blend_files = list(input_dir.glob("*.blend"))

    # Also check subdirectories
    blend_files.extend(input_dir.glob("**/*.blend"))
    blend_files = sorted(set(blend_files))

    print(f"Found {len(blend_files)} .blend files")

    if dry_run:
        print("\nDry run - would process:")
        for f in blend_files[:20]:
            print(f"  {f.name}")
        if len(blend_files) > 20:
            print(f"  ... and {len(blend_files) - 20} more")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for i, blend_file in enumerate(blend_files):
        try:
            # Create output path preserving relative structure
            rel_path = blend_file.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.usda')

            process_blend_file(blend_file, out_path, default_density)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(blend_file.name)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(blend_files)}")

    print(f"\nDone! Success: {success}, Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Convert .blend files to USDA with physics")
    parser.add_argument("--input", type=str, help="Single .blend file to process")
    parser.add_argument("--output", type=str, help="Output USDA path")
    parser.add_argument("--input-dir", type=str, help="Directory of .blend files")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--density", type=float, default=1000,
                        help="Default density kg/m³ if material unknown (default: 1000)")
    args = parser.parse_args(argv)

    if args.input:
        if not args.output:
            args.output = str(Path(args.input).with_suffix('.usda'))
        process_blend_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            default_density=args.density,
        )
    elif args.input_dir:
        if not args.output_dir:
            parser.error("--input-dir requires --output-dir")
        process_batch(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
            default_density=args.density,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
