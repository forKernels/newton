"""
Sim Prep -- Cloth Dataset (Blender)
====================================
Blender script to prepare garment meshes for Newton cloth simulation.

Does NOT touch the mesh geometry. Only:
  1. Scale cm -> m (apply)
  2. Set origin to center of volume
  3. Floor garment (bottom at Z=0)
  4. Shade smooth
  5. Solidify modifier (0.005 thickness), apply
  6. Export cleaned OBJ + cloth_properties.json

Usage (headless):
  blender --background --python scripts/disco/clean_cloth_dataset.py -- \
      --dataset-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Clean"

  # Single garment test:
  blender --background --python scripts/disco/clean_cloth_dataset.py -- \
      --dataset-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --categories dress_sleeveless_2550 --max-items 1 --fabrics default

  # In-place (writes next to original):
  blender --background --python scripts/disco/clean_cloth_dataset.py -- \
      --dataset-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --categories dress_sleeveless_2550 --max-items 1 --fabrics default

  # Dry run:
  blender --background --python scripts/disco/clean_cloth_dataset.py -- \
      --dataset-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --dry-run

  # List all available fabrics:
  blender --background --python scripts/disco/clean_cloth_dataset.py -- --list-fabrics
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import bpy

# Import shared fabric library (no bpy dependency)
from fabric_library import (
    CATEGORY_FABRICS,
    CLOTH_SCALE,
    FABRICS,
    SOLIDIFY_THICKNESS,
    get_fabric_properties,
    get_fabrics_for_category,
)
from mathutils import Vector

# =============================================================================
# Blender mesh processing
# =============================================================================


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def process_garment(obj_path: str, out_path: str) -> dict:
    """Import, prep for sim, and export a single garment.

    Steps:
      1. Import OBJ
      2. Scale cm -> m (apply)
      3. Set origin to center of volume
      4. Floor garment (bottom at Z=0)
      5. Shade smooth
      6. Solidify modifier (0.005), apply
      7. Export OBJ

    Returns stats dict.
    """
    clear_scene()

    # Import OBJ
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    verts_before = len(obj.data.vertices)
    faces_before = len(obj.data.polygons)

    # 1. Scale cm -> m
    obj.scale = (CLOTH_SCALE, CLOTH_SCALE, CLOTH_SCALE)
    bpy.ops.object.transform_apply(scale=True)

    # 2. Origin to center of volume
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")

    # 3. Floor: raise so bottom is at Z=0
    bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(v.z for v in bbox_corners)
    obj.location.z -= min_z
    bpy.ops.object.transform_apply(location=True)

    # 4. Shade smooth
    bpy.ops.object.shade_smooth()

    # 5. Solidify modifier
    bpy.ops.object.modifier_add(type="SOLIDIFY")
    obj.modifiers["Solidify"].thickness = SOLIDIFY_THICKNESS
    bpy.ops.object.modifier_apply(modifier="Solidify")

    verts_after = len(obj.data.vertices)
    faces_after = len(obj.data.polygons)

    # Measure final bbox
    bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bb_min = Vector((min(v.x for v in bbox_corners), min(v.y for v in bbox_corners), min(v.z for v in bbox_corners)))
    bb_max = Vector((max(v.x for v in bbox_corners), max(v.y for v in bbox_corners), max(v.z for v in bbox_corners)))
    extent = bb_max - bb_min

    # Export OBJ
    obj.select_set(True)
    try:
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=out_path,
            use_selection=True,
        )

    return {
        "verts_before": verts_before,
        "faces_before": faces_before,
        "verts_after": verts_after,
        "faces_after": faces_after,
        "extent_m": [round(extent.x, 4), round(extent.y, 4), round(extent.z, 4)],
        "solidify_thickness": SOLIDIFY_THICKNESS,
    }


# =============================================================================
# Dataset scanner
# =============================================================================


def find_garments(dataset_dir: str, categories: list[str] | None = None) -> list[dict]:
    """Find all *_sim.obj garment meshes in the dataset."""
    dataset = Path(dataset_dir)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    if categories:
        cat_dirs = [dataset / c for c in categories if (dataset / c).is_dir()]
    else:
        cat_dirs = sorted(d for d in dataset.iterdir() if d.is_dir() and not d.name.endswith((".zip", ".rar")))

    items = []
    for cat_dir in cat_dirs:
        category = cat_dir.name
        for garment_dir in sorted(cat_dir.iterdir()):
            if not garment_dir.is_dir():
                continue
            sim_objs = list(garment_dir.glob("*_sim.obj"))
            if sim_objs:
                spec_path = garment_dir / "specification.json"
                items.append(
                    {
                        "category": category,
                        "garment_id": garment_dir.name,
                        "obj_path": str(sim_objs[0]),
                        "garment_dir": str(garment_dir),
                        "has_spec": spec_path.exists(),
                        "spec_path": str(spec_path) if spec_path.exists() else None,
                    }
                )
    return items


# =============================================================================
# Main
# =============================================================================


def main():
    # Parse args after "--" (blender passes its own args before that)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Sim prep: scale + floor + solidify garment meshes (Blender)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Root dataset directory (e.g. _Maria_Set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for prepped meshes (omit for in-place)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Category folders to process (omit for all)",
    )
    parser.add_argument(
        "--fabrics",
        type=str,
        nargs="*",
        default=None,
        help="Fabric names to generate (omit for all compatible, 'default' for one per category)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max garments to process (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing files",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy specification.json, PNGs, etc. to each variant folder",
    )
    parser.add_argument(
        "--list-fabrics",
        action="store_true",
        help="Print all available fabrics and exit",
    )
    args = parser.parse_args(argv)

    # --list-fabrics mode
    if args.list_fabrics:
        print(f"\n{'=' * 70}")
        print(f"FABRIC LIBRARY -- {len(FABRICS)} fabrics available")
        print(f"{'=' * 70}\n")
        print(f"  {'Name':<14s} {'Weight':<12s} {'Density':>8s} {'Stiff':>7s} {'Damp':>10s} {'Frict':>6s}  Description")
        print(f"  {'-' * 14} {'-' * 12} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 6}  {'-' * 30}")
        for name, props in sorted(FABRICS.items(), key=lambda x: x[1]["density"]):
            print(
                f"  {name:<14s} {props['weight_class']:<12s} {props['density']:>7.2f}  "
                f"{props['tri_ke']:>6.0f}  {props['tri_kd']:>9.1e}  {props['friction']:>5.2f}  "
                f"{props['description']}"
            )

        print(f"\n{'=' * 70}")
        print("CATEGORY -> COMPATIBLE FABRICS")
        print(f"{'=' * 70}\n")
        for cat, fabrics in sorted(CATEGORY_FABRICS.items()):
            print(f"  {cat:<25s}  {len(fabrics):2d} fabrics: {', '.join(fabrics)}")
        print()
        return

    if not args.dataset_dir:
        parser.error("--dataset-dir is required (unless using --list-fabrics)")

    print("=" * 70)
    print("SIM PREP -- CLOTH DATASET (Blender)")
    print("  Scale cm->m | Origin center of volume | Floor | Shade smooth | Solidify")
    print("=" * 70)

    # Discover garments
    items = find_garments(args.dataset_dir, args.categories)
    print(f"Found {len(items)} garments in {args.dataset_dir}")

    if not items:
        print("Nothing to process.")
        return

    if args.max_items:
        items = items[: args.max_items]

    # Determine fabric mode
    use_default_only = args.fabrics and args.fabrics == ["default"]
    requested_fabrics = None if use_default_only else args.fabrics

    # Preview
    total_variants = 0
    for item in items:
        cat = item["category"]
        if use_default_only:
            total_variants += 1
        else:
            compatible = get_fabrics_for_category(cat)
            if requested_fabrics:
                compatible = [f for f in compatible if f in requested_fabrics]
            total_variants += max(len(compatible), 1)

    print(f"Fabric variants to generate: {total_variants} ({len(items)} garments x fabrics)")
    if args.fabrics:
        print(f"Fabric filter: {args.fabrics}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be written.\n")

    output_root = Path(args.output_dir) if args.output_dir else None
    if output_root and not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_written = 0
    category_stats = {}

    for i, item in enumerate(items):
        cat = item["category"]
        gid = item["garment_id"]
        obj_path = item["obj_path"]

        # Get fabrics for this garment
        if use_default_only:
            fabrics_to_gen = [get_fabrics_for_category(cat)[0]]
        else:
            fabrics_to_gen = get_fabrics_for_category(cat)
            if requested_fabrics:
                fabrics_to_gen = [f for f in fabrics_to_gen if f in requested_fabrics]
            if not fabrics_to_gen:
                fabrics_to_gen = [get_fabrics_for_category(cat)[0]]

        print(f"\n[{i + 1}/{len(items)}] {cat}/{gid}  ({len(fabrics_to_gen)} variants)")

        # Read original spec
        original_units = 100
        if item["has_spec"]:
            with open(item["spec_path"]) as f:
                spec = json.load(f)
            original_units = spec.get("properties", {}).get("units_in_meter", 100)

        # Process mesh once via Blender, export to first variant path
        # Then copy the same OBJ for additional fabric variants
        first_out_obj = None
        mesh_stats = None

        for fabric_name in fabrics_to_gen:
            fabric_props = get_fabric_properties(fabric_name)

            # Determine output path
            if output_root:
                variant_dir = output_root / cat / gid / fabric_name
                out_obj = str(variant_dir / f"{gid}_sim_prep.obj")
            else:
                orig = Path(obj_path)
                variant_dir = orig.parent
                out_obj = str(orig.parent / orig.name.replace("_sim.obj", f"_{fabric_name}_sim_prep.obj"))

            if not args.dry_run:
                variant_dir.mkdir(parents=True, exist_ok=True)

                if first_out_obj is None:
                    # First variant: run Blender processing
                    mesh_stats = process_garment(obj_path, out_obj)
                    first_out_obj = out_obj

                    extent = mesh_stats["extent_m"]
                    print(
                        f"  mesh: {mesh_stats['verts_before']:,}v {mesh_stats['faces_before']:,}f"
                        f" -> {mesh_stats['verts_after']:,}v {mesh_stats['faces_after']:,}f"
                        f" (solidify)"
                    )
                    print(f"  extent: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} m")
                else:
                    # Additional variants: copy the already-processed OBJ
                    if out_obj != first_out_obj:
                        shutil.copy2(first_out_obj, out_obj)

                # Write unified multi-engine fabric properties
                newton_props = fabric_props.get("newton", {})
                md_props = fabric_props.get("marvelous_designer", {})

                cloth_props = {
                    "category": cat,
                    "garment_id": gid,
                    "fabric": fabric_name,
                    "fabric_description": fabric_props["description"],
                    "weight_class": fabric_props["weight_class"],
                    "original_units_in_meter": original_units,
                    "units": "meters",
                    "solidify_thickness": SOLIDIFY_THICKNESS,
                    # Shared
                    "density": fabric_props["density"],
                    "friction": fabric_props["friction"],
                    # Blender cloth modifier
                    "blender": {
                        "tri_ke": fabric_props["tri_ke"],
                        "tri_ka": fabric_props["tri_ka"],
                        "tri_kd": fabric_props["tri_kd"],
                        "density": fabric_props["density"],
                        "friction": fabric_props["friction"],
                    },
                    # Newton (Warp) solver
                    "newton": {
                        "density": newton_props.get("density", fabric_props["density"]),
                        "tri_ke": newton_props.get("tri_ke", fabric_props["tri_ke"]),
                        "tri_ka": newton_props.get("tri_ka", fabric_props["tri_ka"]),
                        "tri_kd": newton_props.get("tri_kd", fabric_props["tri_kd"]),
                        "edge_ke": newton_props.get("edge_ke", 0.01),
                        "edge_kd": newton_props.get("edge_kd", 0.01),
                        "particle_radius": newton_props.get("particle_radius", 0.005),
                        "self_contact_radius": newton_props.get("self_contact_radius", 0.01),
                        "contact_mu": newton_props.get("contact_mu", 0.6),
                    },
                    # Marvelous Designer (placeholder — fill from MD export)
                    "marvelous_designer": {
                        "preset_name": md_props.get("preset_name"),
                        "density_gsm": md_props.get("density_gsm"),
                        "stretch_weft": md_props.get("stretch_weft"),
                        "stretch_warp": md_props.get("stretch_warp"),
                        "shear": md_props.get("shear"),
                        "bending_weft": md_props.get("bending_weft"),
                        "bending_warp": md_props.get("bending_warp"),
                        "buckling_ratio": md_props.get("buckling_ratio"),
                    },
                }
                if mesh_stats:
                    cloth_props["extent_m"] = mesh_stats["extent_m"]

                if output_root:
                    props_path = variant_dir / "cloth_properties.json"
                else:
                    props_path = variant_dir / f"cloth_properties_{fabric_name}.json"
                with open(str(props_path), "w") as f:
                    json.dump(cloth_props, f, indent=2)

                # Copy supporting assets
                if args.copy_assets and output_root:
                    src_dir = Path(item["garment_dir"])
                    for asset in src_dir.iterdir():
                        if asset.suffix in (".json", ".png", ".svg", ".txt"):
                            if "cloth_properties" not in asset.name:
                                dest = variant_dir / asset.name
                                if not dest.exists():
                                    shutil.copy2(str(asset), str(dest))

            total_written += 1
            print(
                f"    {fabric_name:<14s}  density={fabric_props['density']:.2f}  "
                f"stiffness={fabric_props['tri_ke']:.0f}  "
                f"friction={fabric_props['friction']:.2f}  "
                f"({fabric_props['weight_class']})"
            )

        # Track stats
        if cat not in category_stats:
            category_stats[cat] = {"garments": 0, "variants": 0, "fabrics": set()}
        category_stats[cat]["garments"] += 1
        category_stats[cat]["variants"] += len(fabrics_to_gen)
        category_stats[cat]["fabrics"].update(fabrics_to_gen)

    duration = time.time() - t0

    # Write manifest
    if not args.dry_run and output_root:
        manifest = {
            "source_dataset": args.dataset_dir,
            "total_garments": len(items),
            "total_variants": total_written,
            "multiplication_factor": round(total_written / max(len(items), 1), 1),
            "fabrics_used": sorted(set(f for cs in category_stats.values() for f in cs["fabrics"])),
            "categories": {
                cat: {
                    "garments": cs["garments"],
                    "variants": cs["variants"],
                    "fabrics": sorted(cs["fabrics"]),
                }
                for cat, cs in sorted(category_stats.items())
            },
            "prep_params": {
                "scale": CLOTH_SCALE,
                "origin": "ORIGIN_CENTER_OF_VOLUME",
                "floor": True,
                "shade_smooth": True,
                "solidify_thickness": SOLIDIFY_THICKNESS,
                "units": "meters",
            },
            "fabric_library_version": "1.0",
        }
        manifest_path = output_root / "manifest.json"
        with open(str(manifest_path), "w") as f:
            json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"DONE -- {total_written} variants from {len(items)} garments in {duration:.1f}s")
    print(f"{'=' * 70}")
    print(f"\n  {'Category':<35s} {'Garments':>9s} {'Fabrics':>8s} {'Variants':>9s}  Multiplier")
    print(f"  {'-' * 35} {'-' * 9} {'-' * 8} {'-' * 9}  {'-' * 10}")
    for cat, cs in sorted(category_stats.items()):
        mult = cs["variants"] / max(cs["garments"], 1)
        print(f"  {cat:<35s} {cs['garments']:>9d} {len(cs['fabrics']):>8d} {cs['variants']:>9d}  x{mult:.0f}")

    total_garments = sum(cs["garments"] for cs in category_stats.values())
    total_fabrics = len(set(f for cs in category_stats.values() for f in cs["fabrics"]))
    print(f"  {'-' * 35} {'-' * 9} {'-' * 8} {'-' * 9}  {'-' * 10}")
    print(
        f"  {'TOTAL':<35s} {total_garments:>9d} {total_fabrics:>8d} "
        f"{total_written:>9d}  x{total_written / max(total_garments, 1):.1f}"
    )

    if output_root:
        print(f"\n  Output: {output_root}")
    print()


if __name__ == "__main__":
    main()
