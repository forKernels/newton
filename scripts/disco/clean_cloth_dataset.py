"""
Clean Cloth Dataset — Fix meshes + generate fabric variants
============================================================
Processes garment OBJ meshes from the Maria/GarmentCode dataset:
  1. Cleans geometry: scale cm→m, rotate 180° Y, center origin, remove degenerates
  2. Generates fabric variants: each garment × N compatible fabrics = multiplied dataset

The same cleaned mesh is shared across fabric variants — only the
cloth_properties.json differs. This lets Newton simulate the same dress
as silk, cotton, wool, etc. with different physics.

Usage:
  # Clean + all fabric variants:
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --output-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Clean"

  # Specific fabrics only:
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --output-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Clean" \
      --fabrics silk denim cotton

  # Single "default" fabric per category (no multiplication):
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --output-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Clean" \
      --fabrics default

  # Dry run:
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --dry-run

  # List all available fabrics:
  python scripts/disco/clean_cloth_dataset.py --list-fabrics
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

# =============================================================================
# FABRIC LIBRARY — Physical properties for simulation
# =============================================================================
# Sources: textile engineering references, typical values for cloth simulation.
# density = kg/m² (areal density, not volumetric)
# tri_ke  = stretch stiffness (higher = resists stretching more)
# tri_ka  = area preservation stiffness
# tri_kd  = damping coefficient
# friction = surface friction coefficient

FABRICS = {
    # --- Lightweight ---
    "silk": {
        "description": "Silk — very light, smooth, fluid drape",
        "density": 0.08,
        "tri_ke": 30.0,
        "tri_ka": 25.0,
        "tri_kd": 0.5e-6,
        "friction": 0.20,
        "weight_class": "ultralight",
    },
    "chiffon": {
        "description": "Chiffon — sheer, floaty, delicate",
        "density": 0.06,
        "tri_ke": 20.0,
        "tri_ka": 15.0,
        "tri_kd": 0.3e-6,
        "friction": 0.25,
        "weight_class": "ultralight",
    },
    "organza": {
        "description": "Organza — crisp, sheer, holds shape",
        "density": 0.07,
        "tri_ke": 45.0,
        "tri_ka": 40.0,
        "tri_kd": 0.4e-6,
        "friction": 0.22,
        "weight_class": "ultralight",
    },
    "satin": {
        "description": "Satin — smooth, glossy, medium drape",
        "density": 0.12,
        "tri_ke": 40.0,
        "tri_ka": 35.0,
        "tri_kd": 0.6e-6,
        "friction": 0.15,
        "weight_class": "light",
    },

    # --- Light-Medium ---
    "cotton": {
        "description": "Cotton broadcloth — everyday woven, medium body",
        "density": 0.15,
        "tri_ke": 80.0,
        "tri_ka": 70.0,
        "tri_kd": 1.2e-6,
        "friction": 0.45,
        "weight_class": "light",
    },
    "linen": {
        "description": "Linen — natural fiber, crisp, wrinkles easily",
        "density": 0.17,
        "tri_ke": 100.0,
        "tri_ka": 90.0,
        "tri_kd": 1.5e-6,
        "friction": 0.40,
        "weight_class": "light",
    },
    "jersey": {
        "description": "Jersey knit — stretchy, soft, t-shirt fabric",
        "density": 0.18,
        "tri_ke": 35.0,
        "tri_ka": 30.0,
        "tri_kd": 1.0e-6,
        "friction": 0.50,
        "weight_class": "light",
    },
    "polyester": {
        "description": "Polyester — wrinkle-resistant, slightly slippery",
        "density": 0.14,
        "tri_ke": 60.0,
        "tri_ka": 55.0,
        "tri_kd": 0.8e-6,
        "friction": 0.30,
        "weight_class": "light",
    },

    # --- Medium ---
    "cotton_twill": {
        "description": "Cotton twill — chinos, structured but not stiff",
        "density": 0.25,
        "tri_ke": 120.0,
        "tri_ka": 110.0,
        "tri_kd": 2.0e-6,
        "friction": 0.40,
        "weight_class": "medium",
    },
    "wool": {
        "description": "Wool — warm, medium drape, textured surface",
        "density": 0.28,
        "tri_ke": 100.0,
        "tri_ka": 90.0,
        "tri_kd": 2.5e-6,
        "friction": 0.55,
        "weight_class": "medium",
    },
    "flannel": {
        "description": "Flannel — soft, brushed, warm",
        "density": 0.24,
        "tri_ke": 90.0,
        "tri_ka": 80.0,
        "tri_kd": 2.0e-6,
        "friction": 0.60,
        "weight_class": "medium",
    },
    "velvet": {
        "description": "Velvet — plush, heavy drape, grippy surface",
        "density": 0.30,
        "tri_ke": 85.0,
        "tri_ka": 75.0,
        "tri_kd": 2.5e-6,
        "friction": 0.65,
        "weight_class": "medium",
    },
    "corduroy": {
        "description": "Corduroy — ribbed texture, medium-heavy",
        "density": 0.32,
        "tri_ke": 130.0,
        "tri_ka": 120.0,
        "tri_kd": 2.5e-6,
        "friction": 0.55,
        "weight_class": "medium",
    },

    # --- Heavy ---
    "denim": {
        "description": "Denim — heavy, stiff, jeans fabric",
        "density": 0.40,
        "tri_ke": 250.0,
        "tri_ka": 230.0,
        "tri_kd": 4.0e-6,
        "friction": 0.45,
        "weight_class": "heavy",
    },
    "canvas": {
        "description": "Canvas — heavy-duty, very stiff, work wear",
        "density": 0.45,
        "tri_ke": 300.0,
        "tri_ka": 280.0,
        "tri_kd": 5.0e-6,
        "friction": 0.50,
        "weight_class": "heavy",
    },
    "leather": {
        "description": "Leather — heavy, stiff, minimal stretch",
        "density": 0.60,
        "tri_ke": 400.0,
        "tri_ka": 380.0,
        "tri_kd": 6.0e-6,
        "friction": 0.35,
        "weight_class": "heavy",
    },
    "wool_coat": {
        "description": "Wool coating — thick, structured, overcoats",
        "density": 0.50,
        "tri_ke": 280.0,
        "tri_ka": 260.0,
        "tri_kd": 5.0e-6,
        "friction": 0.50,
        "weight_class": "heavy",
    },
    "fleece": {
        "description": "Fleece — soft, thick, stretchy, high friction",
        "density": 0.35,
        "tri_ke": 60.0,
        "tri_ka": 50.0,
        "tri_kd": 3.0e-6,
        "friction": 0.70,
        "weight_class": "heavy",
    },
    "neoprene": {
        "description": "Neoprene — thick, rubbery, wetsuit material",
        "density": 0.55,
        "tri_ke": 150.0,
        "tri_ka": 140.0,
        "tri_kd": 5.0e-6,
        "friction": 0.60,
        "weight_class": "heavy",
    },
}


# =============================================================================
# GARMENT → COMPATIBLE FABRICS mapping
# =============================================================================
# Each garment category maps to a list of fabrics that make physical sense.
# Running with all compatible fabrics multiplies the dataset.

CATEGORY_FABRICS = {
    "dress_sleeveless": [
        "silk", "chiffon", "satin", "cotton", "linen", "jersey", "polyester",
    ],
    "dress": [
        "silk", "chiffon", "satin", "cotton", "linen", "jersey", "polyester",
        "wool", "velvet",
    ],
    "tee_sleeveless": [
        "jersey", "cotton", "polyester", "silk",
    ],
    "tee": [
        "jersey", "cotton", "polyester", "flannel",
    ],
    "jacket": [
        "denim", "canvas", "leather", "wool_coat", "cotton_twill", "corduroy",
        "neoprene",
    ],
    "jacket_hood": [
        "fleece", "denim", "canvas", "neoprene", "wool_coat",
    ],
    "pants_straight_sides": [
        "denim", "cotton_twill", "wool", "corduroy", "linen", "polyester",
        "leather", "flannel",
    ],
    "pants": [
        "denim", "cotton_twill", "wool", "corduroy", "linen", "polyester",
        "leather",
    ],
    "skirt_2_panels": [
        "silk", "cotton", "linen", "polyester", "wool", "satin", "velvet",
    ],
    "skirt_4_panels": [
        "silk", "cotton", "linen", "polyester", "wool", "satin", "velvet",
        "chiffon",
    ],
    "skirt_8_panels": [
        "silk", "cotton", "linen", "polyester", "wool", "chiffon", "organza",
    ],
    "skirt": [
        "silk", "cotton", "linen", "polyester", "wool", "satin",
    ],
    "jumpsuit_sleeveless": [
        "jersey", "cotton", "linen", "polyester", "denim",
    ],
}

# Fallback: if category not found, offer a broad set
DEFAULT_FABRICS = ["cotton", "polyester", "jersey", "wool", "denim"]


def get_fabrics_for_category(category_name: str) -> list[str]:
    """Get compatible fabric names for a garment category."""
    parts = category_name.rsplit("_", 1)
    base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else category_name

    if base_name in CATEGORY_FABRICS:
        return CATEGORY_FABRICS[base_name]

    for key in sorted(CATEGORY_FABRICS.keys(), key=len, reverse=True):
        if base_name.startswith(key):
            return CATEGORY_FABRICS[key]

    return DEFAULT_FABRICS


def get_fabric_properties(fabric_name: str) -> dict:
    """Get simulation properties for a named fabric."""
    if fabric_name not in FABRICS:
        raise ValueError(f"Unknown fabric: {fabric_name}. Available: {list(FABRICS.keys())}")
    return FABRICS[fabric_name].copy()


# =============================================================================
# Mesh cleaning (unchanged geometry operations)
# =============================================================================

def load_obj_raw(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and faces without any library dependency."""
    vertices = []
    faces = []
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face_verts = []
                for p in parts:
                    idx = int(p.split("/")[0]) - 1
                    face_verts.append(idx)
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                elif len(face_verts) > 4:
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def write_obj(filepath: str, vertices: np.ndarray, faces: np.ndarray, comment: str = ""):
    """Write a clean OBJ file."""
    with open(filepath, "w") as f:
        if comment:
            for line in comment.split("\n"):
                f.write(f"# {line}\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    scale: float = 0.01,
    rotate_y_180: bool = True,
    center_origin: bool = True,
    remove_degenerate: bool = True,
    degenerate_threshold: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Clean a garment mesh: scale, rotate, center, remove bad faces."""
    stats = {
        "original_verts": len(vertices),
        "original_faces": len(faces),
    }

    verts = vertices.copy()

    # 1. Scale (cm → meters)
    verts *= scale
    stats["scale_applied"] = scale

    # 2. Rotate 180° around Y axis: (x, y, z) → (-x, y, -z)
    if rotate_y_180:
        verts[:, 0] *= -1
        verts[:, 2] *= -1
        stats["rotated_y_180"] = True

    # 3. Remove degenerate faces
    good_faces = faces
    if remove_degenerate and len(faces) > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        mask = areas > degenerate_threshold
        degenerate_count = int(np.sum(~mask))
        good_faces = faces[mask]
        stats["degenerate_removed"] = degenerate_count
    else:
        stats["degenerate_removed"] = 0

    # 4. Center origin: bbox center XY, bottom at Z=0
    if center_origin and len(verts) > 0:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        center_x = (bbox_min[0] + bbox_max[0]) / 2.0
        center_y = (bbox_min[1] + bbox_max[1]) / 2.0
        bottom_z = bbox_min[2]
        verts[:, 0] -= center_x
        verts[:, 1] -= center_y
        verts[:, 2] -= bottom_z
        stats["origin_centered"] = True

    # Final bbox
    if len(verts) > 0:
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        extent = bbox_max - bbox_min
        stats["clean_bbox_min"] = bbox_min.tolist()
        stats["clean_bbox_max"] = bbox_max.tolist()
        stats["clean_extent_m"] = [round(x, 4) for x in extent.tolist()]

    stats["clean_verts"] = len(verts)
    stats["clean_faces"] = len(good_faces)

    return verts, good_faces, stats


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
        cat_dirs = sorted(
            d for d in dataset.iterdir()
            if d.is_dir() and not d.name.endswith((".zip", ".rar"))
        )

    items = []
    for cat_dir in cat_dirs:
        category = cat_dir.name
        for garment_dir in sorted(cat_dir.iterdir()):
            if not garment_dir.is_dir():
                continue
            sim_objs = list(garment_dir.glob("*_sim.obj"))
            if sim_objs:
                spec_path = garment_dir / "specification.json"
                items.append({
                    "category": category,
                    "garment_id": garment_dir.name,
                    "obj_path": str(sim_objs[0]),
                    "garment_dir": str(garment_dir),
                    "has_spec": spec_path.exists(),
                    "spec_path": str(spec_path) if spec_path.exists() else None,
                })
    return items


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean cloth dataset + generate fabric variants for dataset multiplication"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None,
        help="Root dataset directory (e.g. _Maria_Set)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for cleaned meshes + fabric variants",
    )
    parser.add_argument(
        "--categories", type=str, nargs="*", default=None,
        help="Category folders to process (omit for all)",
    )
    parser.add_argument(
        "--fabrics", type=str, nargs="*", default=None,
        help="Fabric names to generate (omit for all compatible per category, "
             "'default' for one per category)",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Max garments to process (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be done without writing files",
    )
    parser.add_argument(
        "--copy-assets", action="store_true",
        help="Copy specification.json, PNGs, etc. to each variant folder",
    )
    parser.add_argument(
        "--list-fabrics", action="store_true",
        help="Print all available fabrics and exit",
    )
    args = parser.parse_args()

    # --list-fabrics mode
    if args.list_fabrics:
        print(f"\n{'='*70}")
        print(f"FABRIC LIBRARY — {len(FABRICS)} fabrics available")
        print(f"{'='*70}\n")
        print(f"  {'Name':<14s} {'Weight':<12s} {'Density':>8s} {'Stiff':>7s} "
              f"{'Damp':>10s} {'Frict':>6s}  Description")
        print(f"  {'─'*14} {'─'*12} {'─'*8} {'─'*7} {'─'*10} {'─'*6}  {'─'*30}")
        for name, props in sorted(FABRICS.items(), key=lambda x: x[1]["density"]):
            print(f"  {name:<14s} {props['weight_class']:<12s} {props['density']:>7.2f}  "
                  f"{props['tri_ke']:>6.0f}  {props['tri_kd']:>9.1e}  {props['friction']:>5.2f}  "
                  f"{props['description']}")

        print(f"\n{'='*70}")
        print(f"CATEGORY → COMPATIBLE FABRICS")
        print(f"{'='*70}\n")
        for cat, fabrics in sorted(CATEGORY_FABRICS.items()):
            print(f"  {cat:<25s}  {len(fabrics):2d} fabrics: {', '.join(fabrics)}")
        print()
        return

    if not args.dataset_dir:
        parser.error("--dataset-dir is required (unless using --list-fabrics)")

    print("=" * 70)
    print("CLOTH DATASET CLEANER + FABRIC VARIANT GENERATOR")
    print("=" * 70)

    # Discover garments
    items = find_garments(args.dataset_dir, args.categories)
    print(f"Found {len(items)} garments in {args.dataset_dir}")

    if not items:
        print("Nothing to process.")
        return

    if args.max_items:
        items = items[:args.max_items]

    # Determine fabric mode
    use_default_only = args.fabrics and args.fabrics == ["default"]
    requested_fabrics = None if use_default_only else args.fabrics

    # Preview multiplication
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

    print(f"Fabric variants to generate: {total_variants} "
          f"({len(items)} garments × fabrics)")
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

        print(f"\n[{i+1}/{len(items)}] {cat}/{gid}  ({len(fabrics_to_gen)} variants)")

        # Load and clean mesh ONCE
        verts, faces = load_obj_raw(obj_path)
        clean_verts, clean_faces, mesh_stats = clean_mesh(verts, faces)

        extent = mesh_stats.get("clean_extent_m", [0, 0, 0])
        degen = mesh_stats.get("degenerate_removed", 0)
        degen_str = f"  [{degen} degen removed]" if degen > 0 else ""
        print(f"  mesh: {mesh_stats['clean_verts']:,}v {mesh_stats['clean_faces']:,}f  "
              f"extent={extent[0]:.3f}x{extent[1]:.3f}x{extent[2]:.3f}m{degen_str}")

        # Read original spec
        original_units = 100
        if item["has_spec"]:
            with open(item["spec_path"]) as f:
                spec = json.load(f)
            original_units = spec.get("properties", {}).get("units_in_meter", 100)

        # Generate each fabric variant
        for fabric_name in fabrics_to_gen:
            fabric_props = get_fabric_properties(fabric_name)

            cloth_props = {
                "category": cat,
                "garment_id": gid,
                "fabric": fabric_name,
                "fabric_description": fabric_props["description"],
                "weight_class": fabric_props["weight_class"],
                "density": fabric_props["density"],
                "tri_ke": fabric_props["tri_ke"],
                "tri_ka": fabric_props["tri_ka"],
                "tri_kd": fabric_props["tri_kd"],
                "friction": fabric_props["friction"],
                "original_units_in_meter": original_units,
                "units": "meters",
                "clean_extent_m": extent,
            }

            if not args.dry_run and output_root:
                # Output: category/garment_id/fabric_name/
                variant_dir = output_root / cat / gid / fabric_name
                variant_dir.mkdir(parents=True, exist_ok=True)

                # Write cleaned OBJ (same mesh for all variants)
                out_obj = variant_dir / f"{gid}_clean.obj"
                if not out_obj.exists():
                    comment = (
                        f"Cleaned by clean_cloth_dataset.py\n"
                        f"Source: {obj_path}\n"
                        f"Scale: cm -> meters (0.01)\n"
                        f"Origin: bbox center XY, bottom Z=0\n"
                        f"Rotated: 180 deg around Y\n"
                        f"Units: meters\n"
                        f"Fabric variant: {fabric_name}"
                    )
                    write_obj(str(out_obj), clean_verts, clean_faces, comment=comment)

                # Write fabric-specific properties
                out_props = variant_dir / "cloth_properties.json"
                with open(str(out_props), "w") as f:
                    json.dump(cloth_props, f, indent=2)

                # Copy supporting assets
                if args.copy_assets:
                    src_dir = Path(item["garment_dir"])
                    for asset in src_dir.iterdir():
                        if asset.suffix in (".json", ".png", ".svg", ".txt"):
                            if asset.name != "cloth_properties.json":
                                dest = variant_dir / asset.name
                                if not dest.exists():
                                    shutil.copy2(str(asset), str(dest))

            elif not args.dry_run and not output_root:
                # In-place mode: write next to original
                orig = Path(obj_path)
                out_obj = orig.parent / orig.name.replace("_sim.obj", f"_{fabric_name}_clean.obj")
                out_props = orig.parent / f"cloth_properties_{fabric_name}.json"

                comment = (
                    f"Cleaned by clean_cloth_dataset.py\n"
                    f"Fabric: {fabric_name}\n"
                    f"Units: meters"
                )
                write_obj(str(out_obj), clean_verts, clean_faces, comment=comment)
                with open(str(out_props), "w") as f:
                    json.dump(cloth_props, f, indent=2)

            total_written += 1
            print(f"    {fabric_name:<14s}  density={fabric_props['density']:.2f}  "
                  f"stiffness={fabric_props['tri_ke']:.0f}  "
                  f"friction={fabric_props['friction']:.2f}  "
                  f"({fabric_props['weight_class']})")

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
            "fabrics_used": sorted(set(
                f for cs in category_stats.values() for f in cs["fabrics"]
            )),
            "categories": {
                cat: {
                    "garments": cs["garments"],
                    "variants": cs["variants"],
                    "fabrics": sorted(cs["fabrics"]),
                }
                for cat, cs in sorted(category_stats.items())
            },
            "cleaning_params": {
                "scale": 0.01,
                "rotate_y_180": True,
                "center_origin": True,
                "remove_degenerate": True,
                "units": "meters",
            },
            "fabric_library_version": "1.0",
        }
        manifest_path = output_root / "manifest.json"
        with open(str(manifest_path), "w") as f:
            json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"DONE — {total_written} variants from {len(items)} garments in {duration:.1f}s")
    print(f"{'='*70}")
    print(f"\n  {'Category':<35s} {'Garments':>9s} {'Fabrics':>8s} {'Variants':>9s}  Multiplier")
    print(f"  {'─'*35} {'─'*9} {'─'*8} {'─'*9}  {'─'*10}")
    for cat, cs in sorted(category_stats.items()):
        mult = cs["variants"] / max(cs["garments"], 1)
        print(f"  {cat:<35s} {cs['garments']:>9d} {len(cs['fabrics']):>8d} "
              f"{cs['variants']:>9d}  ×{mult:.0f}")

    total_garments = sum(cs["garments"] for cs in category_stats.values())
    total_fabrics = len(set(f for cs in category_stats.values() for f in cs["fabrics"]))
    print(f"  {'─'*35} {'─'*9} {'─'*8} {'─'*9}  {'─'*10}")
    print(f"  {'TOTAL':<35s} {total_garments:>9d} {total_fabrics:>8d} "
          f"{total_written:>9d}  ×{total_written/max(total_garments,1):.1f}")

    if output_root:
        print(f"\n  Output: {output_root}")
    print()


if __name__ == "__main__":
    main()
