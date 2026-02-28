"""
Clean Cloth Dataset — Fix scale, orientation, origin, and add simulation properties
==================================================================================
Processes garment OBJ meshes from the Maria/GarmentCode dataset and outputs
cleaned versions ready for Newton simulation and Blender import.

What it does per garment:
  1. Reads the raw OBJ (centimeters, Y-up)
  2. Scales to meters (÷ 100)
  3. Rotates 180° around Y (front facing up when laid flat)
  4. Centers origin to mesh bounding box center, bottom at Z=0
  5. Removes degenerate (zero-area) faces
  6. Writes a clean OBJ with origin at center
  7. Writes a cloth_properties.json with simulation parameters per category

Usage:
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --output-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Clean" \
      --categories dress_sleeveless_2550 pants_straight_sides_1000

  # Dry run (report only, no writes):
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --dry-run

  # Clean ALL categories:
  python scripts/disco/clean_cloth_dataset.py \
      --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
      --output-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Clean"
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

# =============================================================================
# Cloth simulation properties per garment category
# These map to Newton's tri_ke, tri_ka, tri_kd, density parameters
# =============================================================================

CLOTH_PROPERTIES = {
    "dress_sleeveless": {
        "description": "Light sleeveless dress, flowy fabric",
        "density": 0.15,        # kg/m² — light cotton/chiffon
        "tri_ke": 50.0,         # stretch stiffness
        "tri_ka": 50.0,         # area preservation
        "tri_kd": 1.0e-6,       # damping
        "friction": 0.4,
        "fabric_type": "light_woven",
    },
    "dress": {
        "description": "General dress, medium weight",
        "density": 0.20,
        "tri_ke": 80.0,
        "tri_ka": 80.0,
        "tri_kd": 1.5e-6,
        "friction": 0.4,
        "fabric_type": "medium_woven",
    },
    "tee_sleeveless": {
        "description": "Sleeveless t-shirt, jersey knit",
        "density": 0.18,
        "tri_ke": 40.0,         # knit is stretchier
        "tri_ka": 40.0,
        "tri_kd": 1.0e-6,
        "friction": 0.5,
        "fabric_type": "jersey_knit",
    },
    "tee": {
        "description": "T-shirt, jersey knit",
        "density": 0.20,
        "tri_ke": 50.0,
        "tri_ka": 50.0,
        "tri_kd": 1.2e-6,
        "friction": 0.5,
        "fabric_type": "jersey_knit",
    },
    "jacket": {
        "description": "Jacket, heavier structured fabric",
        "density": 0.35,
        "tri_ke": 200.0,        # stiffer
        "tri_ka": 200.0,
        "tri_kd": 3.0e-6,
        "friction": 0.3,
        "fabric_type": "heavy_woven",
    },
    "jacket_hood": {
        "description": "Hooded jacket, heavy fabric",
        "density": 0.40,
        "tri_ke": 250.0,
        "tri_ka": 250.0,
        "tri_kd": 4.0e-6,
        "friction": 0.3,
        "fabric_type": "heavy_woven",
    },
    "pants_straight_sides": {
        "description": "Straight-leg pants, medium-weight woven",
        "density": 0.30,
        "tri_ke": 150.0,
        "tri_ka": 150.0,
        "tri_kd": 2.0e-6,
        "friction": 0.35,
        "fabric_type": "medium_woven",
    },
    "pants": {
        "description": "General pants",
        "density": 0.30,
        "tri_ke": 150.0,
        "tri_ka": 150.0,
        "tri_kd": 2.0e-6,
        "friction": 0.35,
        "fabric_type": "medium_woven",
    },
    "skirt_2_panels": {
        "description": "Simple 2-panel skirt, light fabric",
        "density": 0.18,
        "tri_ke": 60.0,
        "tri_ka": 60.0,
        "tri_kd": 1.0e-6,
        "friction": 0.4,
        "fabric_type": "light_woven",
    },
    "skirt_4_panels": {
        "description": "4-panel skirt, light-medium fabric",
        "density": 0.20,
        "tri_ke": 70.0,
        "tri_ka": 70.0,
        "tri_kd": 1.2e-6,
        "friction": 0.4,
        "fabric_type": "light_woven",
    },
    "skirt_8_panels": {
        "description": "8-panel skirt, fuller with more drape",
        "density": 0.22,
        "tri_ke": 80.0,
        "tri_ka": 80.0,
        "tri_kd": 1.5e-6,
        "friction": 0.4,
        "fabric_type": "medium_woven",
    },
    "skirt": {
        "description": "General skirt",
        "density": 0.20,
        "tri_ke": 70.0,
        "tri_ka": 70.0,
        "tri_kd": 1.2e-6,
        "friction": 0.4,
        "fabric_type": "light_woven",
    },
    "jumpsuit_sleeveless": {
        "description": "Sleeveless jumpsuit, medium weight",
        "density": 0.25,
        "tri_ke": 100.0,
        "tri_ka": 100.0,
        "tri_kd": 1.5e-6,
        "friction": 0.4,
        "fabric_type": "medium_woven",
    },
}

# Fallback for unknown categories
DEFAULT_CLOTH_PROPERTIES = {
    "description": "Unknown garment type",
    "density": 0.20,
    "tri_ke": 100.0,
    "tri_ka": 100.0,
    "tri_kd": 1.5e-6,
    "friction": 0.4,
    "fabric_type": "medium_woven",
}


def match_category_properties(category_name: str) -> dict:
    """Match a dataset category folder name to cloth properties.

    Tries exact match first, then prefix match (e.g. 'pants_straight_sides_1000'
    matches 'pants_straight_sides'), then generic prefix ('pants').
    """
    # Strip trailing number (vertex count) — e.g. "dress_sleeveless_2550" → "dress_sleeveless"
    parts = category_name.rsplit("_", 1)
    base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else category_name

    # Exact match
    if base_name in CLOTH_PROPERTIES:
        return CLOTH_PROPERTIES[base_name].copy()

    # Prefix match (longest first)
    for key in sorted(CLOTH_PROPERTIES.keys(), key=len, reverse=True):
        if base_name.startswith(key):
            return CLOTH_PROPERTIES[key].copy()

    return DEFAULT_CLOTH_PROPERTIES.copy()


# =============================================================================
# Mesh cleaning
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
                # Handle v, v/vt, v/vt/vn formats
                face_verts = []
                for p in parts:
                    idx = int(p.split("/")[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                # Triangulate quads
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                elif len(face_verts) > 4:
                    # Fan triangulation
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
    """Clean a garment mesh: scale, rotate, center, remove bad faces.

    Returns (cleaned_vertices, cleaned_faces, stats_dict).
    """
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

    # 3. Remove degenerate faces (zero-area triangles)
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
                # Check for specification.json
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
        description="Clean cloth dataset: fix scale, orientation, origin, add cloth properties"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Root dataset directory (e.g. _Maria_Set)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for cleaned meshes. If omitted, writes next to originals as *_clean.obj",
    )
    parser.add_argument(
        "--categories", type=str, nargs="*", default=None,
        help="Category folders to process (omit for all)",
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
        help="Copy specification.json, PNGs, etc. to output dir alongside cleaned OBJ",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CLOTH DATASET CLEANER")
    print("=" * 60)

    # Discover garments
    items = find_garments(args.dataset_dir, args.categories)
    print(f"Found {len(items)} garments in {args.dataset_dir}")

    if not items:
        print("Nothing to process.")
        return

    if args.max_items:
        items = items[:args.max_items]

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    output_root = Path(args.output_dir) if args.output_dir else None
    if output_root and not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_results = []
    category_counts = {}

    for i, item in enumerate(items):
        cat = item["category"]
        gid = item["garment_id"]
        obj_path = item["obj_path"]

        print(f"[{i+1}/{len(items)}] {cat}/{gid} ... ", end="", flush=True)

        # Load
        verts, faces = load_obj_raw(obj_path)

        # Clean
        clean_verts, clean_faces, stats = clean_mesh(verts, faces)

        # Match cloth properties
        cloth_props = match_category_properties(cat)
        cloth_props["category"] = cat
        cloth_props["garment_id"] = gid

        # Read units_in_meter from spec if available
        if item["has_spec"]:
            with open(item["spec_path"]) as f:
                spec = json.load(f)
            units = spec.get("properties", {}).get("units_in_meter", 100)
            cloth_props["original_units_in_meter"] = units

        stats["cloth_properties"] = cloth_props

        extent = stats.get("clean_extent_m", [0, 0, 0])
        degen = stats.get("degenerate_removed", 0)
        degen_str = f"  [{degen} degen removed]" if degen > 0 else ""
        print(
            f"OK  {stats['clean_verts']:,}v {stats['clean_faces']:,}f  "
            f"extent={extent[0]:.3f}x{extent[1]:.3f}x{extent[2]:.3f}m  "
            f"fabric={cloth_props['fabric_type']}{degen_str}"
        )

        # Write outputs
        if not args.dry_run:
            if output_root:
                # Mirror folder structure in output
                out_cat_dir = output_root / cat / gid
                out_cat_dir.mkdir(parents=True, exist_ok=True)
                out_obj = out_cat_dir / f"{gid}_clean.obj"
                out_props = out_cat_dir / "cloth_properties.json"

                # Copy supporting files
                if args.copy_assets:
                    src_dir = Path(item["garment_dir"])
                    for f in src_dir.iterdir():
                        if f.suffix in (".json", ".png", ".svg", ".txt") and f.name != "cloth_properties.json":
                            shutil.copy2(str(f), str(out_cat_dir / f.name))
            else:
                # Write next to original
                orig = Path(obj_path)
                out_obj = orig.parent / orig.name.replace("_sim.obj", "_clean.obj")
                out_props = orig.parent / "cloth_properties.json"

            comment = (
                f"Cleaned by clean_cloth_dataset.py\n"
                f"Source: {obj_path}\n"
                f"Scale: cm -> meters (0.01)\n"
                f"Origin: bbox center XY, bottom Z=0\n"
                f"Rotated: 180 deg around Y\n"
                f"Units: meters"
            )
            write_obj(str(out_obj), clean_verts, clean_faces, comment=comment)
            with open(str(out_props), "w") as f:
                json.dump(cloth_props, f, indent=2)

        all_results.append(stats)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    duration = time.time() - t0

    # Write manifest
    if not args.dry_run and output_root:
        manifest = {
            "source_dataset": args.dataset_dir,
            "cleaned_count": len(all_results),
            "categories": {
                cat: {
                    "count": count,
                    "cloth_properties": match_category_properties(cat),
                }
                for cat, count in category_counts.items()
            },
            "cleaning_params": {
                "scale": 0.01,
                "rotate_y_180": True,
                "center_origin": True,
                "remove_degenerate": True,
                "units": "meters",
            },
        }
        manifest_path = output_root / "manifest.json"
        with open(str(manifest_path), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest: {manifest_path}")

    # Summary
    total_degen = sum(r.get("degenerate_removed", 0) for r in all_results)
    print(f"\n{'='*60}")
    print(f"DONE — {len(all_results)} garments processed in {duration:.1f}s")
    print(f"{'='*60}")
    for cat, count in sorted(category_counts.items()):
        props = match_category_properties(cat)
        print(f"  {cat:35s}  {count:4d} garments  density={props['density']:.2f}  "
              f"stiffness={props['tri_ke']:.0f}  fabric={props['fabric_type']}")
    if total_degen > 0:
        print(f"\n  Degenerate faces removed: {total_degen}")
    if output_root:
        print(f"\n  Output: {output_root}")
    print()


if __name__ == "__main__":
    main()
