# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Preprocessing and validation script for cloth garment datasets.
#
# Scans a dataset directory for garment OBJ meshes and reports per-garment:
#   - Vertex and face count
#   - Bounding box (raw units)
#   - Detected unit scale (mm/cm/m)
#   - Y-up detection
#   - Degenerate face count
#   - Non-manifold edge count (via trimesh)
#
# Output: a JSON summary file for filtering bad meshes before batch simulation.
#
# Usage:
#   uv run python scripts/preprocess_cloth_dataset.py \
#       --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set" \
#       --categories pants_straight_sides_1000 --max-items 3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def analyze_garment(obj_path: str) -> dict:
    """Analyze a single garment OBJ mesh and return a summary dict."""
    import numpy as np  # noqa: PLC0415

    try:
        import trimesh
    except ImportError as e:
        raise ImportError("trimesh is required. Install with: uv sync --extra examples") from e

    result = {
        "obj_path": obj_path,
        "status": "ok",
        "error": None,
    }

    try:
        mesh = trimesh.load(obj_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                result["status"] = "error"
                result["error"] = "No geometry in OBJ"
                return result
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        if not isinstance(mesh, trimesh.Trimesh):
            result["status"] = "error"
            result["error"] = f"Unsupported mesh type: {type(mesh).__name__}"
            return result

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces)

        result["vertex_count"] = int(len(verts))
        result["face_count"] = int(len(faces))

        # Bounding box
        bbox_min = verts.min(axis=0).tolist()
        bbox_max = verts.max(axis=0).tolist()
        bbox_extent = (verts.max(axis=0) - verts.min(axis=0)).tolist()
        result["bbox_min"] = [round(x, 4) for x in bbox_min]
        result["bbox_max"] = [round(x, 4) for x in bbox_max]
        result["bbox_extent"] = [round(x, 4) for x in bbox_extent]

        # Unit scale detection
        max_dim = float(np.max(bbox_extent))
        if max_dim > 250.0:
            result["detected_units"] = "mm"
            result["scale_factor"] = 0.001
        elif max_dim > 3.0:
            result["detected_units"] = "cm"
            result["scale_factor"] = 0.01
        else:
            result["detected_units"] = "m"
            result["scale_factor"] = 1.0

        # Y-up detection
        extent = np.array(bbox_extent)
        y_is_tallest = float(extent[1]) > float(extent[2]) * 1.5 and int(np.argmax(extent)) == 1
        result["y_up_detected"] = bool(y_is_tallest)

        # Degenerate faces
        areas = mesh.area_faces
        degenerate_count = int(np.sum(areas < 1e-10))
        result["degenerate_faces"] = degenerate_count

        # Non-manifold edges
        if hasattr(mesh, "edges_unique") and hasattr(mesh, "edges"):
            try:
                edge_counts = {}
                for edge in mesh.edges_sorted:
                    key = tuple(edge)
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                non_manifold_count = sum(1 for c in edge_counts.values() if c > 2)
                result["non_manifold_edges"] = non_manifold_count
            except Exception:
                result["non_manifold_edges"] = None
        else:
            result["non_manifold_edges"] = None

        result["is_watertight"] = bool(mesh.is_watertight)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def find_garment_objs(dataset_dir: str, categories: list[str] | None = None) -> list[dict]:
    """Scan dataset directory for *_sim.obj garment meshes."""
    dataset = Path(dataset_dir)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if categories:
        cat_dirs = [dataset / c for c in categories]
        cat_dirs = [d for d in cat_dirs if d.is_dir()]
    else:
        cat_dirs = sorted(d for d in dataset.iterdir() if d.is_dir() and not d.name.endswith(".rar"))

    items = []
    for cat_dir in cat_dirs:
        category = cat_dir.name
        for garment_dir in sorted(cat_dir.iterdir()):
            if not garment_dir.is_dir():
                continue
            sim_objs = list(garment_dir.glob("*_sim.obj"))
            if sim_objs:
                items.append(
                    {
                        "category": category,
                        "garment_id": garment_dir.name,
                        "obj_path": str(sim_objs[0]),
                    }
                )
    return items


def main():
    parser = argparse.ArgumentParser(description="Preprocess and validate cloth garment dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root dataset directory containing category folders.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Category folder names to process. Omit to process all.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Max items to process.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON summary file path (default: <dataset-dir>/preprocess_summary.json).",
    )
    args = parser.parse_args()

    # Discover garments
    print(f"Scanning dataset: {args.dataset_dir}")
    items = find_garment_objs(args.dataset_dir, args.categories)
    print(f"Found {len(items)} garments")

    if not items:
        print("No garments found. Check --dataset-dir and --categories.")
        return

    if args.max_items is not None:
        items = items[: args.max_items]
    print(f"Analyzing {len(items)} garments...")

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        cat = item["category"]
        gid = item["garment_id"]
        print(f"  [{i + 1}/{len(items)}] {cat}/{gid} ...", end=" ", flush=True)
        analysis = analyze_garment(item["obj_path"])
        analysis["category"] = cat
        analysis["garment_id"] = gid
        results.append(analysis)

        if analysis["status"] == "ok":
            flags = []
            if analysis["y_up_detected"]:
                flags.append("Y-up")
            if analysis["degenerate_faces"] > 0:
                flags.append(f"{analysis['degenerate_faces']} degen")
            if analysis.get("non_manifold_edges", 0):
                flags.append(f"{analysis['non_manifold_edges']} non-manifold")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"OK  verts={analysis['vertex_count']:,}  faces={analysis['face_count']:,}  "
                f"units={analysis['detected_units']}  extent={analysis['bbox_extent']}{flag_str}"
            )
        else:
            print(f"ERROR: {analysis['error']}")

    duration = time.time() - t0

    # Summary statistics
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    y_up_count = sum(1 for r in results if r.get("y_up_detected", False))
    degen_count = sum(1 for r in results if r.get("degenerate_faces", 0) > 0)

    print(
        f"\nSummary: {ok_count} ok, {err_count} errors, "
        f"{y_up_count} Y-up, {degen_count} with degenerate faces ({duration:.1f}s)"
    )

    # Write output
    output_path = args.output or str(Path(args.dataset_dir) / "preprocess_summary.json")
    summary = {
        "dataset_dir": args.dataset_dir,
        "categories": args.categories,
        "total_items": len(results),
        "ok_count": ok_count,
        "error_count": err_count,
        "y_up_count": y_up_count,
        "degenerate_face_count": degen_count,
        "garments": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {output_path}")


if __name__ == "__main__":
    main()
