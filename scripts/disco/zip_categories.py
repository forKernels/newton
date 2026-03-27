"""
Zip Garment Categories
======================
Standalone utility (no Blender required) to zip each garment category
folder into its own archive for upload / transfer.

Reads the dataset directory and creates one .zip per category folder.
Only includes *_sim_prep.obj files and their sidecar JSONs by default,
or all files with --include-all.

Usage:
  python scripts/disco/zip_categories.py \
      --dataset-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set" \
      --output-dir "D:/_blender/_myBlender/SimulationWork/ClothDataset/_zips"

  # Include all files (not just sim_prep):
  python scripts/disco/zip_categories.py \
      --dataset-dir "..." --output-dir "..." --include-all

  # Specific categories only:
  python scripts/disco/zip_categories.py \
      --dataset-dir "..." --output-dir "..." \
      --categories dress_sleeveless_2550 tee_2300

  # Use config file for paths:
  python scripts/disco/zip_categories.py --config scripts/disco/sim_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path


def load_config(config_path: str | None) -> dict:
    """Load config JSON if provided."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def zip_category(
    cat_dir: Path,
    output_dir: Path,
    include_all: bool = False,
) -> dict:
    """Zip a single category folder.

    Returns metadata dict with file count and size.
    """
    cat_name = cat_dir.name
    zip_path = output_dir / f"{cat_name}.zip"

    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  SKIP (exists): {cat_name}.zip ({size_mb:.1f} MB)")
        return {"skipped": True, "zip_path": str(zip_path)}

    print(f"  Zipping: {cat_name} ...", end="", flush=True)
    t0 = time.time()

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(cat_dir):
            for fname in sorted(files):
                # Filter to sim_prep files unless include_all
                if not include_all:
                    if not (
                        fname.endswith("_sim_prep.obj") or fname.endswith("_sim_prep.mtl") or fname.endswith(".json")
                    ):
                        continue

                fpath = Path(root) / fname
                arcname = f"{cat_name}/{fpath.relative_to(cat_dir)}"
                zf.write(fpath, arcname)
                file_count += 1

    elapsed = time.time() - t0
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f" {file_count} files, {size_mb:.1f} MB ({elapsed:.1f}s)")

    return {
        "category": cat_name,
        "zip_path": str(zip_path),
        "file_count": file_count,
        "size_mb": round(size_mb, 1),
        "time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Zip each garment category into its own archive")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Root garment dataset directory (e.g. _Maria_Set)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for zip files (default: {dataset-dir}/_zips)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Specific categories to zip (default: all)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all files, not just *_sim_prep.obj + JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.json for default paths",
    )
    args = parser.parse_args()

    # Load config for defaults
    config = load_config(args.config)
    dataset_dir = args.dataset_dir or config.get("garment_dir")
    if not dataset_dir:
        parser.error("--dataset-dir is required (or set garment_dir in config)")

    dataset_root = Path(dataset_dir)
    if not dataset_root.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "_zips"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find category folders
    if args.categories:
        cat_dirs = []
        for cat_name in args.categories:
            matches = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith(cat_name)]
            cat_dirs.extend(matches)
    else:
        cat_dirs = sorted(
            d
            for d in dataset_root.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.endswith(".zip")
        )

    print("=" * 60)
    print("ZIP GARMENT CATEGORIES")
    print("=" * 60)
    print(f"Dataset:    {dataset_root}")
    print(f"Output:     {output_dir}")
    print(f"Categories: {len(cat_dirs)}")
    print(f"Mode:       {'all files' if args.include_all else 'sim_prep + JSON only'}")
    print()

    results = []
    total_size = 0.0

    for cat_dir in cat_dirs:
        result = zip_category(cat_dir, output_dir, args.include_all)
        results.append(result)
        if not result.get("skipped"):
            total_size += result.get("size_mb", 0)

    # Summary
    completed = sum(1 for r in results if not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))

    print(f"\n{'=' * 60}")
    print(f"DONE -- {completed} zips created, {skipped} skipped")
    print(f"  Total size: {total_size:.1f} MB")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}")

    # Write manifest
    manifest_path = output_dir / "zip_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "dataset_dir": str(dataset_root),
                "categories": results,
                "total_size_mb": round(total_size, 1),
            },
            f,
            indent=2,
        )
    print(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
