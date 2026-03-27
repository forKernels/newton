"""
MD Post-Processing: Convert Alembic to USDA + Blend
=====================================================
After running MD batch sims, run this script to convert all .abc files
to .usda (via Blender) and create .blend files for review.

Usage:
    uv run python scripts/disco/md_abc_to_usda.py --input-dir C:/_data/SimObj/Output/md_cloth
    uv run python scripts/disco/md_abc_to_usda.py --input-dir C:/_data/SimObj/Output/md_cloth --blender-exe blender
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def convert_abc_to_usda_blend(abc_path: Path, blender_exe: str = "blender") -> bool:
    """Convert a single .abc to .usda and .blend via Blender."""
    usda_path = abc_path.with_suffix(".usda")
    blend_path = abc_path.with_suffix(".blend")

    if usda_path.exists() and blend_path.exists():
        return True  # already done

    script = f'''
import bpy

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Import Alembic
bpy.ops.wm.alembic_import(filepath=r"{abc_path}")

# Export USDA with animation
bpy.ops.wm.usd_export(
    filepath=r"{usda_path}",
    export_animation=True,
    export_mesh_colors=False,
    selected_objects_only=False,
)

# Save as .blend for review
bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")

print("CONVERT_OK")
'''

    try:
        result = subprocess.run(
            [blender_exe, "--background", "--python-expr", script],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
        )
        return "CONVERT_OK" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MD Alembic exports to USDA + Blend")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Root directory containing .abc files from MD sims"
    )
    parser.add_argument("--blender-exe", type=str, default="blender", help="Path to Blender executable")
    args = parser.parse_args()

    root = Path(args.input_dir)
    abc_files = sorted(root.rglob("*.abc"))

    if not abc_files:
        print(f"No .abc files found in {root}")
        sys.exit(1)

    # Skip already-converted
    todo = [f for f in abc_files if not f.with_suffix(".usda").exists()]
    print(f"Found {len(abc_files)} .abc files, {len(todo)} need conversion")

    t0 = time.time()
    ok = 0
    fail = 0

    for i, abc in enumerate(todo):
        rel = abc.relative_to(root)
        print(f"  [{i + 1}/{len(todo)}] {rel} ...", end=" ", flush=True)

        if convert_abc_to_usda_blend(abc, args.blender_exe):
            ok += 1
            print("OK")
        else:
            fail += 1
            print("FAILED")

    elapsed = time.time() - t0
    print(f"\nDone: {ok} converted, {fail} failed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
