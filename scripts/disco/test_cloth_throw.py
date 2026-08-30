"""Quick test: cloth throw simulation with each of the 4 throw styles (100 frames each).

Verifies .blend and .usda output files are created for every style.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloth_throw_sim import THROW_STYLES, run_single_sim

FURNITURE = Path("D:/_blender/_myBlender/SimulationWork/seedAssets/scenes/Chair_01.blend")
GARMENT_DIR = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set")
OUTPUT = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_TestSims")

garment_subdir = GARMENT_DIR / "dress_sleeveless_2550" / "dress_sleeveless_000YCTJ9HS"
obj_file = next(garment_subdir.glob("*_sim_prep.obj"))

garment_info = {
    "category": "dress_sleeveless_2550",
    "garment_id": garment_subdir.name,
    "obj_path": str(obj_file),
}

errors = []

for i, style in enumerate(THROW_STYLES, start=1):
    print(f"\n>>> TEST: CLOTH THROW [{style}] (100 frames) <<<")
    result = run_single_sim(
        furniture_path=FURNITURE,
        garment_info=garment_info,
        preset_name="Cotton",
        output_dir=OUTPUT,
        seed=42 + i,
        sample_idx=i,
        frame_count=100,
        throw_style=style,
        power=0.6,
    )
    print(f"Result: {result}\n")

    if result.get("error"):
        errors.append(f"{style}: error={result['error']}")
        continue

    if result.get("skipped"):
        print("  (skipped -- output already exists)")
        continue

    # Verify outputs
    sim_dir = OUTPUT / "Chair_01" / "throw" / "Cotton"
    base_name = f"{garment_info['garment_id']}_{i:03d}"
    for ext in (".blend", ".usda", ".json"):
        f = sim_dir / f"{base_name}{ext}"
        if not f.exists():
            errors.append(f"MISSING: {f}")
        else:
            size_kb = f.stat().st_size / 1024
            print(f"  OK: {f.name} ({size_kb:.1f} KB)")

if errors:
    print(f"\nFAILURES ({len(errors)}):")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

print(f"\nDONE -- all {len(THROW_STYLES)} throw styles tested, output files verified")
