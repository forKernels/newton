"""Quick test: one cloth drop simulation (60 frames).

Verifies .blend and .usda output files are created.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloth_drop_sim import run_single_sim

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

# --- Drop test ---
print("\n>>> TEST: CLOTH DROP (60 frames) <<<")
drop_result = run_single_sim(
    furniture_path=FURNITURE,
    garment_info=garment_info,
    preset_name="Cotton",
    output_dir=OUTPUT,
    seed=42,
    sample_idx=1,
    frame_count=60,
)
print(f"Drop result: {drop_result}\n")

# --- Verify outputs ---
sim_dir = OUTPUT / "Chair_01" / "drop" / "Cotton"
base_name = f"{garment_info['garment_id']}_001"

blend_file = sim_dir / f"{base_name}.blend"
usda_file = sim_dir / f"{base_name}.usda"
json_file = sim_dir / f"{base_name}.json"

errors = []
for f in (blend_file, usda_file, json_file):
    if not f.exists():
        errors.append(f"MISSING: {f}")
    else:
        size_kb = f.stat().st_size / 1024
        print(f"  OK: {f.name} ({size_kb:.1f} KB)")

if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

print("\nDONE -- drop test complete, all output files verified")
