"""Quick test: grid cloth drag over props (50 frames) with physics scatter.

Discovers props from all configured sources (mujoco, kitchen, warehouse).
Tests with 2 random props on Chair_01 surface, scatter-all enabled.
Verifies .blend, .usda, .json output.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_drag_props_sim import PROP_DIRS, find_props, run_single_sim

FURNITURE = Path("D:/_blender/_myBlender/SimulationWork/seedAssets/scenes/Chair_01.blend")
OUTPUT = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_TestSims")
FRAMES = 50
GRID_SIZE = 1.0
NUM_PROPS = 2

errors = []

# Discover props from ALL configured directories
print("Searching prop directories:")
all_props = find_props(PROP_DIRS)
print(f"Total props available: {len(all_props)}")

if not all_props:
    print("ERROR: No props found. Check PROP_DIRS paths.")
    sys.exit(1)

rng = random.Random(42)
prop_paths = rng.sample(all_props, min(NUM_PROPS, len(all_props)))
print(f"\nSelected {len(prop_paths)} props for test:")
for p in prop_paths:
    if p.is_dir():
        print(f"  [multi-part] {p.parent.name}")
    elif p.name == "model.obj":
        print(f"  {p.parent.name}")
    else:
        print(f"  {p.stem}")

print(f"\n>>> TEST: GRID DRAG OVER PROPS ({FRAMES} frames, {NUM_PROPS} props, scatter_all) <<<")
result = run_single_sim(
    furniture_path=FURNITURE,
    prop_paths=prop_paths,
    preset_name="Cotton",
    output_dir=OUTPUT,
    seed=500,
    sample_idx=1,
    frame_count=FRAMES,
    grid_size=GRID_SIZE,
    grid_cuts=30,
    scatter_props=True,
    scatter_scenes=True,
)
print(f"Result: {result}\n")

if result.get("error"):
    errors.append(f"grid_drag_props: {result['error']}")
elif not result.get("skipped"):
    sim_dir = OUTPUT / "Chair_01" / "grid_drag_props" / "Cotton"
    base = f"grid_{GRID_SIZE}m_001"
    for ext in (".blend", ".usda", ".json"):
        f = sim_dir / f"{base}{ext}"
        if not f.exists():
            errors.append(f"MISSING: {f}")
        else:
            print(f"  OK: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

if errors:
    print(f"\nFAILURES ({len(errors)}):")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

print("\nDONE -- grid drag over props test complete, output files verified")
