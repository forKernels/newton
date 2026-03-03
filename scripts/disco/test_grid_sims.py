"""Quick test: grid cloth simulations (throw, drag, lay) — 50 frames each.

Verifies .blend, .usda, and .json output files are created for each mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_throw_sim import run_single_sim as throw_sim
from grid_drag_sim import run_single_sim as drag_sim
from grid_lay_sim import run_single_sim as lay_sim

FURNITURE = Path("D:/_blender/_myBlender/SimulationWork/seedAssets/scenes/Chair_01.blend")
OUTPUT = Path("D:/_blender/_myBlender/SimulationWork/ClothDataset/_TestSims")
FRAMES = 50
GRID_SIZE = 1.0
GRID_CUTS = 30

errors = []

# ── Grid Throw ───────────────────────────────────────────────────────────────
print(f"\n>>> TEST: GRID THROW (swing_up, {FRAMES} frames) <<<")
result = throw_sim(
    furniture_path=FURNITURE,
    preset_name="Cotton",
    output_dir=OUTPUT,
    seed=100,
    sample_idx=1,
    frame_count=FRAMES,
    grid_size=GRID_SIZE,
    grid_cuts=GRID_CUTS,
    throw_style="swing_up",
    power=0.6,
)
print(f"Result: {result}\n")
if result.get("error"):
    errors.append(f"grid_throw: {result['error']}")
elif not result.get("skipped"):
    sim_dir = OUTPUT / "Chair_01" / "grid_throw" / "Cotton"
    base = f"grid_{GRID_SIZE}m_001"
    for ext in (".blend", ".usda", ".json"):
        f = sim_dir / f"{base}{ext}"
        if not f.exists():
            errors.append(f"MISSING: {f}")
        else:
            print(f"  OK: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ── Grid Drag ────────────────────────────────────────────────────────────────
print(f"\n>>> TEST: GRID DRAG ({FRAMES} frames) <<<")
result = drag_sim(
    furniture_path=FURNITURE,
    preset_name="Cotton",
    output_dir=OUTPUT,
    seed=200,
    sample_idx=1,
    frame_count=FRAMES,
    grid_size=GRID_SIZE,
    grid_cuts=GRID_CUTS,
)
print(f"Result: {result}\n")
if result.get("error"):
    errors.append(f"grid_drag: {result['error']}")
elif not result.get("skipped"):
    sim_dir = OUTPUT / "Chair_01" / "grid_drag" / "Cotton"
    base = f"grid_{GRID_SIZE}m_001"
    for ext in (".blend", ".usda", ".json"):
        f = sim_dir / f"{base}{ext}"
        if not f.exists():
            errors.append(f"MISSING: {f}")
        else:
            print(f"  OK: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ── Grid Lay ─────────────────────────────────────────────────────────────────
print(f"\n>>> TEST: GRID LAY ({FRAMES} frames) <<<")
result = lay_sim(
    furniture_path=FURNITURE,
    preset_name="Cotton",
    output_dir=OUTPUT,
    seed=300,
    sample_idx=1,
    frame_count=FRAMES,
    grid_size=GRID_SIZE,
    grid_cuts=GRID_CUTS,
)
print(f"Result: {result}\n")
if result.get("error"):
    errors.append(f"grid_lay: {result['error']}")
elif not result.get("skipped"):
    sim_dir = OUTPUT / "Chair_01" / "grid_lay" / "Cotton"
    base = f"grid_{GRID_SIZE}m_001"
    for ext in (".blend", ".usda", ".json"):
        f = sim_dir / f"{base}{ext}"
        if not f.exists():
            errors.append(f"MISSING: {f}")
        else:
            print(f"  OK: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# ── Summary ──────────────────────────────────────────────────────────────────
if errors:
    print(f"\nFAILURES ({len(errors)}):")
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)

print("\nDONE -- all 3 grid modes tested, output files verified")
