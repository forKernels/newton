"""
Newton Cloth Simulation Test Suite
=====================================
Exercises all 6 Newton cloth sim types with local assets:
  1. cloth_drop    -- garment drops onto table + props
  2. cloth_drag    -- cloth dragged across table with props
  3. cloth_throw   -- garment launched toward table
  4. cloth_lay     -- cloth settles onto table + props
  5. cloth_fold    -- cloth edges folded inward
  6. cloth_curtain -- hanging cloth with wind
  7. cloth_pull    -- cloth pulled off table edge

Usage:
    uv run python scripts/disco/test_newton_cloth_sims.py
    uv run python scripts/disco/test_newton_cloth_sims.py --sims drop drag
    uv run python scripts/disco/test_newton_cloth_sims.py --preset silk --seed 42
    uv run python scripts/disco/test_newton_cloth_sims.py --device cuda:0
    uv run python scripts/disco/test_newton_cloth_sims.py --list-assets
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import warp as wp
from fabric_library import FABRICS
from newton_sim_utils import (
    CLOTH_PRESETS,
    create_viewer,
    discover_garments,
    discover_usda_props,
    discover_usda_scenes,
    load_config,
    write_sim_metadata,
)

# =============================================================================
# Test functions for each sim type
# =============================================================================


def test_cloth_drop(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth drop simulation."""
    from newton_cloth_drop_sim import build_cloth_drop_scene, run_cloth_drop

    sim_name = f"test_drop_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_drop_scene(
        config=config,
        seed=seed,
        preset=preset,
        num_props=min(2, kw.get("num_props", 2)),
        use_grid=kw.get("use_grid", True),
        grid_res=kw.get("grid_res", 20),
        drop_height=0.5,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    final = run_cloth_drop(model, solver, num_frames=num_frames, substeps=5, viewer=viewer)
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_drag(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth drag simulation."""
    from newton_cloth_drag_sim import build_cloth_drag_scene, run_cloth_drag

    sim_name = f"test_drag_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_drag_scene(
        config=config,
        seed=seed,
        preset=preset,
        num_props=min(2, kw.get("num_props", 2)),
        grid_res=kw.get("grid_res", 20),
        cloth_size=0.6,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    final = run_cloth_drag(
        model,
        solver,
        num_frames=num_frames,
        substeps=5,
        drag_speed=0.3,
        viewer=viewer,
    )
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_throw(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth throw simulation."""
    from newton_cloth_throw_sim import build_cloth_throw_scene, run_cloth_throw

    sim_name = f"test_throw_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_throw_scene(
        config=config,
        seed=seed,
        preset=preset,
        num_props=1,
        use_grid=True,
        grid_res=kw.get("grid_res", 20),
        throw_speed=2.0,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    final = run_cloth_throw(model, solver, num_frames=num_frames, substeps=5, viewer=viewer)
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_lay(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth lay simulation."""
    from newton_cloth_lay_sim import build_cloth_lay_scene, run_cloth_lay

    sim_name = f"test_lay_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_lay_scene(
        config=config,
        seed=seed,
        preset=preset,
        num_props=1,
        use_grid=True,
        grid_res=kw.get("grid_res", 20),
        hover_gap=0.05,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    final = run_cloth_lay(model, solver, num_frames=num_frames, substeps=5, viewer=viewer)
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_fold(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth fold simulation."""
    from newton_cloth_fold_sim import build_cloth_fold_scene, run_cloth_fold

    sim_name = f"test_fold_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_fold_scene(
        config=config,
        seed=seed,
        preset=preset,
        grid_res=kw.get("grid_res", 20),
        cloth_size=0.6,
        fold_axis="x",
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    fold_frames = min(num_frames // 2, 60)
    final = run_cloth_fold(
        model,
        solver,
        num_frames=num_frames,
        substeps=5,
        fold_speed=0.2,
        fold_axis="x",
        fold_frames=fold_frames,
        viewer=viewer,
    )
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_curtain(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth curtain/wind simulation."""
    from newton_cloth_curtain_sim import build_curtain_scene, run_curtain_sim

    sim_name = f"test_curtain_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_curtain_scene(
        config=config,
        seed=seed,
        preset=preset,
        grid_res=kw.get("grid_res", 20),
        curtain_width=1.0,
        curtain_height=0.8,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    final = run_curtain_sim(
        model,
        solver,
        num_frames=num_frames,
        substeps=5,
        wind_strength=3.0,
        viewer=viewer,
    )
    return _validate_state(final, sim_name, sim_dir, scene_info)


def test_cloth_pull(config, preset, seed, num_frames, output_dir, viewer_type, **kw):
    """Test cloth pull simulation."""
    from newton_cloth_pull_sim import build_cloth_pull_scene, run_cloth_pull

    sim_name = f"test_pull_{preset}_s{seed}"
    sim_dir = output_dir / sim_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    model, solver, scene_info = build_cloth_pull_scene(
        config=config,
        seed=seed,
        preset=preset,
        num_props=1,
        grid_res=kw.get("grid_res", 20),
        cloth_size=0.6,
        solver_iters=5,
    )

    usd_path = sim_dir / f"{sim_name}.usda"
    viewer = create_viewer(output_path=usd_path, viewer_type=viewer_type)

    pull_frames = min(num_frames // 2, 60)
    final = run_cloth_pull(
        model,
        solver,
        num_frames=num_frames,
        substeps=5,
        pull_speed=0.4,
        pull_frames=pull_frames,
        viewer=viewer,
    )
    return _validate_state(final, sim_name, sim_dir, scene_info)


# =============================================================================
# Validation
# =============================================================================


def _validate_state(final_state, sim_name, sim_dir, scene_info):
    """Check the final simulation state for sanity."""
    result = {"name": sim_name, "passed": True, "warnings": [], "scene": scene_info}

    if final_state.particle_count > 0:
        pq = final_state.particle_q.numpy()
        min_pos = np.min(pq, axis=0)
        max_pos = np.max(pq, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        result["bbox_size"] = float(bbox_size)
        result["min_z"] = float(min_pos[2])
        result["particle_count"] = int(pq.shape[0])

        if bbox_size > 20.0:
            result["warnings"].append(f"Possible explosion: bbox_size={bbox_size:.2f}")
        if min_pos[2] < -1.0:
            result["warnings"].append(f"Deep ground penetration: min_z={min_pos[2]:.3f}")
        if np.any(np.isnan(pq)):
            result["warnings"].append("NaN detected in particle positions!")
            result["passed"] = False
        if np.any(np.abs(pq) > 100):
            result["warnings"].append("Extreme positions (>100m) detected!")
            result["passed"] = False
    else:
        result["warnings"].append("No particles in final state (rigid-body-only scene?)")

    write_sim_metadata(
        sim_dir / "test_result.json",
        sim_type=sim_name,
        seed=0,
        **result,
    )

    return result


# =============================================================================
# Asset inventory
# =============================================================================


def list_assets(config):
    """Print discovered assets from all sources."""
    print("\n" + "=" * 70)
    print("ASSET INVENTORY")
    print("=" * 70)

    # Garments
    garment_dir = config.get("garment_dir", "")
    garments = discover_garments(garment_dir) if garment_dir else []
    print(f"\n  GARMENTS: {len(garments)}  ({garment_dir})")
    categories = {}
    for g in garments:
        categories.setdefault(g["category"], []).append(g)
    for cat, items in sorted(categories.items()):
        print(f"    {cat:<35s}  {len(items):>3d} garments")

    # USDA Props
    props_dir = config.get("props_usda_dir", config.get("props_dir", ""))
    usda_props = discover_usda_props(props_dir) if props_dir else []
    print(f"\n  USDA PROPS: {len(usda_props)}  ({props_dir})")
    for p in usda_props[:10]:
        dims = p["metadata"].get("dimensions_m", [0, 0, 0])
        print(f"    {p['name']:<30s}  {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}m")
    if len(usda_props) > 10:
        print(f"    ... and {len(usda_props) - 10} more")

    # USDA Scenes
    scenes_dir = config.get("scenes_usda_dir", config.get("scenes_dir", ""))
    scenes = discover_usda_scenes(scenes_dir) if scenes_dir else []
    print(f"\n  USDA SCENES: {len(scenes)}  ({scenes_dir})")
    for s in scenes:
        dims = s["metadata"].get("dimensions_m", [0, 0, 0])
        print(f"    {s['name']:<30s}  {s['category']:<10s}  {dims}")

    # Fabrics
    print(f"\n  FABRIC PRESETS: {len(CLOTH_PRESETS)}")
    for name in sorted(CLOTH_PRESETS.keys()):
        p = CLOTH_PRESETS[name]
        print(f"    {name:<14s}  density={p['density']:.2f}  tri_ke={p['tri_ke']:.0e}  edge_ke={p['edge_ke']:.0e}")

    print()


# =============================================================================
# Main
# =============================================================================

SIM_REGISTRY = {
    "drop": test_cloth_drop,
    "drag": test_cloth_drag,
    "throw": test_cloth_throw,
    "lay": test_cloth_lay,
    "fold": test_cloth_fold,
    "curtain": test_cloth_curtain,
    "pull": test_cloth_pull,
}


def main():
    parser = argparse.ArgumentParser(description="Newton Cloth Simulation Test Suite")
    parser.add_argument(
        "--sims", nargs="*", default=None, choices=list(SIM_REGISTRY.keys()), help="Sim types to test (default: all)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="cotton",
        help=f"Cloth preset (available: {', '.join(sorted(CLOTH_PRESETS.keys()))})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-frames", type=int, default=30, help="Frames per test (default: 30 for quick test)")
    parser.add_argument("--grid-res", type=int, default=20, help="Grid cloth resolution (default: 20 for quick test)")
    parser.add_argument(
        "--viewer", type=str, default="usd", choices=["null", "usd", "gl"], help="Viewer type (usd writes .usda files)"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--list-assets", action="store_true", help="Print discovered assets and exit")
    parser.add_argument("--list-fabrics", action="store_true", help="Print all fabric presets and exit")
    args = parser.parse_args()

    wp.init()
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_preferred_device()

    config = load_config(args.config)

    if args.list_assets:
        list_assets(config)
        return

    if args.list_fabrics:
        print(f"\n{'=' * 70}")
        print(f"FABRIC LIBRARY: {len(FABRICS)} fabrics")
        print(f"{'=' * 70}\n")
        for name, props in sorted(FABRICS.items(), key=lambda x: x[1]["density"]):
            n = props.get("newton", {})
            print(
                f"  {name:<14s}  {props['weight_class']:<10s}  "
                f"density={props['density']:.2f}  friction={props['friction']:.2f}  "
                f"newton: tri_ke={n.get('tri_ke', '?'):>6}  edge_ke={n.get('edge_ke', '?')}"
            )
        print()
        return

    if args.preset not in CLOTH_PRESETS:
        print(f"ERROR: Unknown preset '{args.preset}'. Available: {sorted(CLOTH_PRESETS.keys())}")
        sys.exit(1)

    sims_to_run = args.sims or list(SIM_REGISTRY.keys())
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output")) / "test_suite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEWTON CLOTH SIMULATION TEST SUITE")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  Preset:     {args.preset}")
    print(f"  Seed:       {args.seed}")
    print(f"  Frames:     {args.num_frames}")
    print(f"  Grid res:   {args.grid_res}")
    print(f"  Viewer:     {args.viewer}")
    print(f"  Output:     {output_dir}")
    print(f"  Sims:       {', '.join(sims_to_run)}")
    print("=" * 70)

    results = []
    total_t0 = time.time()

    with wp.ScopedDevice(device):
        for sim_name in sims_to_run:
            sim_fn = SIM_REGISTRY[sim_name]
            print(f"\n{'─' * 50}")
            print(f"  TEST: {sim_name} (preset={args.preset})")
            print(f"{'─' * 50}")

            t0 = time.time()
            try:
                result = sim_fn(
                    config=config,
                    preset=args.preset,
                    seed=args.seed,
                    num_frames=args.num_frames,
                    output_dir=output_dir,
                    viewer_type=args.viewer,
                    grid_res=args.grid_res,
                )
                elapsed = time.time() - t0
                result["time_s"] = round(elapsed, 1)

                status = "PASS" if result["passed"] else "FAIL"
                bbox = result.get("bbox_size", "?")
                particles = result.get("particle_count", "?")
                print(f"  [{status}] {sim_name}: {elapsed:.1f}s, {particles} particles, bbox={bbox}")

                if result["warnings"]:
                    for w in result["warnings"]:
                        print(f"    ⚠ {w}")

            except Exception as e:
                elapsed = time.time() - t0
                result = {"name": sim_name, "passed": False, "error": str(e), "time_s": round(elapsed, 1)}
                print(f"  [FAIL] {sim_name}: {e}")
                import traceback

                traceback.print_exc()

            results.append(result)

    total_time = time.time() - total_t0

    # Summary
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed

    print(f"\n{'=' * 70}")
    print(f"TEST SUMMARY: {passed}/{len(results)} passed, {failed} failed ({total_time:.1f}s)")
    print(f"{'=' * 70}")

    for r in results:
        status = "✓" if r.get("passed") else "✗"
        name = r.get("name", "?")
        t = r.get("time_s", "?")
        extra = ""
        if "bbox_size" in r:
            extra = f"  bbox={r['bbox_size']:.2f}"
        if "error" in r:
            extra = f"  ERROR: {r['error'][:60]}"
        if r.get("warnings"):
            extra += f"  ({len(r['warnings'])} warnings)"
        print(f"  {status} {name:<30s}  {t:>5.1f}s{extra}")

    print(f"\n  Output: {output_dir}")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
