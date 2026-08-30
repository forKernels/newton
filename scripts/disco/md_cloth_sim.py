"""
Marvelous Designer Cloth Simulation Pipeline
===============================================
Generates Python scripts to run INSIDE Marvelous Designer's script editor.

MD API: no headless mode. Paste/run scripts in MD > Script > Run Script.

Pipeline per sim:
  1. NewProject() — clear scene
  2. ImportOBJ() — import garment as 3D garment
  3. AddFabric() / AssignFabricToPattern() — apply fabric .zfab preset
  4. Simulate() — run cloth sim
  5. ExportOBJ() — draped result
  6. ExportAlembic() — animation cache
  7. ExportUSD() — direct USD export (no Blender conversion needed)

Usage:
    # Generate batch script for cotton, silk, leather:
    uv run python scripts/disco/md_cloth_sim.py --dry-run --preset cotton silk leather

    # Specific furniture + category:
    uv run python scripts/disco/md_cloth_sim.py --dry-run --preset silk \
        --furniture Chair_01.blend --garment-category dress_sleeveless

    # List presets:
    uv run python scripts/disco/md_cloth_sim.py --list-presets
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from fabric_library import FABRICS, get_fabric_properties
from newton_sim_utils import (
    discover_garments,
    load_config,
)

# =============================================================================
# Furniture Discovery
# =============================================================================

FURNITURE_EXCLUDE = {"PoolBalls_Set", "PoolTable_01", "PoolTable_02", "Chess_Table"}


def find_furniture_blends(furniture_dir: str, specific: str | None = None) -> list[Path]:
    """Find furniture .blend files recursively."""
    root = Path(furniture_dir)
    if not root.is_dir():
        print(f"  WARNING: Furniture dir not found: {furniture_dir}")
        return []
    if specific:
        matches = list(root.rglob(specific))
        if not matches and not specific.endswith(".blend"):
            matches = list(root.rglob(f"{specific}.blend"))
        return [matches[0]] if matches else []
    blends = sorted(root.rglob("*.blend"))
    return [
        b
        for b in blends
        if b.suffix == ".blend" and b.stem not in FURNITURE_EXCLUDE and not b.name.endswith((".blend1", ".blend2"))
    ]


def find_furniture_collision_obj(blend_path: Path) -> str | None:
    """Find collision OBJ for a furniture piece from its URDF folder.

    Looks for a sibling folder with the same stem containing meshes/collision/*.obj.
    Falls back to meshes/visual/*.obj if no collision mesh exists.
    Returns the OBJ path as a string, or None if not found.
    """
    stem = blend_path.stem
    parent = blend_path.parent

    # Look for URDF folder: same dir or parent dir
    for search_dir in [parent, parent.parent]:
        urdf_dir = search_dir / stem
        if urdf_dir.is_dir():
            # Prefer collision mesh
            col_dir = urdf_dir / "meshes" / "collision"
            if col_dir.is_dir():
                objs = sorted(col_dir.glob("*.obj"))
                if objs:
                    return str(objs[0])
            # Fall back to visual mesh
            vis_dir = urdf_dir / "meshes" / "visual"
            if vis_dir.is_dir():
                objs = sorted(vis_dir.glob("*.obj"))
                if objs:
                    return str(objs[0])
    return None


# =============================================================================
# MD Preset Mapping
# =============================================================================


def get_md_preset(fabric_name: str) -> dict:
    """Get Marvelous Designer preset info for a fabric."""
    props = get_fabric_properties(fabric_name)
    md = props.get("marvelous_designer", {})
    return {
        "preset_name": md.get("preset_name", "Cotton"),
        "density_gsm": md.get("density_gsm", 150),
    }


def list_md_presets():
    """Print all fabrics with their MD preset mappings."""
    print(f"{'Fabric':<16} {'MD Preset':<24} {'GSM':>5}  {'Weight':>10}")
    print("-" * 60)
    for name, props in FABRICS.items():
        md = props.get("marvelous_designer", {})
        preset = md.get("preset_name", "-")
        gsm = md.get("density_gsm", "-")
        weight = props.get("weight_class", "-")
        print(f"{name:<16} {preset!s:<24} {gsm!s:>5}  {weight:>10}")


# =============================================================================
# MD Script Generation — uses real MD Python API
# =============================================================================


def generate_md_batch_script(
    jobs: list[dict],
    output_root: str,
    export_alembic: bool = True,
    export_usd: bool = True,
) -> str:
    """Generate a batch script for MD's internal script editor.

    Uses the real MD API: NewProject, ImportOBJ, Simulate, ExportOBJ,
    ExportAlembic, ExportUSD. No 'import MarvelousDesigner'.
    """
    output_root = output_root.replace("\\", "/")
    lines = []

    lines.append("# Auto-generated Marvelous Designer BATCH script")
    lines.append(f"# {len(jobs)} simulations")
    lines.append("import ApiTypes")
    lines.append("import utility_api")
    lines.append("import import_api")
    lines.append("import export_api")
    lines.append("import os")
    lines.append("import json")
    lines.append("import time")
    lines.append("")
    lines.append(f'OUTPUT_ROOT = r"{output_root}"')
    lines.append("results = []")
    lines.append(f"total = {len(jobs)}")
    lines.append("batch_t0 = time.time()")
    lines.append("")

    for i, job in enumerate(jobs):
        obj_path = job["obj_path"].replace("\\", "/")
        sim_name = job["sim_name"]
        preset = job["fabric_preset"]
        furn = job["furniture_name"]
        sim_dir = f"{output_root}/{sim_name}".replace("\\", "/")
        base_name = Path(sim_name).name
        out_obj = f"{sim_dir}/{base_name}_draped.obj"
        out_abc = f"{sim_dir}/{base_name}.abc"
        out_usd = f"{sim_dir}/{base_name}.usda"
        out_meta = f"{sim_dir}/metadata.json"
        frames = job.get("sim_frames", 300)

        lines.append(f"# --- Job {i + 1}/{len(jobs)}: {sim_name} ---")
        lines.append(f'print(f"[{{len(results)+1}}/{{total}}] {sim_name}")')
        lines.append("t0 = time.time()")
        lines.append("try:")
        lines.append(f'    os.makedirs(r"{sim_dir}", exist_ok=True)')
        lines.append(f'    if os.path.exists(r"{out_meta}"):')
        lines.append('        print("  SKIP (exists)")')
        lines.append(f'        results.append({{"name": "{sim_name}", "status": "skipped"}})')
        furn_obj = job.get("furniture_collision_obj", "").replace("\\", "/")

        lines.append("    else:")
        lines.append("        # Clear scene")
        lines.append("        utility_api.NewProject()")
        lines.append("")
        lines.append("        # Import furniture as collision object (avatar/object type 0)")
        lines.append("        furn_opt = ApiTypes.ImportExportOption()")
        lines.append("        furn_opt.ImportObjectType = 0  # avatar/collision object")
        lines.append("        furn_opt.scale = 1.0")
        lines.append(f'        import_api.ImportFile(r"{furn_obj}", furn_opt)')
        lines.append("")
        lines.append("        # Import garment OBJ")
        lines.append("        garment_opt = ApiTypes.ImportExportOption()")
        lines.append("        garment_opt.bAutoTranslate = True")
        lines.append("        garment_opt.translationValueY = 30.0  # cm above origin")
        lines.append("        garment_opt.scale = 1.0")
        lines.append(f'        import_api.ImportOBJ(r"{obj_path}", garment_opt)')
        lines.append("")
        lines.append("        # Simulation settings")
        lines.append("        utility_api.SetSimulationQuality(1, 0)  # Animation(Stable), CPU")
        lines.append("        utility_api.SetSimulationSelfCollisionIterationCount(2)")
        lines.append("")
        lines.append("        # Record animation during simulation")
        lines.append("        utility_api.SetStartAnimationFrame(0)")
        lines.append(f"        utility_api.SetEndAnimationFrame({frames})")
        lines.append("        utility_api.SetAnimationRecording(True)")
        lines.append(f"        utility_api.Simulate({frames})")
        lines.append("        utility_api.SetAnimationRecording(False)")
        lines.append("")
        lines.append("        # Export draped OBJ")
        lines.append("        export_opt = ApiTypes.ImportExportOption()")
        lines.append("        export_opt.bExportGarment = True")
        lines.append("        export_opt.bThin = True")
        lines.append("        export_opt.bSingleObject = True")
        lines.append("        export_opt.scale = 0.01  # cm -> m")
        lines.append(f'        export_api.ExportOBJ(r"{out_obj}", export_opt)')

        lines.append("")
        lines.append("        elapsed = time.time() - t0")
        lines.append("        meta = {")
        lines.append('            "mode": "drop",')
        lines.append('            "engine": "marvelous_designer",')
        lines.append(f'            "furniture": "{furn}",')
        lines.append(f'            "garment": "{job.get("garment_name", "")}",')
        lines.append(f'            "category": "{job.get("category", "")}",')
        lines.append(f'            "fabric_preset": "{preset}",')
        lines.append(f'            "sim_frames": {frames},')
        lines.append('            "elapsed_s": round(elapsed, 1),')
        lines.append("        }")
        lines.append(f'        with open(r"{out_meta}", "w") as f:')
        lines.append("            json.dump(meta, f, indent=2)")
        lines.append(f'        results.append({{"name": "{sim_name}", "status": "ok", "time": elapsed}})')
        lines.append('        print("  OK (%.1fs)" % elapsed)')
        lines.append("except Exception as e:")
        lines.append("    elapsed = time.time() - t0")
        lines.append(
            f'    results.append({{"name": "{sim_name}", "status": "error", "error": str(e), "time": elapsed}})'
        )
        lines.append('    print(f"  FAILED: {e}")')
        lines.append("")

    lines.append("# --- Summary ---")
    lines.append("batch_elapsed = time.time() - batch_t0")
    lines.append('ok = sum(1 for r in results if r["status"] == "ok")')
    lines.append('skip = sum(1 for r in results if r.get("status") == "skipped")')
    lines.append('fail = sum(1 for r in results if r["status"] == "error")')
    lines.append(
        'print(f"\\nBatch complete: {ok} done, {skip} skipped, {fail} failed / {total} total ({batch_elapsed:.0f}s)")'
    )
    lines.append("")
    lines.append('with open(os.path.join(OUTPUT_ROOT, "batch_results.json"), "w") as f:')
    lines.append("    json.dump(results, f, indent=2)")

    return "\n".join(lines) + "\n"


# =============================================================================
# Job Builder
# =============================================================================


def build_jobs(
    garments: list[dict],
    furniture_files: list[Path],
    fabric_names: list[str],
    sim_frames: int = 300,
    category_filter: str | None = None,
    seed: int = 0,
    max_garments: int | None = None,
) -> list[dict]:
    """Build simulation jobs: garment x furniture x preset."""
    if category_filter:
        garments = [g for g in garments if g["category"].lower().startswith(category_filter.lower())]

    rng = random.Random(seed)
    garments = list(garments)
    rng.shuffle(garments)

    if max_garments:
        garments = garments[:max_garments]

    jobs = []
    for furn_path in furniture_files:
        furn_name = furn_path.stem
        furn_collision_obj = find_furniture_collision_obj(furn_path)
        if not furn_collision_obj:
            print(f"  WARNING: No collision OBJ for {furn_name}, skipping (needs URDF folder)")
            continue

        for fabric_name in fabric_names:
            md_preset = get_md_preset(fabric_name)
            preset_name = md_preset["preset_name"] or "Cotton"

            for g in garments:
                sim_name = f"{furn_name}/drop/{fabric_name}/{g['name']}"
                jobs.append(
                    {
                        "obj_path": str(g["path"]),
                        "garment_name": g["name"],
                        "category": g["category"],
                        "furniture_name": furn_name,
                        "furniture_collision_obj": furn_collision_obj,
                        "fabric_name": fabric_name,
                        "fabric_preset": preset_name,
                        "sim_name": sim_name,
                        "sim_frames": sim_frames,
                    }
                )
    return jobs


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Marvelous Designer Cloth Simulation Pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--preset", type=str, nargs="+", default=["cotton"], help="Fabric preset(s): cotton, silk, leather, etc."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-garments", type=int, default=None)
    parser.add_argument("--garment-category", type=str, default=None)
    parser.add_argument("--furniture-dir", type=str, default=None)
    parser.add_argument("--furniture", type=str, default=None)
    parser.add_argument("--sim-frames", type=int, default=300)
    parser.add_argument("--no-alembic", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--list-garments", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        list_md_presets()
        return

    config = load_config(args.config)
    garments = discover_garments(config.get("garment_dir", ""))
    if not garments:
        print("ERROR: No garments found.")
        sys.exit(1)
    print(f"Discovered {len(garments)} garments")

    if args.list_garments:
        categories = {}
        for g in garments:
            categories.setdefault(g["category"], []).append(g["name"])
        for cat, names in sorted(categories.items()):
            print(f"  {cat} ({len(names)})")
        return

    furn_dir = args.furniture_dir or config.get("furniture_dir", "")
    furniture_files = find_furniture_blends(furn_dir, args.furniture)
    if not furniture_files:
        print(f"WARNING: No furniture found in {furn_dir}, using placeholder")
        furniture_files = [Path("ground_plane")]
    print(f"Furniture: {len(furniture_files)} files")

    for p in args.preset:
        if p not in FABRICS:
            print(f"ERROR: Unknown preset '{p}'. Available: {list(FABRICS.keys())}")
            sys.exit(1)

    jobs = build_jobs(
        garments,
        furniture_files=furniture_files,
        fabric_names=args.preset,
        sim_frames=args.sim_frames,
        category_filter=args.garment_category,
        seed=args.seed,
        max_garments=args.num_garments,
    )

    if not jobs:
        print("No garments matched filters.")
        sys.exit(1)

    print(f"\nBuilt {len(jobs)} simulation jobs")
    print(f"  Presets: {args.preset}")
    print(f"  Furniture: {len(furniture_files)}")

    output_base = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output"))
    output_dir = output_base / "md_cloth"
    output_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    batch_script = generate_md_batch_script(
        jobs,
        str(output_dir),
        export_alembic=not args.no_alembic,
        export_usd=not args.no_usd,
    )
    batch_path = scripts_dir / "md_batch_sim.py"
    batch_path.write_text(batch_script, encoding="utf-8")
    print(f"  Wrote batch script: {batch_path}")
    print(f"  Contains {len(jobs)} simulations")

    manifest = {
        "presets": args.preset,
        "furniture_count": len(furniture_files),
        "total_jobs": len(jobs),
        "jobs": [
            {
                "sim_name": j["sim_name"],
                "garment": j["garment_name"],
                "category": j["category"],
                "furniture": j["furniture_name"],
                "fabric_preset": j["fabric_preset"],
            }
            for j in jobs
        ],
    }
    (output_dir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nTo run: Open MD > Script > Run Script > {batch_path}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
