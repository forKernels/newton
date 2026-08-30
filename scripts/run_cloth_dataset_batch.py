# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Batch runner for cloth dataset pickup simulations.
#
# Scans a dataset directory for garment OBJ meshes, runs Newton GPU
# cloth+Franka simulations, and saves each animation as USDA.
#
# Usage:
#   uv run python newton/examples/cloth/run_cloth_dataset_batch.py \
#       --dataset-dir "D:\_blender\_myBlender\SimulationWork\ClothDataset" \
#       --output-dir "D:\_blender\_myBlender\SimulationWork\ClothOutput" \
#       --categories dress_sleeveless_2550 jacket_2200 \
#       --device cuda:0 --num-frames 600 --max-items 10
#
# Each garment produces:
#   <output-dir>/<category>/<garment_id>.usda   (animation)
#   <output-dir>/<category>/<garment_id>.usda.json (metadata)

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_garment_objs(dataset_dir: str, categories: list[str] | None = None) -> list[dict]:
    """Scan dataset directory for *_sim.obj garment meshes.

    Expected structure:
        dataset_dir/<category>/<garment_id>/<garment_id>_sim.obj

    Returns list of dicts with keys: category, garment_id, obj_path
    """
    dataset = Path(dataset_dir)
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Discover category folders
    if categories:
        cat_dirs = [dataset / c for c in categories]
        cat_dirs = [d for d in cat_dirs if d.is_dir()]
        if not cat_dirs:
            raise FileNotFoundError(f"None of the requested categories found in {dataset_dir}: {categories}")
    else:
        # Auto-discover: pick directories that contain subdirectories with _sim.obj files
        cat_dirs = sorted(d for d in dataset.iterdir() if d.is_dir() and not d.name.endswith(".rar"))

    items = []
    for cat_dir in cat_dirs:
        category = cat_dir.name
        for garment_dir in sorted(cat_dir.iterdir()):
            if not garment_dir.is_dir():
                continue
            # Look for *_sim.obj
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


def run_single_simulation(
    obj_path: str,
    output_path: str,
    device: str = "cuda:0",
    num_frames: int = 600,
    robot_target_mode: str = "bbox-grasp",
    robot: str = "franka",
    extra_args: list[str] | None = None,
) -> dict:
    """Run a single cloth_dataset_pickup simulation as a subprocess.

    Returns dict with status, duration, and any error info.
    """
    cmd = [
        sys.executable,
        "-m",
        "newton.examples",
        "cloth_dataset_pickup",
        "--viewer",
        "usd",
        "--output-path",
        output_path,
        "--device",
        device,
        "--num-frames",
        str(num_frames),
        "--cloth-path",
        obj_path,
        "--cloth-scale-auto",
        "--cloth-center-mode",
        "bbox-center-ground",
        "--robot-target-mode",
        robot_target_mode,
        "--robot",
        robot,
        "--robot-start-delay",
        "1.5",
        "--quiet",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per garment
            cwd=str(Path(__file__).resolve().parents[1]),  # newton_zhao root
            env=env,
        )
        duration = time.time() - t0
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "duration_s": round(duration, 2),
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "duration_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"status": "exception", "error": str(e), "duration_s": round(time.time() - t0, 2)}


def main():
    parser = argparse.ArgumentParser(description="Batch cloth dataset simulation runner")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root dataset directory containing category folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for USDA animations.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Category folder names to process. Omit to process all.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Warp device (default: cuda:0).")
    parser.add_argument("--num-frames", type=int, default=600, help="Simulation frames per garment (default: 600).")
    parser.add_argument(
        "--robot-target-mode",
        type=str,
        default="bbox-grasp",
        choices=["default", "bbox-grasp", "bbox-fold"],
        help="Robot trajectory mode (default: bbox-grasp).",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        choices=["franka", "ur5e-robotiq", "ur5e-leap", "ur5e-shadow"],
        help="Robot arm+gripper configuration (default: franka).",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Max items to process (for testing).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip garments that already have output USDA.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this item index (for resuming).")
    parser.add_argument("--dry-run", action="store_true", help="List items without running simulations.")
    parser.add_argument("--extra-args", type=str, nargs="*", default=None, help="Extra args passed to the example.")
    parser.add_argument(
        "--openusd-root",
        type=str,
        default=None,
        help="Path to OpenUSD install (sets OPENUSD_ROOT env var for subprocesses). "
        "Can also be set via OPENUSD_ROOT environment variable.",
    )
    args = parser.parse_args()

    # Propagate OpenUSD root to environment for subprocesses
    if args.openusd_root:
        os.environ["OPENUSD_ROOT"] = args.openusd_root
    if sys.platform.startswith("win") and not os.environ.get("OPENUSD_ROOT") and not os.environ.get("USD_ROOT"):
        print("WARNING: OPENUSD_ROOT is not set. USD output will fail on Windows.")
        print("         Set it via --openusd-root or the OPENUSD_ROOT environment variable.")

    # Discover garments
    print(f"Scanning dataset: {args.dataset_dir}")
    items = find_garment_objs(args.dataset_dir, args.categories)
    print(f"Found {len(items)} garments")

    if not items:
        print("No garments found. Check --dataset-dir and --categories.")
        return

    # Apply start index and max items
    items = items[args.start_index :]
    if args.max_items is not None:
        items = items[: args.max_items]
    print(f"Processing {len(items)} garments (start_index={args.start_index})")

    if args.dry_run:
        for i, item in enumerate(items):
            print(f"  [{args.start_index + i}] {item['category']}/{item['garment_id']}")
        return

    # Process
    output_dir = Path(args.output_dir)
    results_log = output_dir / "batch_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    err_count = 0
    skip_count = 0
    total = len(items)

    for i, item in enumerate(items):
        idx = args.start_index + i
        cat = item["category"]
        gid = item["garment_id"]
        obj_path = item["obj_path"]

        cat_output_dir = output_dir / cat
        cat_output_dir.mkdir(parents=True, exist_ok=True)
        usda_path = str(cat_output_dir / f"{gid}.usda")

        if args.skip_existing and os.path.isfile(usda_path):
            skip_count += 1
            print(f"[{idx}] SKIP {cat}/{gid} (exists)")
            continue

        print(f"[{idx}] ({i + 1}/{total}) {cat}/{gid} ...", end=" ", flush=True)

        result = run_single_simulation(
            obj_path=obj_path,
            output_path=usda_path,
            device=args.device,
            num_frames=args.num_frames,
            robot_target_mode=args.robot_target_mode,
            robot=args.robot,
            extra_args=args.extra_args,
        )

        status = result["status"]
        dur = result.get("duration_s", 0)

        if status == "ok":
            ok_count += 1
            print(f"OK ({dur:.1f}s)")
        else:
            err_count += 1
            tail = result.get("stderr_tail", result.get("error", ""))
            # Print just the last line of error for brevity
            last_line = tail.strip().split("\n")[-1] if tail.strip() else "unknown error"
            print(f"FAIL ({dur:.1f}s) {last_line}")

        # Append to results log
        log_entry = {"index": idx, "category": cat, "garment_id": gid, "obj_path": obj_path, "output": usda_path}
        log_entry.update(result)
        with open(results_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nDone: {ok_count} ok, {err_count} errors, {skip_count} skipped")
    print(f"Results log: {results_log}")


if __name__ == "__main__":
    main()
