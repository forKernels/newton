"""Publish preview bundles of a real Newton simulation.

Producer half only. Per the CLI-Anything preview methodology, production and
consumption are separate roles:

    publish   cli-anything-newton preview capture ...
    inspect   cli-hub previews inspect /path/to/bundle

This module never reads or renders an existing bundle for display; that is
`cli-hub`'s job.

**Everything here comes out of the real solver.** The truthfulness rule ranks
sources: native render/export from the real backend first, native inspection
second, offscreen capture third, and explicitly rules out anything synthesized
outside the tool. So a recipe runs `run_simulation` - the same loop `sim run`
uses - with one of Newton's own viewers attached, and writes what the viewer
writes. There is no plotting, no reconstruction, and no approximation of what
Newton "would" produce.

Which viewers: **ViewerUSD** writes time-sampled USD and **ViewerFile** writes
Newton's own recording format, and both work headless. ViewerGL needs a
display, so it is not a preview path on a build machine and is not offered as
one.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.preview_bundle import (
    artifact_record,
    build_cache_key,
    find_latest_manifest,
    fingerprint_data,
    finalize_bundle,
    prepare_bundle,
)

SOFTWARE = "newton"
HARNESS_VERSION = "1.0.0"

#: Each recipe names what it PRODUCES, not how pretty it is. `frames` is the
#: knob that costs wall clock; everything else is fixed so two captures of the
#: same scene are comparable.
RECIPES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "description": "30 frames, state snapshot only. The cheapest honest "
                       "look at whether a scene simulates at all.",
        "frames": 30, "substeps": 8, "usd": False,
    },
    "usd": {
        "description": "60 frames exported as time-sampled USD through "
                       "Newton's own ViewerUSD, plus the state snapshot.",
        "frames": 60, "substeps": 8, "usd": True,
    },
    "settle": {
        "description": "180 frames as USD - long enough to see a pile settle "
                       "or fail to. Stacking problems do not show in 60.",
        "frames": 180, "substeps": 8, "usd": True,
    },
}


def list_recipes() -> List[Dict[str, Any]]:
    return [{"name": k, **v} for k, v in RECIPES.items()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _source_fingerprint(scene_path: Optional[str], scene_type: Optional[str],
                        recipe: str, solver: str) -> str:
    """What makes this capture the same capture.

    A scene FILE is fingerprinted by its bytes, so editing it invalidates the
    cache. A procedural scene has no file, so it is fingerprinted by the
    parameters that generate it - which is the same statement, since those
    parameters ARE the source.
    """
    if scene_path:
        p = Path(scene_path)
        return fingerprint_data({
            "path": str(p.resolve()), "size": p.stat().st_size,
            "mtime": int(p.stat().st_mtime), "recipe": recipe, "solver": solver,
        })
    return fingerprint_data({"procedural": scene_type, "recipe": recipe,
                             "solver": solver})


def capture(scene_path: Optional[str] = None, scene_type: Optional[str] = None,
            recipe: str = "quick", solver_type: str = "mujoco",
            device: str = "cuda:0", root_dir: Optional[str] = None,
            force: bool = False) -> Dict[str, Any]:
    """Run the real solver and publish one immutable bundle."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; have {sorted(RECIPES)}")
    if not scene_path and not scene_type:
        raise ValueError("give either a scene file or --procedural <type>")

    cfg = RECIPES[recipe]
    fp = _source_fingerprint(scene_path, scene_type, recipe, solver_type)
    options = {"solver": solver_type, "frames": cfg["frames"],
               "substeps": cfg["substeps"], "usd": cfg["usd"]}

    prep = prepare_bundle(
        software=SOFTWARE, recipe=recipe, bundle_kind="static",
        source_fingerprint=fp, options=options,
        harness_version=HARNESS_VERSION, project_path=scene_path,
        root_dir=root_dir, force=force,
    )
    # A cache hit is a real answer: the source has not changed, so re-rendering
    # would produce the same bytes and burn a GPU minute saying so.
    if prep.get("cached") and prep.get("manifest"):
        return {**prep["manifest"], "reused": True}

    bundle_dir = Path(prep["bundle_dir"])
    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from .scene import build_procedural_scene, get_model_info, load_scene
    from .simulation import create_solver, run_simulation

    built = (load_scene(scene_path, device=device) if scene_path
             else build_procedural_scene(scene_type, device=device))
    model = built["model"]
    solver = create_solver(model, solver_type=solver_type)

    viewer = None
    usd_path = artifacts_dir / "capture.usda"
    if cfg["usd"]:
        from newton.viewer import ViewerUSD
        viewer = ViewerUSD(output_path=str(usd_path), fps=60,
                           num_frames=cfg["frames"])

    t0 = time.perf_counter()
    # run_simulation CLOSES the viewer itself. Closing it again here raised
    # `'NoneType' object has no attribute 'GetRootLayer'` from viewer_usd - the
    # USD had already been written and the stage dropped, so the capture had
    # actually succeeded and the harness reported a traceback over the top of it.
    result = run_simulation(model, solver, num_frames=cfg["frames"],
                            substeps=cfg["substeps"], viewer=viewer)
    wall = time.perf_counter() - t0

    artifacts = []
    state_path = artifacts_dir / "state.json"
    info = get_model_info(model)
    state_path.write_text(json.dumps({"model": info, "result": result},
                                     indent=2, default=str))
    artifacts.append(artifact_record(
        str(bundle_dir), str(state_path), "state", "gallery", "data",
        "final state and model summary", media_type="application/json"))
    if cfg["usd"] and usd_path.exists():
        artifacts.append(artifact_record(
            str(bundle_dir), str(usd_path), "usd", "hero", "scene",
            f"{cfg['frames']} frames, time-sampled USD",
            media_type="model/vnd.usda"))

    # A recipe that promised USD and produced none is PARTIAL, not ok. Saying
    # ok here would let a consumer trust a hero artifact that is not there.
    warnings = []
    status = "ok"
    if cfg["usd"] and not usd_path.exists():
        status = "partial"
        warnings.append("ViewerUSD wrote no file; the bundle carries the state "
                        "snapshot only and has no hero artifact")

    return finalize_bundle(
        bundle_dir=str(bundle_dir), bundle_id=prep["bundle_id"],
        bundle_kind="static", software=SOFTWARE, recipe=recipe,
        source={"scene": scene_path or f"procedural:{scene_type}",
                "fingerprint": fp, "captured_at": _now()},
        artifacts=artifacts,
        summary={"recipe": recipe, "solver": solver_type,
                 "frames": cfg["frames"], "substeps": cfg["substeps"],
                 "wall_seconds": round(wall, 3),
                 "bodies": info.get("body_count"),
                 "shapes": info.get("shape_count"),
                 "particles": info.get("particle_count")},
        cache_key=prep["cache_key"],
        generator={"harness": "cli-anything-newton",
                   "harness_version": HARNESS_VERSION,
                   "backend": "newton", "device": device},
        status=status, warnings=warnings or None,
        metrics={"wall_seconds": round(wall, 3), "frames": cfg["frames"]},
    )


def latest(recipe: Optional[str] = None, scene_path: Optional[str] = None,
           root_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The newest existing bundle. READ-ONLY - never renders.

    `preview latest` exists so an agent can ask "what is current" without
    paying for a capture. If it rendered on a miss it would be a slow `capture`
    wearing a cheap name, and the loop would stop being predictable.
    """
    return find_latest_manifest(software=SOFTWARE, recipe=recipe,
                                bundle_kind="static", project_path=scene_path,
                                root_dir=root_dir)
