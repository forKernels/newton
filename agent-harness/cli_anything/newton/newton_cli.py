# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cli-anything-newton: CLI harness for the Newton physics simulation engine.

Provides headless, agent-driven access to Newton's simulation capabilities:
  - Load URDF/MJCF/USD scenes
  - Run physics simulations with configurable solvers
  - Export results to USD/JSON
  - Inspect models, meshes, and simulation state
  - Interactive REPL mode for exploratory workflows
"""

from __future__ import annotations

import json
import sys

import click

# ── JSON output helper ────────────────────────────────────────────────


class _Ctx:
    """Shared context for JSON output mode."""

    def __init__(self):
        self.json_mode = False
        self.session_path = None
        self.dry_run = False

    def output(self, data: dict):
        """Print data as JSON (if --json) or skip (human output handled inline)."""
        if self.json_mode:
            click.echo(json.dumps(data, indent=2, default=str))


pass_ctx = click.make_pass_decorator(_Ctx, ensure=True)


# ══════════════════════════════════════════════════════════════════════
#  Main CLI group
# ══════════════════════════════════════════════════════════════════════


@click.group(invoke_without_command=True)
@click.option("--json", "json_mode", is_flag=True, help="Machine-readable JSON output.")
@click.option("--device", default="cuda:0", help="Warp device (cuda:0, cpu, etc.).")
@click.option("--session", "session_path", default=None, type=click.Path(),
              help="Persist the REPL session to this JSON file. Without it the "
                   "session is in-memory only and is lost on exit.")
@click.option("--dry-run", is_flag=True,
              help="Never write the session file, even when --session is set.")
@click.pass_context
def cli(ctx, json_mode, device, session_path, dry_run):
    """Newton Physics Engine — CLI Harness.

    Headless, agent-driven interface to Newton's GPU-accelerated physics
    simulation. Load scenes, run sims, export results.
    """
    obj = _Ctx()
    obj.json_mode = json_mode
    obj.device = device
    obj.session_path = session_path
    obj.dry_run = dry_run
    ctx.ensure_object(dict)
    ctx.obj = obj

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


# ══════════════════════════════════════════════════════════════════════
#  scene — Load and inspect scenes
# ══════════════════════════════════════════════════════════════════════


@cli.group()
def scene():
    """Load and inspect simulation scenes (URDF/MJCF/USD)."""
    pass


@scene.command("load")
@click.argument("scene_path", type=click.Path(exists=True))
@click.option("--collapse-fixed-joints/--no-collapse-fixed-joints", default=True)
@click.option("--self-collisions/--no-self-collisions", default=False)
@click.option("--num-worlds", default=1, type=int, help="Number of parallel worlds.")
@click.option("--up-axis", default="y", type=click.Choice(["y", "z"]))
@pass_ctx
def scene_load(ctx, scene_path, collapse_fixed_joints, self_collisions, num_worlds, up_axis):
    """Load a scene file and print model info."""
    from cli_anything.newton.core.scene import load_scene

    result = load_scene(
        scene_path,
        device=ctx.device,
        collapse_fixed_joints=collapse_fixed_joints,
        enable_self_collisions=self_collisions,
        num_worlds=num_worlds,
        up_axis=up_axis,
    )

    info = result["scene_info"]
    if ctx.json_mode:
        ctx.output(info)
    else:
        click.echo(f"Loaded: {info['path']}")
        click.echo(f"  Format:    {info['parser']}")
        click.echo(f"  Bodies:    {info['body_count']}")
        click.echo(f"  Joints:    {info['joint_count']}")
        click.echo(f"  Shapes:    {info['shape_count']}")
        click.echo(f"  Particles: {info['particle_count']}")
        click.echo(f"  Springs:   {info['spring_count']}")
        click.echo(f"  Triangles: {info['tri_count']}")


@scene.command("info")
@click.argument("scene_path", type=click.Path(exists=True))
@click.option("--up-axis", default="y", type=click.Choice(["y", "z"]))
@pass_ctx
def scene_info(ctx, scene_path, up_axis):
    """Show detailed info about a scene file."""
    from cli_anything.newton.core.scene import get_model_info, load_scene

    result = load_scene(scene_path, device=ctx.device, up_axis=up_axis)
    info = get_model_info(result["model"])
    info["scene_path"] = scene_path

    if ctx.json_mode:
        ctx.output(info)
    else:
        click.echo(f"Scene: {scene_path}")
        click.echo(f"  Bodies:     {info['body_count']}")
        click.echo(f"  Joints:     {info['joint_count']}")
        click.echo(f"  Shapes:     {info['shape_count']}")
        click.echo(f"  Particles:  {info['particle_count']}")
        click.echo(f"  Springs:    {info['spring_count']}")
        click.echo(f"  Triangles:  {info['tri_count']}")
        click.echo(f"  Tets:       {info['tet_count']}")
        click.echo(f"  Edges:      {info['edge_count']}")
        click.echo(f"  Worlds:     {info['world_count']}")
        click.echo(f"  Gravity:    {info['gravity']}")
        if info.get("shape_types"):
            click.echo(f"  Shape types: {info['shape_types']}")


@scene.command("procedural")
@click.argument("scene_type", type=click.Choice(["ground", "cloth_grid", "pendulum"]))
@click.option("--resolution", default=30, type=int, help="Grid resolution (cloth_grid).")
@click.option("--size", default=1.0, type=float, help="Grid size in meters (cloth_grid).")
@click.option("--num-links", default=5, type=int, help="Number of links (pendulum).")
@pass_ctx
def scene_procedural(ctx, scene_type, resolution, size, num_links):
    """Build a procedural scene (no file needed)."""
    from cli_anything.newton.core.scene import build_procedural_scene

    result = build_procedural_scene(
        scene_type=scene_type,
        device=ctx.device,
        resolution=resolution,
        size=size,
        num_links=num_links,
    )
    info = result["scene_info"]

    if ctx.json_mode:
        ctx.output(info)
    else:
        click.echo(f"Built procedural scene: {scene_type}")
        for k, v in info.items():
            if k not in ("path", "format"):
                click.echo(f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════
#  sim — Run simulations
# ══════════════════════════════════════════════════════════════════════


@cli.group()
def sim():
    """Run physics simulations."""
    pass


@sim.command("run")
@click.argument("scene_path", type=click.Path(exists=True))
@click.option(
    "--solver",
    "solver_type",
    default="xpbd",
    type=click.Choice(["vbd", "xpbd", "mujoco", "featherstone", "semi_implicit", "style3d", "mpm"]),
)
@click.option("--frames", default=100, type=int, help="Number of frames to simulate.")
@click.option("--substeps", default=8, type=int, help="Physics substeps per frame.")
@click.option("--dt", default=1.0 / 60.0, type=float, help="Frame timestep (seconds).")
@click.option("--iterations", default=10, type=int, help="Solver iterations per substep.")
@click.option("--collision-margin", default=0.01, type=float, help="Collision margin (meters).")
@click.option("--output", "-o", default=None, help="Output file path (USD/JSON).")
@click.option("--output-format", default="usd", type=click.Choice(["usd", "json", "null"]))
@click.option("--up-axis", default="y", type=click.Choice(["y", "z"]))
@pass_ctx
def sim_run(
    ctx, scene_path, solver_type, frames, substeps, dt, iterations, collision_margin, output, output_format, up_axis
):
    """Load a scene and run a full simulation."""
    from cli_anything.newton.core.export import create_viewer
    from cli_anything.newton.core.scene import load_scene
    from cli_anything.newton.core.simulation import create_solver, run_simulation

    # Load scene
    if not ctx.json_mode:
        click.echo(f"Loading scene: {scene_path}")
    scene = load_scene(scene_path, device=ctx.device, up_axis=up_axis)
    model = scene["model"]

    # Create solver
    if not ctx.json_mode:
        click.echo(f"Solver: {solver_type} (iterations={iterations})")
    solver = create_solver(model, solver_type, iterations)

    # Create viewer for output
    viewer = None
    if output_format != "null" and output:
        viewer = create_viewer(output_format, output, up_axis=up_axis)

    # Progress callback
    def progress(frame, total):
        if not ctx.json_mode and frame % 25 == 0:
            click.echo(f"  Frame {frame}/{total}")

    # Run simulation
    if not ctx.json_mode:
        click.echo(f"Simulating {frames} frames ({substeps} substeps, dt={dt:.4f}s)...")
    result = run_simulation(
        model,
        solver,
        num_frames=frames,
        substeps=substeps,
        dt=dt,
        collision_margin=collision_margin,
        viewer=viewer,
        progress_callback=progress,
    )

    result["scene_path"] = scene_path
    result["solver"] = solver_type
    result["output"] = output

    if ctx.json_mode:
        ctx.output(result)
    else:
        click.echo(f"Done: {result['num_frames']} frames in {result['wall_time_s']}s")
        click.echo(f"  FPS: {result['fps']}")
        click.echo(f"  Steps/s: {result['steps_per_second']}")
        if output:
            click.echo(f"  Output: {output}")


@sim.command("solvers")
@pass_ctx
def sim_solvers(ctx):
    """List available physics solvers."""
    from cli_anything.newton.core.simulation import list_solvers

    solvers = list_solvers()

    if ctx.json_mode:
        ctx.output({"solvers": solvers})
    else:
        click.echo("Available solvers:")
        for s in solvers:
            click.echo(f"  {s['name']:<16} {s['description']}")


# ══════════════════════════════════════════════════════════════════════
#  export — Export simulation results
# ══════════════════════════════════════════════════════════════════════


@cli.group()
def export():
    """Export simulation results."""
    pass


@export.command("formats")
@pass_ctx
def export_formats(ctx):
    """List available export formats."""
    from cli_anything.newton.core.export import list_export_formats

    formats = list_export_formats()

    if ctx.json_mode:
        ctx.output({"formats": formats})
    else:
        click.echo("Available export formats:")
        for f in formats:
            req = f" (requires: {f['requires']})" if f["requires"] else ""
            click.echo(f"  {f['format']:<8} {f['extension']:<10} {f['description']}{req}")


# ══════════════════════════════════════════════════════════════════════
#  mesh — Mesh preprocessing
# ══════════════════════════════════════════════════════════════════════


@cli.group()
def mesh():
    """Mesh preprocessing for cloth simulation."""
    pass


@mesh.command("inspect")
@click.argument("mesh_path", type=click.Path(exists=True))
@pass_ctx
def mesh_inspect(ctx, mesh_path):
    """Inspect a mesh file and print statistics."""
    from cli_anything.newton.core.mesh import inspect_mesh

    info = inspect_mesh(mesh_path)

    if ctx.json_mode:
        ctx.output(info)
    else:
        click.echo(f"Mesh: {info['path']}")
        click.echo(f"  Vertices:    {info['vertex_count']}")
        click.echo(f"  Faces:       {info['face_count']}")
        click.echo(f"  Components:  {info['num_components']}")
        click.echo(f"  Watertight:  {info['is_watertight']}")
        click.echo(f"  BBox min:    {info['bbox_min']}")
        click.echo(f"  BBox max:    {info['bbox_max']}")
        click.echo(f"  Extents:     {info['extents']}")
        click.echo(f"  Area:        {info['area']:.4f} m^2")
        if info.get("volume") is not None:
            click.echo(f"  Volume:      {info['volume']:.6f} m^3")


@mesh.command("stitch")
@click.argument("mesh_path", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output path for stitched mesh.")
@click.option("--threshold", default=0.0005, type=float, help="Vertex merge distance threshold (meters).")
@pass_ctx
def mesh_stitch(ctx, mesh_path, output, threshold):
    """Stitch garment seams by merging near-duplicate vertices."""
    from cli_anything.newton.core.mesh import stitch_mesh

    result = stitch_mesh(mesh_path, output_path=output, threshold=threshold)

    if ctx.json_mode:
        ctx.output(result)
    else:
        click.echo(f"Stitched: {result['input']}")
        click.echo(f"  Threshold:    {result['threshold']}m")
        click.echo(
            f"  Vertices:     {result['original_vertices']} -> {result['stitched_vertices']} "
            f"(-{result['vertices_merged']})"
        )
        click.echo(f"  Faces:        {result['original_faces']} -> {result['stitched_faces']}")
        click.echo(f"  Components:   {result['original_components']} -> {result['stitched_components']}")
        click.echo(f"  Output:       {result['output']}")


@mesh.command("check")
@click.argument("mesh_path", type=click.Path(exists=True))
@pass_ctx
def mesh_check(ctx, mesh_path):
    """Check mesh connectivity for cloth simulation readiness."""
    from cli_anything.newton.core.mesh import check_connectivity

    result = check_connectivity(mesh_path)

    if ctx.json_mode:
        ctx.output(result)
    else:
        click.echo(f"Mesh: {result['path']}")
        click.echo(f"  Vertices:     {result['vertex_count']}")
        click.echo(f"  Faces:        {result['face_count']}")
        click.echo(f"  Components:   {result['num_components']}")
        click.echo(f"  Connected:    {result['is_single_component']}")
        click.echo(f"  Watertight:   {result['is_watertight']}")
        if result.get("components"):
            click.echo("  Component breakdown:")
            for c in result["components"]:
                click.echo(f"    [{c['index']}] {c['vertex_count']} verts, {c['face_count']} faces")


# ══════════════════════════════════════════════════════════════════════
#  example — Run built-in Newton examples
# ══════════════════════════════════════════════════════════════════════


@cli.group()
def example():
    """Run built-in Newton examples."""
    pass


@example.command("list")
@pass_ctx
def example_list(ctx):
    """List available Newton examples."""
    from cli_anything.newton.utils.newton_backend import get_examples

    examples = get_examples()

    if ctx.json_mode:
        ctx.output({"examples": examples})
    else:
        if examples:
            click.echo("Available examples:")
            for ex in examples:
                click.echo(f"  {ex['name']}")
        else:
            click.echo("No examples found. Is Newton installed?")


@example.command("run")
@click.argument("name")
@click.option("--frames", default=100, type=int)
@click.option("--viewer", default="null", type=click.Choice(["null", "usd", "gl", "rerun"]))
@click.option("--output-path", default=None)
@pass_ctx
def example_run(ctx, name, frames, viewer, output_path):
    """Run a built-in Newton example by name."""
    import subprocess

    cmd = [sys.executable, "-m", "newton.examples", name, "--viewer", viewer, "--num-frames", str(frames)]
    if output_path:
        cmd.extend(["--output-path", output_path])
    cmd.extend(["--device", ctx.device])

    if not ctx.json_mode:
        click.echo(f"Running example: {name}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if ctx.json_mode:
        ctx.output(
            {
                "example": name,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    else:
        if result.stdout:
            click.echo(result.stdout)
        if result.returncode != 0:
            click.echo(f"Example failed (exit {result.returncode})", err=True)
            if result.stderr:
                click.echo(result.stderr, err=True)


# ═════════════════════════════════════════════════════════════════════
#  preview — publish bundles of a real simulation (PRODUCER)
# ═════════════════════════════════════════════════════════════════════


@cli.group()
def preview():
    """Publish preview bundles. Inspect them with `cli-hub previews`.

    Producer and consumer are separate roles on purpose: this half talks to the
    real backend and writes bundles; `cli-hub previews inspect|html|watch|open`
    reads them and never renders.
    """


@preview.command("recipes")
@pass_ctx
def preview_recipes(ctx):
    """What each recipe produces, and what it costs."""
    from cli_anything.newton.core.preview import list_recipes
    rs = list_recipes()
    ctx.output({"recipes": rs})
    if not ctx.json_mode:
        for r in rs:
            click.echo(f"{r['name']:>8}  {r['frames']:>4} frames  "
                       f"usd={str(r['usd']):<5}  {r['description']}")


@preview.command("capture")
@click.argument("scene", type=click.Path(exists=True), required=False)
@click.option("--procedural", "scene_type",
              type=click.Choice(["ground", "cloth_grid", "pendulum"]),
              help="build a scene instead of loading one")
@click.option("--recipe", default="quick", help="quick | usd | settle")
@click.option("--solver", "solver_type", default="mujoco")
@click.option("--force", is_flag=True, help="re-render even on a cache hit")
@pass_ctx
def preview_capture(ctx, scene, scene_type, recipe, solver_type, force):
    """Run the real solver and publish an immutable bundle."""
    from cli_anything.newton.core.preview import capture
    if getattr(ctx, "dry_run", False):
        ctx.output({"dry_run": True, "recipe": recipe,
                    "would_capture": scene or f"procedural:{scene_type}"})
        if not ctx.json_mode:
            click.echo(f"dry-run: would capture {scene or scene_type} [{recipe}]")
        return
    try:
        m = capture(scene_path=scene, scene_type=scene_type, recipe=recipe,
                    solver_type=solver_type, device=ctx.device, force=force)
    except ValueError as exc:
        raise SystemExit(str(exc))
    ctx.output(m)
    if not ctx.json_mode:
        click.echo(f"{m.get('status','ok')}  {m['bundle_id']}"
                   f"{'  (reused)' if m.get('reused') else ''}")
        click.echo(f"  {m['_bundle_dir']}")
        for a in m.get("artifacts", []):
            click.echo(f"  {a['role']:>7}  {a['label']}")
        for w in m.get("warnings") or []:
            click.echo(f"  ! {w}")
        click.echo(f"\n  inspect it:  cli-hub previews inspect {m['_bundle_dir']}")


@preview.command("latest")
@click.option("--recipe", default=None)
@click.option("--scene", type=click.Path(), default=None)
@pass_ctx
def preview_latest(ctx, recipe, scene):
    """The newest existing bundle. Never renders."""
    from cli_anything.newton.core.preview import latest
    m = latest(recipe=recipe, scene_path=scene)
    ctx.output(m if m else {"found": False})
    if not ctx.json_mode:
        if not m:
            click.echo("no bundle yet - run `preview capture`")
        else:
            click.echo(f"{m['bundle_id']}  {m.get('recipe')}  {m['_bundle_dir']}")
            click.echo(f"  inspect it:  cli-hub previews inspect {m['_bundle_dir']}")


# ══════════════════════════════════════════════════════════════════════
#  info — System and backend info
# ══════════════════════════════════════════════════════════════════════


@cli.command("signature")
@click.argument("targets", nargs=-1)
@click.option("--accepts", default=None,
              help="Comma-separated keywords: would each one REACH this build, "
                   "or be silently dropped?")
@click.option("--list", "list_known", is_flag=True,
              help="List the well-known targets and exit.")
@pass_ctx
def signature(ctx, targets, accepts, list_known):
    """Accepted keyword names of any newton API, read off the real build.

    Newton's keyword lists move between versions and an unaccepted keyword is
    DROPPED rather than refused, so a wrong name produces a working call that
    ignores the setting. `density=` moved onto ShapeConfig exactly this way and
    every rigid body silently used the default for a whole release.

    Never guess a keyword against this engine. Ask it.

        newton-cli signature ModelBuilder.add_shape_mesh
        newton-cli signature solvers.SolverMuJoCo --accepts njmax,nconmax
    """
    from cli_anything.newton.core import signature as sig

    if list_known:
        data = {"well_known": sig.WELL_KNOWN,
                "note": "any dotted path under the newton module also works"}
        if ctx.json_mode:
            ctx.output(data)
        else:
            for name in sig.WELL_KNOWN:
                click.echo(name)
        return

    if not targets:
        targets = ("ModelBuilder.ShapeConfig",)

    try:
        import newton
    except ImportError as exc:
        msg = f"newton is not importable in this interpreter: {exc}"
        if ctx.json_mode:
            ctx.output({"error": msg})
        else:
            click.echo(msg, err=True)
        sys.exit(1)

    probe = [k for k in (accepts or "").split(",") if k.strip()]
    results = sig.signatures(newton, list(targets), accepts=probe)

    if ctx.json_mode:
        ctx.output({"newton": getattr(newton, "__version__", "?"),
                    "results": results})
        return

    for entry in results:
        if not entry.get("resolved"):
            click.echo(f"{entry['target']}: {entry['error']}", err=True)
            continue
        click.echo(f"{entry['target']}  [{entry['kind']}]")
        if not entry["introspectable"]:
            click.echo(f"  not introspectable: {entry['error']}")
            continue
        if entry["required"]:
            # These cannot be rescued by dropping them - omitting one raises.
            click.echo(f"  required: {', '.join(entry['required'])}")
        click.echo(f"  keywords: {', '.join(entry['keywords']) or '(none)'}")
        if entry["accepts_var_keyword"]:
            click.echo("  takes **kwargs - nothing is dropped, and nothing "
                       "is validated either")
        if "accepts" in entry:
            for name, ok in entry["accepts"].items():
                click.echo(f"    {name}: {'ACCEPTED' if ok else 'WOULD BE DROPPED'}")


@cli.command("info")
@pass_ctx
def info(ctx):
    """Show Newton backend info (versions, devices)."""
    from cli_anything.newton.utils.newton_backend import find_newton

    try:
        backend = find_newton()
    except RuntimeError as e:
        if ctx.json_mode:
            ctx.output({"error": str(e)})
        else:
            click.echo(str(e), err=True)
        sys.exit(1)

    if ctx.json_mode:
        ctx.output(backend)
    else:
        click.echo(f"Newton:  {backend['newton_version']}")
        click.echo(f"Warp:    {backend['warp_version']}")
        click.echo(
            f"CUDA:    {'yes' if backend['cuda_available'] else 'no'} ({backend.get('cuda_device_count', 0)} devices)"
        )
        if backend.get("devices"):
            click.echo(f"Devices: {', '.join(backend['devices'])}")


# ══════════════════════════════════════════════════════════════════════
#  REPL — Interactive mode
# ══════════════════════════════════════════════════════════════════════


@cli.command("repl", hidden=True)
@click.pass_context
def repl(ctx):
    """Interactive REPL mode (default when no subcommand given)."""
    from cli_anything.newton.core.session import Session
    from cli_anything.newton.utils.repl_skin import ReplSkin

    parent_ctx = ctx.parent or ctx
    obj = parent_ctx.obj if hasattr(parent_ctx, "obj") and isinstance(parent_ctx.obj, _Ctx) else _Ctx()

    skin = ReplSkin("newton", version="1.0.0")
    skin.print_banner()

    session = Session(obj.session_path)
    pt_session = skin.create_prompt_session()

    def autosave(what):
        """Persist after a mutation, unless suppressed.

        Session tracked `modified` and DISPLAYED it from the first commit, but
        `Session(session_path)` was only ever constructed with no path, the
        REPL had no save command, and there was no --session flag - so the
        whole persistence path, file locking included, was unreachable. A user
        could load a scene, step a hundred frames, watch the prompt say
        modified, quit, and lose all of it silently. `modified` was a promise
        the harness had no way to keep.
        """
        if obj.dry_run:
            skin.status("dry-run", f"{what} not saved")
            return
        if not session.session_path:
            return
        try:
            session.save()
            skin.status("saved", session.session_path)
        except Exception as exc:      # a failed save must be LOUD
            skin.error(f"could not save session to {session.session_path}: {exc}")

    commands = {
        "help":      "Show this help",
        "info":      "Show Newton backend info",
        "load":      "Load a scene file (load <path>)",
        "status":    "Show current session status",
        "solver":    "Set solver (solver <type> [iterations])",
        "solvers":   "List available solvers",
        "step":      "Advance simulation by N frames (step [N])",
        "run":       "Run full simulation (run <frames>)",
        "export":    "Export state (export <path>)",
        "mesh":      "Inspect mesh (mesh inspect <path>)",
        "stitch":    "Stitch mesh (stitch <path> [threshold])",
        "examples":  "List built-in examples",
        "save":      "Save the session (save [path])",
        "quit":      "Exit the REPL",
    }

    while True:
        try:
            line = skin.get_input(
                pt_session,
                project_name=session.scene_name,
                modified=session.modified,
            )
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("quit", "exit", "q"):
                if session.modified and not session.session_path:
                    skin.error("unsaved changes and no --session path: this "
                               "session is being discarded. Re-run with "
                               "--session <file>, or use `save <path>`.")
                elif session.modified:
                    autosave("session")
                break

            elif cmd == "save":
                target = args[0] if args else None
                if obj.dry_run:
                    skin.status("dry-run", "not saved")
                    continue
                try:
                    skin.success(f"Saved: {session.save(target)}")
                except ValueError:
                    skin.error("No save path. Use `save <path>` or start the "
                               "REPL with --session <file>.")

            elif cmd == "help":
                skin.help(commands)

            elif cmd == "info":
                from cli_anything.newton.utils.newton_backend import find_newton

                try:
                    backend = find_newton()
                    skin.status("Newton", backend["newton_version"])
                    skin.status("Warp", backend["warp_version"])
                    skin.status("CUDA", "yes" if backend["cuda_available"] else "no")
                    if backend.get("devices"):
                        skin.status("Devices", ", ".join(backend["devices"]))
                except RuntimeError as e:
                    skin.error(str(e))

            elif cmd == "load":
                if not args:
                    skin.error("Usage: load <scene_path>")
                    continue
                session.load_scene(args[0], device=obj.device)
                skin.success(f"Loaded {session.scene_name}")
                skin.status("Bodies", str(session.model.body_count))
                skin.status("Joints", str(session.model.joint_count))
                skin.status("Particles", str(session.model.particle_count))

            elif cmd == "status":
                status = session.get_status()
                for k, v in status.items():
                    skin.status(k, str(v))

            elif cmd == "solver":
                if not args:
                    skin.error("Usage: solver <type> [iterations]")
                    continue
                solver_type = args[0]
                iters = int(args[1]) if len(args) > 1 else 10
                session.set_solver(solver_type, iters)
                skin.success(f"Solver set: {solver_type} (iterations={iters})")
                autosave("solver")

            elif cmd == "solvers":
                from cli_anything.newton.core.simulation import list_solvers

                for s in list_solvers():
                    skin.status(s["name"], s["description"])

            elif cmd == "step":
                n = int(args[0]) if args else 1
                for i in range(n):
                    session.step()
                skin.success(f"Advanced {n} frame(s). Now at frame {session.frame}")
                autosave("step")

            elif cmd == "run":
                if not args:
                    skin.error("Usage: run <num_frames>")
                    continue
                frames = int(args[0])
                for i in range(frames):
                    session.step()
                    if (i + 1) % 25 == 0:
                        skin.info(f"Frame {i + 1}/{frames}")
                skin.success(f"Simulated {frames} frames. Now at frame {session.frame}")

            elif cmd == "export":
                if not args:
                    skin.error("Usage: export <output_path>")
                    continue
                if not session.is_loaded:
                    skin.error("No scene loaded.")
                    continue
                from cli_anything.newton.core.export import export_state_json

                result = export_state_json(session.state_0, session.model, args[0])
                skin.success(f"Exported to {result['output']} ({result['file_size']:,} bytes)")

            elif cmd == "mesh":
                if len(args) < 2:
                    skin.error("Usage: mesh inspect <path>")
                    continue
                if args[0] == "inspect":
                    from cli_anything.newton.core.mesh import inspect_mesh

                    info = inspect_mesh(args[1])
                    for k, v in info.items():
                        if k != "path":
                            skin.status(k, str(v))

            elif cmd == "stitch":
                if not args:
                    skin.error("Usage: stitch <mesh_path> [threshold]")
                    continue
                from cli_anything.newton.core.mesh import stitch_mesh

                threshold = float(args[1]) if len(args) > 1 else 0.0005
                result = stitch_mesh(args[0], threshold=threshold)
                skin.success(f"Stitched: {result['original_vertices']} -> {result['stitched_vertices']} vertices")
                skin.status("Output", result["output"])

            elif cmd == "examples":
                from cli_anything.newton.utils.newton_backend import get_examples

                examples = get_examples()
                if examples:
                    for ex in examples:
                        skin.status(ex["name"], ex.get("module", ""))
                else:
                    skin.warning("No examples found.")

            else:
                skin.error(f"Unknown command: {cmd}. Type 'help' for commands.")

        except Exception as e:
            skin.error(str(e))

    skin.print_goodbye()


# ── Entry point ───────────────────────────────────────────────────────


def main():
    cli()


if __name__ == "__main__":
    main()
