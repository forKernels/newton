---
name: "cli-anything-newton"
description: "CLI harness for the Newton GPU-accelerated physics simulation engine. Load URDF/MJCF/USD scenes, run physics simulations with multiple solvers, export results, and preprocess meshes — all headlessly from the command line."
---

# cli-anything-newton

CLI interface for the **Newton** physics simulation engine. Enables AI agents to
operate Newton headlessly: load robot/cloth/scene models, run GPU-accelerated
physics simulations, and export results without a GUI.

## Prerequisites

- Newton physics engine (installed via `pip install -e .` in the Newton repo)
- NVIDIA Warp (`warp-lang>=1.11.0`)
- NVIDIA GPU with CUDA (recommended; CPU fallback available)

## Installation

```bash
cd /path/to/newton/agent-harness
pip install -e .
```

## Sessions and `--dry-run`

The REPL keeps a `Session` - loaded scene, solver, frame counter, history. It
is **in-memory by default and lost on exit**. To persist it:

```bash
cli-anything-newton --session run.json          # auto-saves after mutations
cli-anything-newton --session run.json --dry-run  # never writes
```

Inside the REPL, `save [path]` writes on demand. With `--session` set, `solver`
and `step` auto-save, and quitting with unsaved changes saves rather than
discarding. Without it, quitting while modified prints a warning naming the
flag instead of losing the work silently.

`--dry-run` suppresses every write while leaving the rest of the behaviour
identical, so a mutation can be rehearsed before it touches the file.

## Preview: publish with this, inspect with `cli-hub`

Producer and consumer are separate roles. This harness PUBLISHES bundles;
`cli-hub previews` reads them and never renders.

```bash
cli-anything-newton preview recipes --json
cli-anything-newton preview capture scene.usda --recipe usd --json
cli-anything-newton preview capture --procedural pendulum --recipe quick --json
cli-anything-newton preview latest --recipe usd --json

cli-hub previews inspect /path/to/bundle          # the consumer half
```

Recipes, and what they cost:

| recipe | frames | USD | for |
|---|---|---|---|
| `quick` | 30 | no | does this scene simulate at all |
| `usd` | 60 | yes | the ordinary look |
| `settle` | 180 | yes | piles - stacking failures do not show in 60 |

**Everything in a bundle comes out of the real solver.** A recipe runs the same
`run_simulation` loop `sim run` uses, with one of Newton's own viewers attached,
and writes what the viewer writes. Nothing is plotted, reconstructed, or
approximated. ViewerUSD and ViewerFile work headless; ViewerGL needs a display
and is therefore not a preview path on a build machine.

Artifact roles: `hero` is the time-sampled USD, `gallery` is the state and model
summary. A recipe that promised USD and produced none publishes `status:
partial` with a warning rather than `ok`, so a consumer never trusts a hero
artifact that is not there.

Bundles are immutable and content-addressed by a source fingerprint: a scene
FILE by its bytes, so editing it invalidates the cache; a procedural scene by
the parameters that generate it, since those are its source. Re-capturing an
unchanged source returns the existing bundle marked `reused` rather than
burning a GPU minute to produce the same bytes. `--force` overrides.

`preview latest` is READ-ONLY and never renders. `--dry-run` suppresses all
writes.

The preview default solver is `mujoco`.

## Command Groups

### `scene` — Load and inspect scenes
```bash
cli-anything-newton scene load <path.urdf|.xml|.usd>
cli-anything-newton scene info <path>
cli-anything-newton scene procedural <ground|cloth_grid|pendulum>
```

### `sim` — Run simulations
```bash
cli-anything-newton sim run <scene_path> --solver <type> --frames <N> -o <output>
cli-anything-newton sim solvers
```
Solvers: `xpbd`, `vbd`, `mujoco`, `featherstone`, `semi_implicit`, `style3d`, `mpm`

### `export` — Export results
```bash
cli-anything-newton export formats
```
Formats: USD (time-sampled), JSON (state snapshot), ViewerFile (recording)

### `mesh` — Mesh preprocessing for cloth
```bash
cli-anything-newton mesh inspect <path.obj|.stl|.glb>
cli-anything-newton mesh stitch <path> --threshold 0.0005
cli-anything-newton mesh check <path>
```

### `example` — Built-in Newton examples
```bash
cli-anything-newton example list
cli-anything-newton example run <name> --frames 200
```

### `info` — System info
```bash
cli-anything-newton info
```

## Agent Usage

### JSON output
Add `--json` before any subcommand for machine-readable output:
```bash
cli-anything-newton --json sim run scene.urdf --solver xpbd --frames 100
cli-anything-newton --json info
cli-anything-newton --json mesh inspect garment.obj
```

### Device selection
```bash
cli-anything-newton --device cuda:0 sim run scene.urdf --solver mujoco --frames 500
cli-anything-newton --device cpu sim run scene.urdf --solver xpbd --frames 50
```

### Common workflows

**Robot simulation:**
```bash
cli-anything-newton --json sim run robot.urdf --solver mujoco --frames 500 --substeps 16 -o result.usd
```

**Cloth simulation pipeline:**
```bash
cli-anything-newton mesh inspect garment.obj
cli-anything-newton mesh stitch garment.obj --threshold 0.0005
cli-anything-newton mesh check garment_stitched.obj
```

**Inspect a scene without running:**
```bash
cli-anything-newton --json scene info robot.urdf
```

### Interactive REPL
```bash
cli-anything-newton  # enters REPL mode by default
```
REPL commands: `load`, `solver`, `step`, `run`, `export`, `status`, `help`, `quit`

## Error Handling

- Missing Newton/Warp: clear install instructions in error message
- Invalid scene format: lists supported extensions
- Invalid solver: lists available solvers
- File not found: exact path in error message
