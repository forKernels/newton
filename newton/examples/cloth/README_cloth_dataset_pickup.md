# Cloth Dataset Pickup — Batch Garment Simulation

Simulates a Franka Emika Panda robot arm interacting with garment meshes from
external datasets (e.g. [Garment-Pattern-Generator](https://github.com/maria-korosteleva/Garment-Pattern-Generator)),
saving each animation as a USDA file.

## Setup (all machines)

Before running, set the `OPENUSD_ROOT` environment variable to your local
OpenUSD install. Both `<root>\bin` and `<root>\lib` are automatically added
to the DLL search path.

**Windows (CMD):**
```cmd
set OPENUSD_ROOT=C:\_tools\OpenUSD\25.08
```

**Windows (PowerShell):**
```powershell
$env:OPENUSD_ROOT = "C:\_tools\OpenUSD\25.08"
```

**Linux / macOS:**
```bash
export OPENUSD_ROOT=/opt/openusd/25.08
```

To make it permanent, add to your shell profile or system environment variables.

## Quick Start

### Single garment

```bash
uv run -m newton.examples cloth_dataset_pickup ^
  --viewer usd ^
  --output-path output/my_dress.usda ^
  --cloth-path path/to/garment_sim.obj ^
  --cloth-scale-auto ^
  --cloth-center-mode bbox-center-ground ^
  --robot-target-mode bbox-grasp ^
  --device cuda:0 ^
  --num-frames 600
```

### Batch processing

```bash
uv run python scripts/run_cloth_dataset_batch.py ^
  --dataset-dir path/to/ClothDataset ^
  --output-dir path/to/ClothOutput ^
  --categories dress_sleeveless_2550 jacket_2200 ^
  --device cuda:0 ^
  --num-frames 600 ^
  --skip-existing
```

The batch runner also accepts `--openusd-root` if you prefer not to set the
environment variable:

```bash
uv run python scripts/run_cloth_dataset_batch.py ^
  --openusd-root "C:\_tools\OpenUSD\25.08" ^
  --dataset-dir path/to/ClothDataset ^
  --output-dir path/to/ClothOutput
```

## Dataset Structure

The batch runner expects this layout (matches Garment-Pattern-Generator output):

```
ClothDataset/
  dress_sleeveless_2550/
    dress_sleeveless_000YCTJ9HS/
      dress_sleeveless_000YCTJ9HS_sim.obj    <-- simulation mesh (triangulated)
      specification.json
      ...
    dress_sleeveless_02785H9ILO/
      ...
  jacket_2200/
    ...
  pants_straight_sides_1000/
    ...
```

Each garment folder must contain a `*_sim.obj` file. The meshes are triangulated
OBJ files in **centimeters** (the OBJ header states this; `--cloth-scale-auto`
detects the unit automatically).

## Output

Per garment, two files are produced:

| File | Contents |
|------|----------|
| `<garment_id>.usda` | Animated USD scene with Franka arm + cloth mesh, time-sampled at 60 fps |
| `<garment_id>.usda.json` | Metadata sidecar (cloth scale, bounding box, sim parameters, etc.) |

The batch runner also writes `batch_results.jsonl` — one JSON line per garment
with status, duration, and any error details.

## How It Works

1. **Mesh loading** — The `*_sim.obj` is loaded via trimesh (triangulated meshes required)
2. **Auto-scaling** — `--cloth-scale-auto` inspects the bounding box extent:
   - Extents > 250 units = millimeters (scale 0.001)
   - Extents 3-250 = centimeters (scale 0.01)
   - Extents < 3 = meters (scale 1.0)
3. **Recentering** — `--cloth-center-mode bbox-center-ground` shifts the mesh so
   its XY center is at origin and its bottom touches z=0
4. **Platform placement** — The cloth is auto-shifted in +Z to rest on the
   collision platform (default top at z=0.26m)
5. **Robot trajectory** — `--robot-target-mode bbox-grasp` computes a grasp-and-lift
   trajectory from the cloth's world-space bounding box. `bbox-fold` adds a fold motion.
6. **Simulation** — VBD solver for cloth, Featherstone for the robot, with
   self-contact and cloth-body contact
7. **USD export** — `ViewerUSD` records all body transforms and cloth vertex
   positions as time-sampled USD attributes

## Key CLI Arguments

### Cloth mesh

| Argument | Default | Description |
|----------|---------|-------------|
| `--cloth-path` | built-in shirt | Path to `.obj` or `.usd` cloth mesh |
| `--cloth-scale` | 0.01 | Uniform scale factor (mesh units to meters) |
| `--cloth-scale-auto` | off | Auto-detect units from mesh bounding box |
| `--cloth-center-mode` | auto | Recentering: `none`, `bbox-center`, `bbox-center-ground` |
| `--cloth-pos X Y Z` | 0.0 0.70 0.28 | Initial cloth position in world [m] |
| `--cloth-yaw-deg` | 180.0 | Initial yaw rotation [degrees] |
| `--cloth-density` | 0.2 | Cloth areal density |

### Platform / table

| Argument | Default | Description |
|----------|---------|-------------|
| `--platform-enable / --no-platform-enable` | on | Add a collision box as table surface |
| `--platform-pos X Y Z` | 0.0 -0.45 0.18 | Platform center [m] |
| `--platform-size SX SY SZ` | 0.8 0.8 0.16 | Platform full size [m] |
| `--platform-top-z` | (computed) | Override platform top surface height [m] |
| `--cloth-place-on-platform` | on | Auto-shift cloth Z to rest on platform |

### Robot

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot-target-mode` | default | `default` (hardcoded poses), `bbox-grasp`, `bbox-fold` |
| `--robot-base-pos X Y Z` | -0.55 -0.85 -0.10 | Franka base position [m] |
| `--robot-start-delay` | 2.0 | Seconds before robot starts moving (cloth settling time) |
| `--trajectory-json` | none | Custom keypose trajectory file [N, 9] |

### Simulation

| Argument | Default | Description |
|----------|---------|-------------|
| `--sim-substeps` | 15 | Physics substeps per frame |
| `--iterations` | 5 | VBD solver iterations per substep |
| `--num-frames` | 3850 | Total simulation frames (at 60 fps) |
| `--device` | (warp default) | e.g. `cuda:0` |

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--viewer` | gl | Use `usd` for USDA file output |
| `--output-path` | (required for usd) | Output `.usda` file path |
| `--metadata-path` | `<output>.json` | Sidecar metadata JSON |

## Batch Runner Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-dir` | (required) | Root dataset directory |
| `--output-dir` | (required) | Output directory for USDA files |
| `--categories` | all | Category folder names to process |
| `--openusd-root` | (env var) | OpenUSD install path (alternative to OPENUSD_ROOT env var) |
| `--device` | cuda:0 | Warp device |
| `--num-frames` | 600 | Frames per garment |
| `--robot-target-mode` | bbox-grasp | `bbox-grasp` or `bbox-fold` |
| `--max-items` | all | Limit items (for testing) |
| `--start-index` | 0 | Resume from this index |
| `--skip-existing` | off | Skip garments with existing output |
| `--dry-run` | off | List items without running |

## Performance

| Metric | Value |
|--------|-------|
| Per-garment (120 frames) | ~43 seconds |
| Per-garment (600 frames) | ~3-4 minutes |
| 8,450 garments (600 frames) | ~14-24 hours on 1 GPU |

## Deploying to Another Machine

1. Clone the repo and run `uv sync --extra examples`
2. Install or build OpenUSD and set `OPENUSD_ROOT`
3. Ensure `pxr` is importable (via `PYTHONPATH` or installed in venv)
4. Copy the dataset to a local path
5. Run:
   ```bash
   uv run python scripts/run_cloth_dataset_batch.py ^
     --openusd-root /path/to/OpenUSD ^
     --dataset-dir /path/to/ClothDataset ^
     --output-dir /path/to/output ^
     --device cuda:0
   ```
