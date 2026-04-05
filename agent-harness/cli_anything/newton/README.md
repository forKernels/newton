# cli-anything-newton

CLI harness for the **Newton** GPU-accelerated physics simulation engine.
Provides headless, agent-driven access to Newton's simulation capabilities.

## Prerequisites

- **Newton** (the physics engine this CLI wraps):
  ```bash
  cd /path/to/newton && pip install -e ".[examples]"
  # or: uv sync --extra examples
  ```
- **NVIDIA Warp** (GPU acceleration):
  ```bash
  pip install warp-lang>=1.11.0
  ```
- **NVIDIA GPU** with CUDA support (recommended but CPU fallback available)

## Installation

```bash
cd /path/to/newton/agent-harness
pip install -e .
```

Verify installation:
```bash
cli-anything-newton --help
cli-anything-newton info
```

## Usage

### One-shot commands

```bash
# System info
cli-anything-newton info
cli-anything-newton --json info

# Load and inspect a scene
cli-anything-newton scene load path/to/robot.urdf
cli-anything-newton scene info path/to/scene.xml
cli-anything-newton --json scene info path/to/scene.usd

# Run a simulation
cli-anything-newton sim run path/to/robot.urdf --solver xpbd --frames 200 -o output.usd
cli-anything-newton sim run scene.xml --solver mujoco --frames 500 --output-format json -o result.json

# List solvers
cli-anything-newton sim solvers

# Mesh preprocessing
cli-anything-newton mesh inspect garment.obj
cli-anything-newton mesh stitch garment.obj --threshold 0.0005
cli-anything-newton mesh check garment.obj

# Run built-in examples
cli-anything-newton example list
cli-anything-newton example run basic_pendulum --frames 200

# Procedural scenes
cli-anything-newton scene procedural cloth_grid --resolution 40
cli-anything-newton scene procedural pendulum --num-links 8
```

### Interactive REPL

```bash
cli-anything-newton  # enters REPL by default
```

REPL commands:
- `load <path>` — Load a scene file
- `solver <type> [iterations]` — Set physics solver
- `step [N]` — Advance simulation by N frames
- `run <frames>` — Run full simulation
- `export <path>` — Export state to JSON
- `status` — Show session status
- `info` — Show backend info
- `help` — List all commands

### JSON output (for agents)

Add `--json` before any subcommand for machine-readable output:

```bash
cli-anything-newton --json sim run scene.urdf --solver xpbd --frames 100
cli-anything-newton --json mesh inspect garment.obj
cli-anything-newton --json info
```

## Running tests

```bash
cd /path/to/newton/agent-harness
python -m pytest cli_anything/newton/tests/ -v -s
```
