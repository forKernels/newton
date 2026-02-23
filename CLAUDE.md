# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## What is Newton?

Newton is a GPU-accelerated physics simulation engine built on NVIDIA Warp, targeting robotics and simulation research. It extends Warp's deprecated `warp.sim` module and integrates MuJoCo Warp as its primary backend. The sole required dependency is `warp-lang>=1.11.0`.

## Common Commands

```bash
# Setup
uv sync --extra examples        # full environment with examples + visualization
uv sync --extra dev              # development environment with test dependencies

# Run examples
uv run -m newton.examples                      # list all available examples
uv run -m newton.examples basic_pendulum        # run a specific example
uv run -m newton.examples robot_anymal_c_walk --device cuda:0  # run on specific GPU

# Run tests
uv run --extra dev -m newton.tests              # run all tests
uv run --extra dev -m newton.tests.test_examples -k test_basic.example_basic_shapes  # single test
uv run --extra dev --extra torch-cu12 -m newton.tests  # include PyTorch-dependent tests

# Lint/format
uvx pre-commit run -a            # run all pre-commit hooks (ruff lint+format, typos, uv-lock)
uvx pre-commit install           # install hooks for automatic checking on commit

# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!   # Unix
uvx --with virtualenv asv run --launch-method spawn main^^!  # Windows CMD

# Update API docs after adding public symbols
uv run docs/generate_api.py
```

## Architecture Overview

### Public API boundary

Users interact only through top-level modules — never `newton._src`:

| Module | Exposes |
|---|---|
| `newton` | `Model`, `ModelBuilder`, `State`, `Control`, `Contacts`, `CollisionPipeline`, `eval_fk`, `eval_ik`, enums (`JointType`, `GeoType`, `ActuatorMode`) |
| `newton.solvers` | `SolverXPBD`, `SolverFeatherstone`, `SolverMuJoCo`, `SolverSemiImplicit`, `SolverStyle3D`, `SolverVBD`, `SolverImplicitMPM` |
| `newton.viewer` | `ViewerGL`, `ViewerUSD`, `ViewerRerun`, `ViewerViser`, `ViewerNull`, `ViewerFile` |
| `newton.sensors` | `SensorContact`, `SensorIMU`, `SensorRaycast`, `SensorFrameTransform`, `SensorTiledCamera` |
| `newton.geometry` | Broad-phase algorithms, primitive collision functions, terrain generators, SDF utilities |
| `newton.ik` | `IKSolver`, `IKOptimizer`, `IKObjective`, `IKSampler` |
| `newton.selection` | `ArticulationView` for batch multi-world operations |
| `newton.utils` | Mesh creation, spatial math, asset download, benchmarking |

### Internal structure (`newton/_src/`)

- **`core/`** — Foundation types (`Vec3`, `Quat`, `Transform`), spatial math utilities
- **`sim/`** — Core simulation abstractions:
  - `model.py` — `Model`: all static simulation data (bodies, shapes, joints, springs, particles). Supports multi-world via `*_world` arrays. Has custom attribute system (`AttributeAssignment`, `AttributeFrequency`)
  - `builder.py` — `ModelBuilder`: scene construction API (`add_body()`, `add_joint_revolute()`, `add_shape_box()`, `parse_urdf()`, `parse_mjcf()`, `parse_usd()`). `finalize()` produces GPU-ready `Model`
  - `state.py` — `State`: time-varying data (`particle_q/qd/f`, `body_q/qd/f`, `joint_q/qd`)
  - `control.py` — `Control`: control inputs (`joint_f`, targets, activations)
  - `contacts.py` — `Contacts`: collision contact buffers
  - `collide.py` / `collide_unified.py` — `CollisionPipeline` / `CollisionPipelineUnified` (broadphase + narrowphase)
  - `articulation.py` — Forward/inverse kinematics (`eval_fk`, `eval_ik`)
  - `joints.py` — Joint types and actuator modes
- **`geometry/`** — Collision detection subsystem (broad phase NxN/SAP, narrow phase primitives, GJK/MPR, SDF, contact reduction, hydroelastic contacts)
- **`solvers/`** — Seven solver implementations, each in its own subdirectory, all inheriting `SolverBase` from `solver.py`
- **`sensors/`** — Sensor implementations plus `warp_raytrace/` GPU ray tracing subsystem
- **`viewer/`** — Viewer hierarchy: `ViewerBase` → GL/USD/Rerun/Viser/Null/File backends
- **`utils/`** — URDF/MJCF/USD importers, mesh utilities, benchmarking, asset download
- **`usd/`** — USD schema resolution

### Simulation pipeline

```
1. BUILD:    ModelBuilder → add bodies/joints/shapes or parse_urdf/mjcf/usd → finalize() → Model
2. INIT:     Model → state(), control(), eval_fk()
3. LOOP:     state.clear_forces() → model.collide(state) → solver.step(in, out, ctrl, contacts, dt) → swap states
4. RENDER:   viewer.begin_frame() → viewer.log_state(state) → viewer.end_frame()
```

CUDA graph capture is supported for zero-overhead simulation loops using `wp.ScopedCapture`.

### Solver overview

| Solver | Coordinates | Method | Primary use |
|---|---|---|---|
| `SolverMuJoCo` | Generalized | Euler/RK4/implicit | Rigid-body robotics (primary backend) |
| `SolverFeatherstone` | Generalized | CRBA + semi-implicit Euler | Articulated rigid bodies |
| `SolverXPBD` | Maximal | Position-based dynamics | Rigid + soft bodies |
| `SolverSemiImplicit` | Maximal | Semi-implicit Euler | Simple explicit integration |
| `SolverStyle3D` | Implicit | BVH self-collision | Cloth simulation |
| `SolverVBD` | Implicit | Vertex Block Descent | Cloth + rigid coupling |
| `SolverImplicitMPM` | Implicit | Material Point Method | Granular/fluid materials |

### Example pattern

All examples under `newton/examples/` follow this class structure:

```python
class Example:
    def __init__(self, viewer, args=None):
        # Build model, create solver, states, control, eval_fk, set viewer
    def step(self):
        # Per-frame: run simulate() or launch captured CUDA graph
    def simulate(self):
        # Per-substep: clear_forces → collide → solver.step → swap states
    def render(self):
        # Update viewer with current state
    def test_final(self):       # REQUIRED — assert simulation correctness
    def test_post_step(self):   # OPTIONAL — assert after each step
```

Run via: `uv run -m newton.examples <name>`. CLI args: `--viewer` (gl/usd/rerun/null), `--device`, `--num-frames`, `--output-path`.

### Testing

Tests live in `newton/tests/` (~69 test files). The custom runner at `newton/tests/__main__.py` uses parallel unittest execution. `test_examples.py` dynamically generates tests that run each example as a subprocess with `--test --viewer null`.

### Linting and formatting

Pre-commit hooks run Ruff (lint + format, line length 120, Google-style docstrings), uv-lock sync, and typos spellchecker. Configuration is in `pyproject.toml` under `[tool.ruff]`.
