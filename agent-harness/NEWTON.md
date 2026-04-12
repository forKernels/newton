# Newton CLI Harness — SOP

## Software Overview

Newton is a GPU-accelerated physics simulation engine built on NVIDIA Warp.
It provides a Python API for building scenes (URDF/MJCF/USD), running simulations
with multiple solvers (VBD, XPBD, MuJoCo, Featherstone, etc.), and exporting
results (USD, JSON/CBOR recordings).

## Backend

Newton itself IS the backend — it's a Python library (`warp-lang` + `mujoco-warp`).
The CLI wraps Newton's Python API for headless, agent-driven simulation workflows.

## Data Model

- **Model**: Static simulation data (bodies, shapes, joints, particles, materials)
- **State**: Time-varying data (positions, velocities, forces)
- **Control**: Actuation inputs (joint forces, PD targets)
- **Contacts**: Collision results from broadphase + narrowphase pipeline

## CLI Command Groups

| Group     | Purpose                                              |
|-----------|------------------------------------------------------|
| `scene`   | Load URDF/MJCF/USD, inspect model, list bodies/joints |
| `sim`     | Run simulation with configurable solver/dt/frames    |
| `export`  | Export results to USD, JSON, or CBOR                 |
| `example` | Run built-in Newton examples                         |
| `mesh`    | Mesh preprocessing (stitch seams, check connectivity) |
| `info`    | Introspect loaded models (counts, shapes, materials) |

## File Formats

- **Input**: URDF (.urdf), MJCF (.xml), USD (.usd/.usda), meshes (.obj/.stl/.glb)
- **Output**: USD (.usd/.usda), JSON (.json), CBOR (.bin)

## Required Dependencies

- `warp-lang>=1.11.0` (NVIDIA Warp GPU framework)
- Newton itself (the library this CLI wraps)
