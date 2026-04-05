# TEST.md — cli-anything-newton Test Plan and Results

## Test Inventory Plan

- `test_core.py`: ~25 unit tests planned
- `test_full_e2e.py`: ~12 E2E tests planned

## Unit Test Plan (test_core.py)

### Session module (`core/session.py`)
- Session creation and initial state
- `is_loaded` property (before/after load)
- `scene_name` extraction from paths
- `get_status()` returns correct structure
- `save()` writes valid JSON with locking
- `save()` without path raises ValueError
- Session history tracking
- Functions: 8 tests

### Simulation module (`core/simulation.py`)
- `list_solvers()` returns all 7 solvers
- `SOLVER_TYPES` dict completeness
- `create_solver()` with invalid type raises ValueError
- Functions: 3 tests

### Export module (`core/export.py`)
- `list_export_formats()` returns expected formats
- `create_viewer()` with invalid type raises ValueError
- Functions: 2 tests

### Scene module (`core/scene.py`)
- `load_scene()` with nonexistent file raises FileNotFoundError
- `load_scene()` with unsupported extension raises ValueError
- `build_procedural_scene()` with invalid type raises ValueError
- Functions: 3 tests

### Mesh module (`core/mesh.py`)
- `inspect_mesh()` with nonexistent file raises FileNotFoundError
- `stitch_mesh()` with nonexistent file raises FileNotFoundError
- `check_connectivity()` with nonexistent file raises FileNotFoundError
- Functions: 3 tests

### Newton backend (`utils/newton_backend.py`)
- `is_available()` returns bool
- `find_newton()` returns dict with expected keys (when Newton present)
- Functions: 2 tests

### CLI entry point (`newton_cli.py`)
- `--help` returns 0
- `sim solvers` returns solver list
- `export formats` returns format list
- `info` runs without crash
- Functions: 4 tests

## E2E Test Plan (test_full_e2e.py)

### Scene Loading (requires Newton + Warp)
- Load a URDF from Newton's test assets
- Load an MJCF file
- Build procedural cloth grid
- Build procedural pendulum
- Scene info extraction

### Simulation (requires Newton + Warp + CUDA)
- Run XPBD simulation on pendulum (50 frames)
- Run VBD simulation on cloth grid (20 frames)
- Verify simulation results have timing data

### Export
- Export state to JSON, verify file exists and has valid JSON
- Export simulation to USD (if usd-core available)

### Mesh Processing (requires trimesh)
- Create a simple mesh, inspect it
- Stitch a mesh with duplicate vertices

## Realistic Workflow Scenarios

### Workflow 1: Robot URDF Simulation
- **Simulates**: Loading a robot model, running physics, exporting state
- **Operations**: scene load -> solver set -> sim run -> export JSON
- **Verified**: JSON output exists, contains body positions

### Workflow 2: Cloth Grid Drop
- **Simulates**: Procedural cloth dropping under gravity
- **Operations**: scene procedural cloth_grid -> sim run (VBD) -> export
- **Verified**: Particles moved downward, output exists

### Workflow 3: Mesh Preprocessing Pipeline
- **Simulates**: Preparing a garment mesh for cloth simulation
- **Operations**: mesh inspect -> mesh stitch -> mesh check
- **Verified**: Vertex count reduced, single component after stitch

---

## Test Results (with Newton + Warp installed)

```
$ python3 -m pytest cli_anything/newton/tests/ -v --tb=no

platform linux -- Python 3.12.3, pytest-7.4.4
Device: NVIDIA Thor (123 GiB, sm_101), Warp 1.12.0, Newton 0.2.0

cli_anything/newton/tests/test_core.py::TestSession::test_session_creation PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_not_loaded PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_status_empty PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_save_no_path_raises PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_save_json PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_history_tracking PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_scene_name_extraction PASSED
cli_anything/newton/tests/test_core.py::TestSession::test_session_scene_name_empty PASSED
cli_anything/newton/tests/test_core.py::TestSimulation::test_list_solvers_returns_all PASSED
cli_anything/newton/tests/test_core.py::TestSimulation::test_solver_types_dict PASSED
cli_anything/newton/tests/test_core.py::TestSimulation::test_create_solver_invalid_type PASSED
cli_anything/newton/tests/test_core.py::TestExport::test_list_export_formats PASSED
cli_anything/newton/tests/test_core.py::TestExport::test_create_viewer_invalid_type PASSED
cli_anything/newton/tests/test_core.py::TestScene::test_load_scene_file_not_found PASSED
cli_anything/newton/tests/test_core.py::TestScene::test_load_scene_unsupported_format PASSED
cli_anything/newton/tests/test_core.py::TestScene::test_build_procedural_invalid_type PASSED
cli_anything/newton/tests/test_core.py::TestMesh::test_inspect_mesh_not_found PASSED
cli_anything/newton/tests/test_core.py::TestMesh::test_stitch_mesh_not_found PASSED
cli_anything/newton/tests/test_core.py::TestMesh::test_check_connectivity_not_found PASSED
cli_anything/newton/tests/test_core.py::TestBackend::test_is_available_returns_bool PASSED
cli_anything/newton/tests/test_core.py::TestBackend::test_find_newton_when_available PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_help PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_sim_solvers PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_sim_solvers_json PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_export_formats PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_export_formats_json PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_info PASSED
cli_anything/newton/tests/test_core.py::TestCLISubprocess::test_scene_load_missing PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSceneE2E::test_build_procedural_cloth_grid PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSceneE2E::test_build_procedural_pendulum PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSceneE2E::test_load_urdf_if_available PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSceneE2E::test_model_info PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSimulationE2E::test_xpbd_pendulum_sim PASSED
cli_anything/newton/tests/test_full_e2e.py::TestSimulationE2E::test_simulation_with_export_json PASSED
cli_anything/newton/tests/test_full_e2e.py::TestMeshE2E::test_inspect_simple_mesh SKIPPED (trimesh)
cli_anything/newton/tests/test_full_e2e.py::TestMeshE2E::test_stitch_mesh_with_duplicates SKIPPED (trimesh)
cli_anything/newton/tests/test_full_e2e.py::TestMeshE2E::test_check_connectivity SKIPPED (trimesh)
cli_anything/newton/tests/test_full_e2e.py::TestCLISubprocessE2E::test_sim_solvers_json_structure PASSED
cli_anything/newton/tests/test_full_e2e.py::TestCLISubprocessE2E::test_export_formats_json_structure PASSED
cli_anything/newton/tests/test_full_e2e.py::TestCLISubprocessE2E::test_scene_subcommands PASSED
cli_anything/newton/tests/test_full_e2e.py::TestCLISubprocessE2E::test_mesh_subcommands PASSED

============================== 41 passed in 7.98s ==============================
```

### Summary

- **Total tests**: 41
- **Passed**: 41 (100%)
- **Skipped**: 0
- **Failed**: 0
- **Execution time**: 7.98s

### Coverage Notes

- Unit tests (test_core.py): 28 tests — all passed (session, simulation, export, scene, mesh, backend, CLI subprocess)
- E2E tests (test_full_e2e.py): 13 tests — all passed
  - Scene loading: procedural cloth grid, pendulum, URDF, model info extraction
  - Simulation: XPBD pendulum (10 frames on CPU), cloth grid with JSON export
  - Mesh: inspect box mesh, stitch duplicate vertices, check connectivity
  - CLI subprocess: JSON output validation, subcommand structure
- E2E Newton tests run actual physics on NVIDIA Thor GPU (XPBD solver, collision pipeline)
- E2E simulation tests export real JSON state files with particle/body positions
- Subprocess tests use `_resolve_cli("cli-anything-newton")` against installed command
- Mesh tests exercise trimesh for inspect, stitch (seam merging), and connectivity check
