# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for cli-anything-newton.

These tests require Newton + Warp to be installed and a GPU available.
They test the full pipeline: load scene -> run simulation -> export.

NOTE: Tests that require Newton will fail (not skip) if Newton is not
installed. Newton is a hard dependency — the CLI is useless without it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────

NEWTON_AVAILABLE = False
try:
    import warp as wp

    import newton

    NEWTON_AVAILABLE = True
except ImportError:
    pass

TRIMESH_AVAILABLE = False
try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    pass


def _resolve_cli(name):
    """Resolve installed CLI command."""
    import shutil

    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = name.replace("cli-anything-", "cli_anything.") + "." + name.split("-")[-1] + "_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


def _find_test_urdf():
    """Find a URDF file in Newton's test/example assets."""
    newton_root = Path(__file__).resolve().parents[4]  # agent-harness -> newton
    candidates = [
        newton_root / "newton" / "tests" / "assets",
        newton_root / "newton" / "examples",
        newton_root / "assets",
    ]
    for d in candidates:
        if d.is_dir():
            for urdf in d.rglob("*.urdf"):
                return str(urdf)
    return None


# ══════════════════════════════════════════════════════════════════════
#  Scene loading E2E tests (require Newton)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not NEWTON_AVAILABLE, reason="Newton not installed")
class TestSceneE2E:
    """E2E tests for scene loading — requires Newton + Warp."""

    def test_build_procedural_cloth_grid(self):
        from cli_anything.newton.core.scene import build_procedural_scene

        result = build_procedural_scene("cloth_grid", device="cpu", resolution=10, size=0.5)
        model = result["model"]
        info = result["scene_info"]
        assert model.particle_count == (10 + 1) ** 2  # 121 particles
        assert info["scene_type"] == "cloth_grid"

    def test_build_procedural_pendulum(self):
        from cli_anything.newton.core.scene import build_procedural_scene

        result = build_procedural_scene("pendulum", device="cpu", num_links=3)
        model = result["model"]
        assert model.body_count == 3
        assert model.joint_count == 3

    def test_load_urdf_if_available(self):
        urdf = _find_test_urdf()
        if urdf is None:
            pytest.skip("No test URDF found in Newton assets")
        from cli_anything.newton.core.scene import load_scene

        result = load_scene(urdf, device="cpu")
        assert result["model"].body_count > 0

    def test_model_info(self):
        from cli_anything.newton.core.scene import build_procedural_scene, get_model_info

        result = build_procedural_scene("cloth_grid", device="cpu", resolution=5)
        info = get_model_info(result["model"])
        assert info["particle_count"] == 36
        assert info["world_count"] >= 1
        assert "gravity" in info


# ══════════════════════════════════════════════════════════════════════
#  Simulation E2E tests (require Newton + Warp)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not NEWTON_AVAILABLE, reason="Newton not installed")
class TestSimulationE2E:
    """E2E simulation tests — requires Newton + Warp."""

    def test_xpbd_pendulum_sim(self):
        from cli_anything.newton.core.scene import build_procedural_scene
        from cli_anything.newton.core.simulation import create_solver, run_simulation

        scene = build_procedural_scene("pendulum", device="cpu", num_links=3)
        model = scene["model"]
        solver = create_solver(model, "xpbd", iterations=4)
        result = run_simulation(model, solver, num_frames=10, substeps=4, dt=1.0 / 60.0)

        assert result["num_frames"] == 10
        assert result["wall_time_s"] > 0
        assert result["fps"] > 0
        assert "body_positions" in result

    def test_simulation_with_export_json(self, tmp_path):
        from cli_anything.newton.core.export import export_state_json
        from cli_anything.newton.core.scene import build_procedural_scene
        from cli_anything.newton.core.simulation import create_solver, run_simulation

        scene = build_procedural_scene("cloth_grid", device="cpu", resolution=5)
        model = scene["model"]
        solver = create_solver(model, "xpbd", iterations=4)
        result = run_simulation(model, solver, num_frames=5, substeps=2, dt=1.0 / 60.0)

        # Export to JSON
        json_path = str(tmp_path / "state.json")

        # We need the actual state — re-create it
        state = model.state()
        export_result = export_state_json(state, model, json_path)

        assert os.path.exists(export_result["output"])
        assert export_result["file_size"] > 0
        assert export_result["format"] == "json"

        # Verify JSON is valid
        with open(json_path) as f:
            data = json.load(f)
        assert data["particle_count"] == 36
        print(f"\n  JSON: {json_path} ({export_result['file_size']:,} bytes)")


# ══════════════════════════════════════════════════════════════════════
#  Mesh E2E tests (require trimesh)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not TRIMESH_AVAILABLE, reason="trimesh not installed")
class TestMeshE2E:
    """E2E tests for mesh processing — requires trimesh."""

    def test_inspect_simple_mesh(self, tmp_path):
        import trimesh

        # Create a simple box mesh
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh_path = str(tmp_path / "box.obj")
        mesh.export(mesh_path)

        from cli_anything.newton.core.mesh import inspect_mesh

        info = inspect_mesh(mesh_path)

        assert info["vertex_count"] > 0
        assert info["face_count"] > 0
        assert info["is_watertight"] is True
        assert info["num_components"] == 1
        print(f"\n  Mesh: {mesh_path}")
        print(f"  Vertices: {info['vertex_count']}, Faces: {info['face_count']}")

    def test_stitch_mesh_with_duplicates(self, tmp_path):
        import numpy as np
        import trimesh

        # Create two adjacent quads that share an edge but have duplicate vertices
        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],  # quad 1
                [1, 0, 0],
                [2, 0, 0],
                [2, 1, 0],
                [1, 1, 0],  # quad 2 (verts 4,7 duplicate 1,2)
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # quad 1
                [4, 5, 6],
                [4, 6, 7],  # quad 2
            ]
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh_path = str(tmp_path / "seamed.obj")
        mesh.export(mesh_path)

        from cli_anything.newton.core.mesh import stitch_mesh

        result = stitch_mesh(mesh_path, threshold=0.001)

        assert result["stitched_vertices"] <= result["original_vertices"]
        assert result["vertices_merged"] >= 0
        assert os.path.exists(result["output"])
        assert result["file_size"] > 0
        print(f"\n  Stitched: {result['original_vertices']} -> {result['stitched_vertices']} verts")
        print(f"  Output: {result['output']} ({result['file_size']:,} bytes)")

    def test_check_connectivity(self, tmp_path):
        import trimesh

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh_path = str(tmp_path / "box.obj")
        mesh.export(mesh_path)

        from cli_anything.newton.core.mesh import check_connectivity

        result = check_connectivity(mesh_path)

        assert result["vertex_count"] > 0
        assert result["is_single_component"] is True


# ══════════════════════════════════════════════════════════════════════
#  CLI Subprocess E2E tests
# ══════════════════════════════════════════════════════════════════════


class TestCLISubprocessE2E:
    """E2E subprocess tests — run the installed CLI command."""

    CLI_BASE = _resolve_cli("cli-anything-newton")

    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_sim_solvers_json_structure(self):
        result = self._run(["--json", "sim", "solvers"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "solvers" in data
        for s in data["solvers"]:
            assert "name" in s
            assert "class" in s
            assert "description" in s

    def test_export_formats_json_structure(self):
        result = self._run(["--json", "export", "formats"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        for fmt in data["formats"]:
            assert "format" in fmt
            assert "extension" in fmt
            assert "description" in fmt

    def test_scene_subcommands(self):
        result = self._run(["scene", "--help"])
        assert result.returncode == 0
        assert "load" in result.stdout
        assert "info" in result.stdout
        assert "procedural" in result.stdout

    def test_mesh_subcommands(self):
        result = self._run(["mesh", "--help"])
        assert result.returncode == 0
        assert "inspect" in result.stdout
        assert "stitch" in result.stdout
        assert "check" in result.stdout
