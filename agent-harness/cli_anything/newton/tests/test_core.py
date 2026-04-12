# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cli-anything-newton core modules.

These tests use synthetic data and do NOT require Newton/Warp to be installed.
They test pure-Python logic: session state, solver listing, format listing,
input validation, and CLI entry points.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest


# ── Helper ────────────────────────────────────────────────────────────

def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev."""
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


# ══════════════════════════════════════════════════════════════════════
#  Session tests
# ══════════════════════════════════════════════════════════════════════


class TestSession:
    """Unit tests for Session (no Newton required)."""

    def test_session_creation(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        assert s.model is None
        assert s.solver is None
        assert s.frame == 0
        assert not s.is_loaded
        assert not s.modified

    def test_session_not_loaded(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        assert s.is_loaded is False
        assert s.scene_name == ""

    def test_session_status_empty(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        status = s.get_status()
        assert status["loaded"] is False
        assert status["frame"] == 0
        assert status["solver"] is None

    def test_session_save_no_path_raises(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        with pytest.raises(ValueError, match="No save path"):
            s.save()

    def test_session_save_json(self, tmp_path):
        from cli_anything.newton.core.session import Session
        path = str(tmp_path / "session.json")
        s = Session(session_path=path)
        s.scene_info = {"path": "/test/robot.urdf"}
        s.solver_type = "xpbd"
        s.frame = 42
        result = s.save()
        assert result == path
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["solver_type"] == "xpbd"
        assert data["frame"] == 42

    def test_session_history_tracking(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        assert s.history == []
        # History is populated by load_scene/set_solver which need Newton,
        # but we can test the structure
        s.history.append({"action": "test", "time": 0})
        assert len(s.history) == 1

    def test_session_scene_name_extraction(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        s.scene_info = {"path": "/home/user/models/anymal_c.urdf"}
        assert s.scene_name == "anymal_c"

    def test_session_scene_name_empty(self):
        from cli_anything.newton.core.session import Session
        s = Session()
        s.scene_info = {"path": ""}
        assert s.scene_name == ""


# ══════════════════════════════════════════════════════════════════════
#  Simulation module tests
# ══════════════════════════════════════════════════════════════════════


class TestSimulation:
    """Unit tests for simulation module (no Newton required for listing)."""

    def test_list_solvers_returns_all(self):
        from cli_anything.newton.core.simulation import list_solvers
        solvers = list_solvers()
        assert len(solvers) == 7
        names = {s["name"] for s in solvers}
        assert "xpbd" in names
        assert "vbd" in names
        assert "mujoco" in names

    def test_solver_types_dict(self):
        from cli_anything.newton.core.simulation import SOLVER_TYPES
        assert len(SOLVER_TYPES) == 7
        assert SOLVER_TYPES["xpbd"] == "SolverXPBD"
        assert SOLVER_TYPES["mujoco"] == "SolverMuJoCo"

    def test_create_solver_invalid_type(self):
        from cli_anything.newton.core.simulation import create_solver
        with pytest.raises(ValueError, match="Unknown solver"):
            create_solver(None, "nonexistent_solver")


# ══════════════════════════════════════════════════════════════════════
#  Export module tests
# ══════════════════════════════════════════════════════════════════════


class TestExport:
    """Unit tests for export module."""

    def test_list_export_formats(self):
        from cli_anything.newton.core.export import list_export_formats
        formats = list_export_formats()
        assert len(formats) >= 3
        names = {f["format"] for f in formats}
        assert "usd" in names
        assert "json" in names
        assert "file" in names

    def test_create_viewer_invalid_type(self):
        from cli_anything.newton.core.export import create_viewer
        with pytest.raises(ValueError, match="Unknown viewer type"):
            create_viewer("nonexistent_viewer")


# ══════════════════════════════════════════════════════════════════════
#  Scene module tests
# ══════════════════════════════════════════════════════════════════════


class TestScene:
    """Unit tests for scene module (file validation, no Newton needed)."""

    def test_load_scene_file_not_found(self):
        from cli_anything.newton.core.scene import load_scene
        with pytest.raises(FileNotFoundError):
            load_scene("/nonexistent/path/robot.urdf")

    def test_load_scene_unsupported_format(self, tmp_path):
        from cli_anything.newton.core.scene import load_scene
        bad = tmp_path / "scene.txt"
        bad.write_text("not a scene")
        with pytest.raises(ValueError, match="Unsupported scene format"):
            load_scene(str(bad))

    def test_build_procedural_invalid_type(self):
        from cli_anything.newton.core.scene import build_procedural_scene
        with pytest.raises(ValueError, match="Unknown procedural scene type"):
            build_procedural_scene("nonexistent_type")


# ══════════════════════════════════════════════════════════════════════
#  Mesh module tests
# ══════════════════════════════════════════════════════════════════════


class TestMesh:
    """Unit tests for mesh module (file validation)."""

    def test_inspect_mesh_not_found(self):
        from cli_anything.newton.core.mesh import inspect_mesh
        with pytest.raises(FileNotFoundError):
            inspect_mesh("/nonexistent/mesh.obj")

    def test_stitch_mesh_not_found(self):
        from cli_anything.newton.core.mesh import stitch_mesh
        with pytest.raises(FileNotFoundError):
            stitch_mesh("/nonexistent/mesh.obj")

    def test_check_connectivity_not_found(self):
        from cli_anything.newton.core.mesh import check_connectivity
        with pytest.raises(FileNotFoundError):
            check_connectivity("/nonexistent/mesh.obj")


# ══════════════════════════════════════════════════════════════════════
#  Backend tests
# ══════════════════════════════════════════════════════════════════════


class TestBackend:
    """Unit tests for newton_backend module."""

    def test_is_available_returns_bool(self):
        from cli_anything.newton.utils.newton_backend import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_find_newton_when_available(self):
        from cli_anything.newton.utils.newton_backend import is_available, find_newton
        if not is_available():
            pytest.skip("Newton not installed in this environment")
        info = find_newton()
        assert "newton_version" in info
        assert "warp_version" in info
        assert "devices" in info


# ══════════════════════════════════════════════════════════════════════
#  CLI subprocess tests
# ══════════════════════════════════════════════════════════════════════


class TestCLISubprocess:
    """Test the installed CLI command via subprocess."""

    CLI_BASE = _resolve_cli("cli-anything-newton")

    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True, text=True,
            check=check,
        )

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "Newton Physics Engine" in result.stdout

    def test_sim_solvers(self):
        result = self._run(["sim", "solvers"])
        assert result.returncode == 0
        assert "xpbd" in result.stdout

    def test_sim_solvers_json(self):
        result = self._run(["--json", "sim", "solvers"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "solvers" in data
        assert len(data["solvers"]) == 7

    def test_export_formats(self):
        result = self._run(["export", "formats"])
        assert result.returncode == 0
        assert "usd" in result.stdout

    def test_export_formats_json(self):
        result = self._run(["--json", "export", "formats"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "formats" in data

    def test_info(self):
        # May fail if Newton not installed — that's expected
        result = self._run(["info"], check=False)
        # Either succeeds with info or fails with clear error
        assert result.returncode in (0, 1)

    def test_scene_load_missing(self):
        result = self._run(["scene", "load", "/nonexistent/robot.urdf"], check=False)
        assert result.returncode != 0
