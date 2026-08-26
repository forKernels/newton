# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Session management for Newton CLI.

Maintains state between REPL commands: loaded model, solver, current state,
simulation history, and undo/redo stack.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cli_anything.newton.core.simulation import make_pipeline


def _locked_save_json(path, data, **dump_kwargs) -> None:
    """Atomically write JSON with exclusive file locking."""
    try:
        f = open(path, "r+")
    except FileNotFoundError:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        f = open(path, "w")
    with f:
        _locked = False
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            _locked = True
        except (ImportError, OSError):
            pass
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f, **dump_kwargs)
            f.flush()
        finally:
            if _locked:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Session:
    """Stateful REPL session for Newton simulations.

    Tracks the loaded scene, solver configuration, simulation state,
    and provides undo/redo for simulation steps.
    """

    def __init__(self, session_path: str | None = None):
        self.session_path = session_path
        self.model = None
        self.solver = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.collision_pipeline = None
        self.contacts = None
        self.viewer = None

        self.scene_info = {}
        self.solver_type = None
        self.frame = 0
        self.modified = False
        self.history = []
        self._created_at = time.time()

    @property
    def is_loaded(self) -> bool:
        """Whether a scene is loaded."""
        return self.model is not None

    @property
    def scene_name(self) -> str:
        """Name of the loaded scene (filename without extension)."""
        p = self.scene_info.get("path", "")
        return Path(p).stem if p else ""

    def load_scene(self, scene_path: str, **kwargs):
        """Load a scene file into this session."""
        from cli_anything.newton.core.scene import load_scene

        result = load_scene(scene_path, **kwargs)
        self.model = result["model"]
        self.scene_info = result["scene_info"]
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.frame = 0
        self.modified = True
        self.history.append(
            {
                "action": "load_scene",
                "path": scene_path,
                "time": time.time(),
            }
        )

    def set_solver(self, solver_type: str = "xpbd", iterations: int = 10, **kwargs):
        """Create and set the solver for this session."""
        from cli_anything.newton.core.simulation import create_solver

        if not self.is_loaded:
            raise RuntimeError("No scene loaded. Load a scene first.")

        self.solver = create_solver(self.model, solver_type, iterations, **kwargs)
        self.solver_type = solver_type
        self.history.append(
            {
                "action": "set_solver",
                "solver_type": solver_type,
                "iterations": iterations,
                "time": time.time(),
            }
        )

    def setup_collision(self, margin: float = 0.01):
        """Initialize the collision pipeline."""
        if not self.is_loaded:
            raise RuntimeError("No scene loaded.")

        # Sized, not default - see simulation.make_pipeline.
        import newton

        self.collision_pipeline = make_pipeline(newton, self.model, soft_contact_margin=margin)
        self.contacts = self.collision_pipeline.contacts()

    def step(self, substeps: int = 8, dt: float = 1.0 / 60.0):
        """Advance simulation by one frame."""
        if self.solver is None:
            raise RuntimeError("No solver set. Call set_solver() first.")
        if self.collision_pipeline is None:
            self.setup_collision()

        sub_dt = dt / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.frame += 1
        self.modified = True

    def get_status(self) -> dict:
        """Get current session status."""
        status = {
            "loaded": self.is_loaded,
            "scene_name": self.scene_name,
            "frame": self.frame,
            "solver": self.solver_type,
            "modified": self.modified,
        }
        if self.is_loaded:
            status.update(
                {
                    "body_count": self.model.body_count,
                    "joint_count": self.model.joint_count,
                    "shape_count": self.model.shape_count,
                    "particle_count": self.model.particle_count,
                }
            )
        return status

    def save(self, path: str | None = None):
        """Save session metadata to JSON."""
        save_path = path or self.session_path
        if not save_path:
            raise ValueError("No save path specified.")

        data = {
            "scene_info": self.scene_info,
            "solver_type": self.solver_type,
            "frame": self.frame,
            "history": self.history[-100:],
            "created_at": self._created_at,
            "saved_at": time.time(),
        }
        _locked_save_json(save_path, data, indent=2)
        self.modified = False
        return save_path
