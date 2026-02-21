# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declarative scene configuration dataclasses for the orchestrator pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ElementType(Enum):
    """Type of scene element."""

    CLOTH_USD = "cloth_usd"
    RIGID_USD = "rigid_usd"  # future
    ARTICULATED_URDF = "articulated_urdf"  # future


@dataclass
class ClothParams:
    """Physics parameters for cloth simulation.

    Defaults match example_cloth_franka.py.
    """

    density: float = 0.2
    """Cloth density [kg/m^2]."""
    tri_ke: float = 1e2
    """Triangle stretch stiffness [Pa]."""
    tri_ka: float = 1e2
    """Triangle area stiffness [Pa]."""
    tri_kd: float = 1.5e-6
    """Triangle damping [Pa*s]."""
    bending_ke: float = 1e-4
    """Edge bending stiffness [N*m]."""
    bending_kd: float = 1e-3
    """Edge bending damping [N*m*s]."""
    particle_radius: float = 0.008
    """Cloth particle radius for collision [m]."""


@dataclass
class ElementConfig:
    """A scene element (garment, rigid object, etc.)."""

    element_type: ElementType = ElementType.CLOTH_USD
    asset_path: str = ""
    """Path to the USD file on disk."""
    prim_path: str = ""
    """USD prim path within the file (e.g. '/root/shirt')."""
    position: tuple[float, float, float] = (0.0, 0.70, 0.28)
    """World position [m]."""
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Axis for rotation."""
    rotation_angle: float = 3.141592653589793
    """Rotation angle [rad]."""
    scale: float = 0.01
    """Uniform scale (e.g. 0.01 for cm → m)."""
    cloth_params: ClothParams = field(default_factory=ClothParams)
    """Physics params (only used for CLOTH_USD elements)."""


@dataclass
class FrankaConfig:
    """Franka Panda robot configuration."""

    position: tuple[float, float, float] = (-0.5, -0.5, -0.1)
    """Robot base position [m]."""
    initial_joint_q: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307])
    """Initial joint angles [rad] for the first 6 joints."""
    endeffector_offset: tuple[float, float, float] = (0.0, 0.0, 0.22)
    """End-effector offset from the last link [m]."""
    key_poses: list[list[float]] = field(default_factory=list)
    """Waypoints: [duration, x, y, z, qw, qx, qy, qz, gripper].
    If empty, uses the default cloth_franka folding trajectory."""


@dataclass
class EnvironmentConfig:
    """Static environment configuration."""

    table_position: tuple[float, float, float] = (0.0, -0.5, 0.1)
    """Table center position [m]."""
    table_half_extents: tuple[float, float, float] = (0.4, 0.4, 0.1)
    """Table half-extents (hx, hy, hz) [m]."""
    add_ground_plane: bool = True


@dataclass
class SimConfig:
    """Simulation parameters."""

    fps: int = 60
    """Frames per second for output."""
    num_frames: int = 500
    """Total number of output frames."""
    sim_substeps: int = 15
    """Substeps per frame."""
    solver_iterations: int = 5
    """VBD solver iterations per substep."""
    soft_contact_ke: float = 100.0
    """Soft contact elastic stiffness [N/m]."""
    soft_contact_kd: float = 2e-3
    """Soft contact damping [N*s/m]."""
    self_contact_friction: float = 0.25
    """Particle self-contact friction coefficient."""
    robot_friction: float = 1.0
    """Robot shape friction coefficient."""
    table_friction: float = 1.0
    """Table shape friction coefficient."""
    cloth_body_contact_margin: float = 0.01
    """Margin for cloth-body collision detection [m]."""
    particle_self_contact_radius: float = 0.002
    """Particle self-contact radius [m]."""
    particle_self_contact_margin: float = 0.003
    """Particle self-contact margin [m]."""


@dataclass
class SceneConfig:
    """Top-level scene configuration."""

    franka: FrankaConfig = field(default_factory=FrankaConfig)
    elements: list[ElementConfig] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    output_path: str = "output.usda"

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        # Convert ElementType enum to string
        for elem in d.get("elements", []):
            if isinstance(elem.get("element_type"), ElementType):
                elem["element_type"] = elem["element_type"].value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SceneConfig:
        """Deserialize from a dict."""
        franka = FrankaConfig(**d.get("franka", {}))
        environment = EnvironmentConfig(**d.get("environment", {}))
        sim = SimConfig(**d.get("sim", {}))

        elements = []
        for elem_d in d.get("elements", []):
            elem_d = dict(elem_d)  # noqa: PLW2901
            if "element_type" in elem_d:
                elem_d["element_type"] = ElementType(elem_d["element_type"])
            if "cloth_params" in elem_d and isinstance(elem_d["cloth_params"], dict):
                elem_d["cloth_params"] = ClothParams(**elem_d["cloth_params"])
            # Convert tuple fields from lists
            for tfield in ("position", "rotation_axis"):
                if tfield in elem_d and isinstance(elem_d[tfield], list):
                    elem_d[tfield] = tuple(elem_d[tfield])
            elements.append(ElementConfig(**elem_d))

        return cls(
            franka=franka,
            elements=elements,
            environment=environment,
            sim=sim,
            output_path=d.get("output_path", "output.usda"),
        )

    def to_json(self, path: str | Path) -> None:
        """Write config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> SceneConfig:
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
