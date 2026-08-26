# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene loading and model building for Newton CLI.

Wraps Newton's ModelBuilder to load URDF/MJCF/USD scenes and build
simulation-ready Models from the command line.
"""

from __future__ import annotations

from pathlib import Path

_real_newton = None


def _ensure_newton():
    """Import the real Newton physics engine.

    Handles the case where ``cli_anything.newton`` shadows the real
    ``newton`` package (common when pytest runs from agent-harness/).
    Stores the real module in a private global to avoid repeated lookups.
    """
    global _real_newton
    if _real_newton is not None:
        return _real_newton

    import sys

    # Fast path: already loaded correctly in sys.modules
    mod = sys.modules.get("newton")
    if mod is not None and hasattr(mod, "ModelBuilder"):
        _real_newton = mod
        return mod

    # The bare "import newton" may resolve to cli_anything.newton (our package).
    # Use newton._src as a fingerprint to find the real one.
    try:
        import newton as _n
        import newton._src.sim.builder as _builder_mod

        if hasattr(_n, "ModelBuilder"):
            _real_newton = _n
            return _n
    except (ImportError, AttributeError):
        pass

    # Last resort: find the real newton package directory on sys.path
    # and patch sys.modules so relative imports (from ._src) work.
    import importlib.util
    import os

    for entry in sys.path:
        if not entry or not os.path.isdir(entry):
            continue
        candidate = os.path.join(entry, "newton", "__init__.py")
        if os.path.isfile(candidate) and "cli_anything" not in candidate:
            newton_dir = os.path.join(entry, "newton")
            spec = importlib.util.spec_from_file_location(
                "newton",
                candidate,
                submodule_search_locations=[newton_dir],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # Must register BEFORE exec so relative imports work
                sys.modules["newton"] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    sys.modules.pop("newton", None)
                    continue
                if hasattr(mod, "ModelBuilder"):
                    _real_newton = mod
                    return mod
                else:
                    sys.modules.pop("newton", None)

    raise RuntimeError(
        "Newton is not installed. Install it with:\n"
        "  pip install newton-sim\n"
        "Or from source:\n"
        "  cd /path/to/newton && pip install -e ."
    )


def _axis(newton, name):
    """A string like "y" as newton's Axis enum.

    `builder.up_axis` is an Axis, not a string, and assigning the string "Y"
    to it does not raise - it fails much later and somewhere else, inside
    newton's own builder:

        AttributeError: 'str' object has no attribute 'to_vector'

    because the builder eventually calls `self.up_axis.to_vector()`. The
    traceback points at newton and the cause is here, which is why this went
    unnoticed: it looked like engine API drift.

    Passed to the CONSTRUCTOR rather than assigned afterwards. up_axis
    participates in how the builder derives gravity and shape orientation, so
    setting it post-hoc is a second chance to be inconsistent - and the old
    code only set it for "y" at all, leaving "z" to rely on a default it never
    stated.
    """
    axis = getattr(newton, "Axis", None)
    if axis is None:  # a build without the enum
        return str(name).upper()
    from_any = getattr(axis, "from_any", None)
    if from_any is not None:
        return from_any(str(name).upper())
    return getattr(axis, str(name).upper(), axis.Z)


def load_scene(
    scene_path: str,
    device: str = "cuda:0",
    collapse_fixed_joints: bool = True,
    enable_self_collisions: bool = False,
    num_worlds: int = 1,
    spacing: float = 2.0,
    up_axis: str = "y",
) -> dict:
    """Load a scene file (URDF/MJCF/USD) and build a Newton Model.

    Args:
        scene_path: Path to scene file (.urdf, .xml/.mjcf, .usd/.usda).
        device: Warp device string.
        collapse_fixed_joints: Merge fixed joints for efficiency.
        enable_self_collisions: Enable particle self-collision.
        num_worlds: Number of independent simulation worlds.
        spacing: Spacing between replicated worlds.
        up_axis: Up axis for the scene ("y" or "z").

    Returns:
        Dict with keys: model, builder, scene_info.
    """
    path = Path(scene_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scene file not found: {path}")

    ext = path.suffix.lower()
    if ext not in (".urdf", ".xml", ".mjcf", ".usd", ".usda", ".usdc"):
        raise ValueError(f"Unsupported scene format: {ext}. Use .urdf, .xml, .usd, or .usda")

    newton = _ensure_newton()

    builder = newton.ModelBuilder(up_axis=_axis(newton, up_axis))

    scene_info = {
        "path": str(path),
        "format": ext,
        "num_worlds": num_worlds,
        "device": device,
    }

    if ext == ".urdf":
        builder.add_urdf(
            str(path),
            collapse_fixed_joints=collapse_fixed_joints,
            enable_self_collisions=enable_self_collisions,
        )
        scene_info["parser"] = "urdf"
    elif ext in (".xml", ".mjcf"):
        builder.add_mjcf(
            str(path),
            collapse_fixed_joints=collapse_fixed_joints,
            enable_self_collisions=enable_self_collisions,
        )
        scene_info["parser"] = "mjcf"
    else:  # .usd, .usda, .usdc — already validated above
        builder.add_usd(
            str(path),
            collapse_fixed_joints=collapse_fixed_joints,
            enable_self_collisions=enable_self_collisions,
        )
        scene_info["parser"] = "usd"

    # Replicate for multi-world if requested
    if num_worlds > 1:
        base_builder = builder
        builder = newton.ModelBuilder()
        builder.replicate(base_builder, count=num_worlds, spacing=spacing)

    model = builder.finalize(device=device)

    scene_info.update(
        {
            "body_count": model.body_count,
            "joint_count": model.joint_count,
            "shape_count": model.shape_count,
            "particle_count": model.particle_count,
            "spring_count": model.spring_count,
            "tri_count": model.tri_count,
        }
    )

    return {
        "model": model,
        "builder": builder,
        "scene_info": scene_info,
    }


def get_model_info(model) -> dict:
    """Extract detailed information from a Newton Model.

    Args:
        model: A finalized Newton Model.

    Returns:
        Dict with model statistics and structure.
    """

    info = {
        "body_count": model.body_count,
        "joint_count": model.joint_count,
        "shape_count": model.shape_count,
        "particle_count": model.particle_count,
        "spring_count": model.spring_count,
        "tri_count": model.tri_count,
        "tet_count": model.tet_count,
        "edge_count": model.edge_count,
        "world_count": model.world_count,
        "gravity": model.gravity.numpy().tolist() if hasattr(model.gravity, "numpy") else list(model.gravity),
    }

    # Shape type distribution
    if model.shape_count > 0:
        try:
            shape_types = model.shape_type.numpy()
            type_names = {
                0: "sphere",
                1: "box",
                2: "capsule",
                3: "cylinder",
                4: "cone",
                5: "mesh",
                6: "heightfield",
                7: "plane",
                8: "sdf",
            }
            counts = {}
            for t in shape_types:
                name = type_names.get(int(t), f"unknown({t})")
                counts[name] = counts.get(name, 0) + 1
            info["shape_types"] = counts
        except Exception:
            info["shape_types"] = {}

    return info


def build_procedural_scene(
    scene_type: str = "ground",
    device: str = "cuda:0",
    **kwargs,
) -> dict:
    """Build a procedural scene without loading a file.

    Args:
        scene_type: Type of scene to build ("ground", "cloth_grid", "pendulum").
        device: Warp device string.

    Returns:
        Dict with keys: model, builder, scene_info.
    """
    valid_types = ("ground", "cloth_grid", "pendulum")
    if scene_type not in valid_types:
        raise ValueError(f"Unknown procedural scene type: {scene_type}")

    newton = _ensure_newton()

    builder = newton.ModelBuilder()

    scene_info = {
        "path": None,
        "format": "procedural",
        "scene_type": scene_type,
        "device": device,
    }

    if scene_type == "ground":
        builder.add_body()
        builder.add_shape_plane(body=-1)

    elif scene_type == "cloth_grid":
        import warp as wp

        res = kwargs.get("resolution", 30)
        size = kwargs.get("size", 1.0)
        height = kwargs.get("height", 2.0)

        cell = size / res
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, height, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=res,
            dim_y=res,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
        )

        scene_info["resolution"] = res
        scene_info["particle_count_expected"] = (res + 1) ** 2

    elif scene_type == "pendulum":
        import warp as wp

        num_links = kwargs.get("num_links", 5)
        hx = kwargs.get("link_length", 0.5)
        hy = 0.1
        hz = 0.1

        links = []
        joints = []
        for i in range(num_links):
            link = builder.add_link()
            links.append(link)
            builder.add_shape_box(link, hx=hx, hy=hy, hz=hz)

            if i == 0:
                j = builder.add_joint_revolute(
                    parent=-1,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(p=wp.vec3(0.0, 2.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )
            else:
                j = builder.add_joint_revolute(
                    parent=links[i - 1],
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                )
            joints.append(j)

        builder.add_articulation(joints, label="pendulum")
        builder.add_ground_plane()

    model = builder.finalize(device=device)

    scene_info.update(
        {
            "body_count": model.body_count,
            "joint_count": model.joint_count,
            "shape_count": model.shape_count,
            "particle_count": model.particle_count,
        }
    )

    return {
        "model": model,
        "builder": builder,
        "scene_info": scene_info,
    }
