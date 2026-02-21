# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Post-process a USDA file to add UsdPreviewSurface materials.

Newton's ViewerUSD writes displayColor primvars but no UsdShade materials.
This module reopens the exported stage and binds proper PBR materials so
that Omniverse, Blender, and other renderers display the scene correctly.
"""

from __future__ import annotations


def bind_materials(usda_path: str) -> None:
    """Add UsdPreviewSurface materials to all geometry prims in a USDA file.

    Creates a ``/root/Looks`` scope with materials for:
    - Robot links (metallic silver)
    - Table / ground plane (matte gray)
    - Cloth / triangle mesh (fabric-like diffuse)

    Args:
        usda_path: Path to the USDA file to modify in-place.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usda_path)
    root = stage.GetDefaultPrim()
    root_path = root.GetPath()

    # Create /root/Looks scope for all materials
    looks_path = root_path.AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks_path)

    def _create_material(name, diffuse, roughness=0.5, metallic=0.0, opacity=1.0):
        """Create a UsdPreviewSurface material and return its prim."""
        mat_path = looks_path.AppendChild(name)
        mat = UsdShade.Material.Define(stage, mat_path)

        shader_path = mat_path.AppendChild("PBRShader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        if opacity < 1.0:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)

        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    # Pre-create materials
    mat_robot = _create_material("RobotMetal", (0.75, 0.75, 0.78), roughness=0.3, metallic=0.8)
    mat_table = _create_material("Table", (0.45, 0.42, 0.40), roughness=0.8, metallic=0.0)
    mat_ground = _create_material("Ground", (0.35, 0.35, 0.35), roughness=0.9, metallic=0.0)
    mat_cloth = _create_material("Cloth", (0.85, 0.75, 0.55), roughness=0.95, metallic=0.0)
    mat_gripper = _create_material("Gripper", (0.2, 0.2, 0.22), roughness=0.4, metallic=0.6)

    bound_count = 0
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        # Skip non-geometry prims
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Gprim)):
            continue

        # Determine which material to bind based on prim path
        mat = None
        if "/model/triangles" in path_str:
            mat = mat_cloth
        elif "/geometry/plane" in path_str or path_str.endswith("/ground"):
            mat = mat_ground
        elif "/geometry/" in path_str:
            # Shape instances (robot links, table)
            # Heuristic: table is usually the last static shape, robot links are body-parented
            if "instance_" in path_str:
                # Extract instance index to distinguish table from robot
                # Table is typically the shape on body -1 (unparented), logged last
                # Robot shapes reference mesh prototypes
                mat = mat_robot
            else:
                mat = mat_robot
        elif "/shapes/" in path_str:
            mat = mat_robot

        if mat is None:
            continue

        # Refine: check if this is a gripper finger (last 2-3 instances in robot)
        if mat == mat_robot and "instance_" in path_str:
            try:
                idx = int(path_str.rsplit("instance_", 1)[1].split("/", maxsplit=1)[0])
                # Franka has ~14 collision shapes; last few are finger pads
                parent_path = path_str.rsplit("/instance_", 1)[0]
                parent_prim = stage.GetPrimAtPath(parent_path)
                children = list(parent_prim.GetChildren()) if parent_prim else []
                num_instances = len(children)
                # Table is the static box — typically the instance right after robot shapes
                if num_instances > 0:
                    if idx == num_instances - 1:
                        # Last instance is often the table
                        mat = mat_table
                    elif idx >= num_instances - 3:
                        mat = mat_gripper
            except (ValueError, IndexError):
                pass

        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(mat)
        bound_count += 1

    stage.GetRootLayer().Save()
    print(f"  Materials bound: {bound_count} prims")
