# maria_obj_to_cloth_usd.py
#
# Converts Maria garment OBJs to USDA with standard UsdPhysics APIs.
# Newton loads these via parse_usd() — cloth properties (stiffness, damping)
# are applied in the Newton sim script, not baked into the USD.

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, Vt
import trimesh
import numpy as np
from pathlib import Path


def convert_maria_obj_to_cloth_usd(obj_path, usd_path, up_axis="Z"):
    mesh = trimesh.load(str(obj_path), process=False)

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Create stage
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root xform
    world = UsdGeom.Xform.Define(stage, "/World")

    # Physics scene under world
    physics_scene = UsdPhysics.Scene.Define(stage, world.GetPath().AppendPath("PhysicsScene"))
    physics_scene.CreateGravityDirectionAttr().Set(
        Gf.Vec3f(0.0, 0.0, -1.0) if up_axis == "Z" else Gf.Vec3f(0.0, -1.0, 0.0)
    )
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Garment mesh under world
    garment_path = world.GetPath().AppendPath("Garment")
    garment_mesh = UsdGeom.Mesh.Define(stage, garment_path)

    # Geometry
    garment_mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices]))
    garment_mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    garment_mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))

    # UVs if present
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
        primvars_api = UsdGeom.PrimvarsAPI(garment_mesh)
        uv_primvar = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        uv_primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))

    # Normals
    if mesh.vertex_normals is not None:
        garment_mesh.CreateNormalsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*n) for n in mesh.vertex_normals])
        )

    # Get prim for physics APIs
    prim = stage.GetPrimAtPath(garment_path)

    # Collision
    UsdPhysics.CollisionAPI.Apply(prim)

    # Mass — per particle
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(0.03 * len(vertices))

    # Non-kinematic deformable body
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateKinematicEnabledAttr().Set(False)

    stage.GetRootLayer().Save()

    print(f"Saved: {usd_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Up axis: {up_axis}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert Maria OBJ garments to USDA with physics")
    parser.add_argument("--maria-dir", type=str, required=True, help="Root Maria dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for USDA files")
    parser.add_argument("--pattern", type=str, default="*_sim_prep.obj", help="Glob pattern for OBJ files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of conversions (0 = all)")
    args = parser.parse_args()

    maria_dir = Path(args.maria_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objs = sorted(maria_dir.rglob(args.pattern))
    if args.limit > 0:
        objs = objs[:args.limit]

    print(f"Converting {len(objs)} garments from {maria_dir}")
    print(f"Output: {output_dir}")
    print()

    for i, obj_file in enumerate(objs, 1):
        usd_file = output_dir / (obj_file.stem + ".usda")
        if usd_file.exists():
            print(f"[{i}/{len(objs)}] SKIP (exists): {usd_file.name}")
            continue
        print(f"[{i}/{len(objs)}] {obj_file.name}")
        convert_maria_obj_to_cloth_usd(obj_file, usd_file, up_axis="Z")
        print()
