# maria_obj_to_cloth_usd.py
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, Vt
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
        uv_primvar = garment_mesh.CreatePrimvar(
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

    # NOT kinematic — this is the key
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateKinematicEnabledAttr().Set(False)

    # Particle cloth API — tells Newton this is cloth
    PhysxSchema.PhysxParticleAPI.Apply(prim)
    cloth_api = PhysxSchema.PhysxParticleClothAPI.Apply(prim)

    # Cloth properties
    cloth_api.CreateSelfCollisionAttr().Set(True)

    # Stiffness and damping
    auto_cloth = PhysxSchema.PhysxAutoParticleClothAPI.Apply(prim)
    auto_cloth.CreateStretchStiffnessAttr().Set(80.0)
    auto_cloth.CreateShearStiffnessAttr().Set(40.0)
    auto_cloth.CreateBendStiffnessAttr().Set(5.0)
    auto_cloth.CreateDampingAttr().Set(0.005)

    stage.GetRootLayer().Save()

    print(f"Saved: {usd_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Up axis: {up_axis}")
    print("  Kinematic: False")
    print("  Cloth API: Applied")


# Batch convert all MARIA garments
maria_dir = Path("/path/to/maria_objs")
output_dir = Path("/path/to/newton_usd")
output_dir.mkdir(exist_ok=True)

for obj_file in sorted(maria_dir.glob("*.obj")):
    usd_file = output_dir / (obj_file.stem + ".usda")
    convert_maria_obj_to_cloth_usd(obj_file, usd_file, up_axis="Z")