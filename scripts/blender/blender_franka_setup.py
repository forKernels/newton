"""
Blender Franka FR3 Setup — Mesh Import + Rigging-Ready Layout
==============================================================
Imports all Franka FR3 visual meshes at their correct zero-config positions,
plus a garment from the Maria dataset laid flat on the table.

The meshes are NOT rigged — you do the rigging yourself.
Joint positions are marked with empties and printed to console as reference.

Usage:
  1. Open Blender 3.6+
  2. Scripting workspace > Open this file > Run Script
  3. All meshes arrive at zero-config (arm straight up)
  4. Joint empties show where each revolute joint is
  5. Rig it: create armature, parent meshes, add constraints
  6. When done, run export_trajectories.py

Scene layout (matches Newton simulation):
  Robot base:   (-0.5, -0.5, -0.1)
  Table center: (0.0, -0.5, 0.1), top at Z=0.20

URDF joint reference (all axes are LOCAL Z in the URDF frame):
  Joint 1: Z +0.333 from base, axis=Z (base rotation)
  Joint 2: same pos, RPY(-90°,0,0), axis=Z (shoulder pitch, world Y after RPY)
  Joint 3: Y -0.316 from J2 frame, RPY(+90°,0,0), axis=Z (elbow)
  Joint 4: X +0.0825 from J3 frame, RPY(+90°,0,0), axis=Z
  Joint 5: X -0.0825, Y +0.384 from J4, RPY(-90°,0,0), axis=Z
  Joint 6: same pos, RPY(+90°,0,0), axis=Z (wrist pitch)
  Joint 7: X +0.088 from J6, RPY(+90°,0,0), axis=Z (wrist roll)
  Hand:    Z +0.107 from J7, then RPY(0,0,-45°) fixed
  TCP:     Z +0.1034 from hand
  Fingers: prismatic along local Y, ±0.04m
"""

import bpy
import math
import os
import xml.etree.ElementTree as ET
from mathutils import Vector, Matrix, Euler

# =============================================================================
# CONFIGURATION — Edit these paths
# =============================================================================

DATASET_ROOT = r"D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set"

# Garment to load (relative to DATASET_ROOT). Set to "" to skip.
GARMENT_OBJ = "dress_sleeveless_2550/dress_sleeveless_000YCTJ9HS/dress_sleeveless_000YCTJ9HS_sim.obj"

CLOTH_SCALE = 0.01  # cm to meters

# Newton-cached Franka assets
FRANKA_ASSET_DIR = (
    r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache"
    r"\newton-assets_franka_emika_panda_c9e36853\franka_emika_panda"
)
FRANKA_URDF = os.path.join(FRANKA_ASSET_DIR, "urdf", "fr3_franka_hand.urdf")

NEWTON_SCENE = {
    "robot_base": (-0.5, -0.5, -0.1),
    "platform_center": (0.0, -0.5, 0.1),
    "platform_size": (0.8, 0.8, 0.2),
    "platform_top_z": 0.2,
}


# =============================================================================
# URDF PARSING + FK
# =============================================================================

def parse_urdf(urdf_path):
    """Parse the FR3 URDF, return ordered joints and link mesh paths."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz_str = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy_str = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = [float(v) for v in xyz_str.split()]
        rpy = [float(v) for v in rpy_str.split()]

        axis_elem = j.find("axis")
        axis = [float(v) for v in axis_elem.get("xyz").split()] if axis_elem is not None else [0, 0, 1]

        lim = j.find("limit")
        lower = float(lim.get("lower", "0")) if lim is not None else 0
        upper = float(lim.get("upper", "0")) if lim is not None else 0

        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent_link": j.find("parent").get("link"),
            "child_link": j.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        })

    # Resolve visual DAE mesh paths
    link_meshes = {}
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        if vis is not None:
            geom = vis.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    raw = mesh_el.get("filename", "")
                    if "package://" in raw:
                        rel = raw.split("package://franka_emika_panda/", 1)[-1]
                    else:
                        rel = raw
                    link_meshes[name] = os.path.join(FRANKA_ASSET_DIR, rel)

    return joints, link_meshes


def compute_link_transforms(joints):
    """Compute world transform of every link at zero configuration (all q=0)."""
    transforms = {"base": Matrix.Identity(4)}

    for j in joints:
        parent_T = transforms.get(j["parent_link"], Matrix.Identity(4))
        xyz = j["xyz"]
        rpy = j["rpy"]  # roll, pitch, yaw
        T_translate = Matrix.Translation(Vector(xyz))
        R = Euler((rpy[0], rpy[1], rpy[2]), "XYZ").to_matrix().to_4x4()
        transforms[j["child_link"]] = parent_T @ T_translate @ R

    return transforms


# =============================================================================
# SCENE HELPERS
# =============================================================================

def clean_scene():
    """Remove all objects, meshes, materials, armatures, actions."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in [bpy.data.meshes, bpy.data.materials,
                 bpy.data.armatures, bpy.data.actions]:
        for item in coll:
            coll.remove(item)


def create_table(garment_bbox=None):
    """Create table sized to fit the garment (or larger)."""
    cx, cy, cz = NEWTON_SCENE["platform_center"]
    sx, sy, sz = NEWTON_SCENE["platform_size"]

    # Expand table to fit garment with 10cm margin on each side
    if garment_bbox is not None:
        margin = 0.10
        sx = max(sx, garment_bbox[0] + 2 * margin)
        sy = max(sy, garment_bbox[1] + 2 * margin)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, cz))
    table = bpy.context.active_object
    table.name = "WorkTable"
    table.scale = (sx / 2, sy / 2, sz / 2)

    mat = bpy.data.materials.new(name="TableMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.4, 0.35, 0.3, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    table.data.materials.append(mat)
    return table


def create_ground():
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    g.data.materials.append(mat)
    return g


# =============================================================================
# DAE MESH IMPORT
# =============================================================================

def parse_dae_node_transform(filepath):
    """Extract the per-node transform matrix from a COLLADA DAE file.

    Blender's COLLADA importer ignores these per-node <matrix> transforms,
    leaving geometry in the authoring tool's coordinate frame at mm scale.
    Each Franka link has a DIFFERENT rotation+scale+offset.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for matrix_el in root.iter(f"{ns}matrix"):
            text = matrix_el.text.strip()
            vals = [float(v) for v in text.split()]
            if len(vals) == 16:
                # COLLADA stores row-major
                return Matrix((
                    vals[0:4],
                    vals[4:8],
                    vals[8:12],
                    vals[12:16],
                ))
    except Exception as e:
        print(f"  WARNING: could not parse DAE transform: {e}")
    return Matrix.Identity(4)


def import_dae(filepath, name):
    """Import a COLLADA .dae visual mesh and apply the embedded transform."""
    if not os.path.isfile(filepath):
        print(f"  WARNING: mesh not found: {filepath}")
        return None

    dae_transform = parse_dae_node_transform(filepath)

    before = set(bpy.data.objects)
    bpy.ops.wm.collada_import(filepath=filepath)
    new_objs = [o for o in bpy.data.objects if o not in before]

    if not new_objs:
        return None

    # Join multiple sub-objects into one
    bpy.ops.object.select_all(action="DESELECT")
    for o in new_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    obj.name = name

    # Reset object transform — bake correction into mesh data
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)

    # Apply the DAE's embedded transform (rotation + 0.001 scale + offset)
    obj.data.transform(dae_transform)
    obj.data.update()

    return obj


# =============================================================================
# FRANKA MESH IMPORT (no rigging — you do that)
# =============================================================================

def import_franka_meshes():
    """
    Import all Franka FR3 visual meshes at their zero-config world positions.

    Creates:
      - Mesh objects: fr3_link0_visual through fr3_link7_visual, fr3_hand_visual
      - Joint empties: JNT_fr3_joint1 through JNT_fr3_joint7, JNT_hand, JNT_tcp
      - All parented under a "FrankaRoot" empty at the robot base position

    Nothing is rigged. Meshes and empties are just placed correctly.
    """
    base_pos = Vector(NEWTON_SCENE["robot_base"])
    base_matrix = Matrix.Translation(base_pos)

    # Parse URDF
    joints, link_meshes = parse_urdf(FRANKA_URDF)
    link_transforms = compute_link_transforms(joints)

    # Root empty at robot base
    bpy.ops.object.empty_add(type="ARROWS", location=base_pos, radius=0.08)
    root_empty = bpy.context.active_object
    root_empty.name = "FrankaRoot"

    # --- Import meshes and position at zero config ---
    imported = {}
    for link_name, mesh_path in link_meshes.items():
        mesh = import_dae(mesh_path, f"{link_name}_visual")
        if mesh is None:
            continue

        # Position: robot_base × link_FK_transform
        T = link_transforms.get(link_name, Matrix.Identity(4))
        mesh.matrix_world = base_matrix @ T

        # Parent to root empty (keep current position)
        mesh.parent = root_empty
        mesh.matrix_parent_inverse = root_empty.matrix_world.inverted()

        imported[link_name] = mesh
        print(f"    {link_name}: {mesh_path.split('meshes/')[-1]}")

    # --- Create empties at joint positions (rigging reference) ---
    print("\n  Joint positions (world coords, zero config):")
    for j in joints:
        if j["type"] not in ("revolute", "prismatic"):
            continue

        child_T = link_transforms.get(j["child_link"], Matrix.Identity(4))
        world_pos = base_matrix @ child_T

        # Compute world rotation axis
        world_axis = child_T.to_3x3() @ Vector(j["axis"])
        world_axis.normalize()

        bpy.ops.object.empty_add(
            type="SINGLE_ARROW",
            location=world_pos.translation,
            radius=0.04,
        )
        emp = bpy.context.active_object
        emp.name = f"JNT_{j['name']}"

        # Point the arrow along the rotation axis
        # Single arrow points along +Z by default, rotate to match joint axis
        emp.rotation_euler = world_axis.to_track_quat("Z", "Y").to_euler()

        emp.parent = root_empty
        emp.matrix_parent_inverse = root_empty.matrix_world.inverted()

        # Store URDF data as custom properties (useful for export later)
        emp["joint_type"] = j["type"]
        emp["axis_local"] = list(j["axis"])
        emp["axis_world"] = [round(v, 4) for v in world_axis]
        emp["limit_lower"] = j["lower"]
        emp["limit_upper"] = j["upper"]
        emp["rpy"] = list(j["rpy"])
        emp["child_link"] = j["child_link"]

        ax = f"({world_axis.x:+.2f}, {world_axis.y:+.2f}, {world_axis.z:+.2f})"
        pos = world_pos.translation
        lim = f"[{math.degrees(j['lower']):+.1f}° .. {math.degrees(j['upper']):+.1f}°]"
        print(f"    {j['name']:20s}  pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})  "
              f"axis={ax}  {lim}")

    # TCP empty
    tcp_T = link_transforms.get("fr3_hand_tcp", Matrix.Identity(4))
    tcp_world = base_matrix @ tcp_T
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tcp_world.translation, radius=0.06)
    tcp = bpy.context.active_object
    tcp.name = "JNT_tcp"
    tcp.parent = root_empty
    tcp.matrix_parent_inverse = root_empty.matrix_world.inverted()

    return root_empty


# =============================================================================
# GARMENT IMPORT
# =============================================================================

def import_garment(obj_path, color=(0.9, 0.5, 0.6, 1.0)):
    """
    Import garment OBJ from Maria dataset and lay flat on the table.
    Uses direct mesh.data.transform() for reliability.
    """
    if not os.path.isfile(obj_path):
        print(f"ERROR: garment not found: {obj_path}")
        return None

    # Import OBJ with NO axis conversion
    try:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="NEGATIVE_Y", up_axis="Z")
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="-Y", axis_up="Z")

    cloth = bpy.context.selected_objects[0]
    cloth.name = "Garment"

    cloth.location = (0, 0, 0)
    cloth.rotation_euler = (0, 0, 0)
    cloth.scale = (1, 1, 1)

    # Scale cm → meters, flip Y (neckline up) and Z (lay flat)
    scale = CLOTH_SCALE
    cloth.data.transform(Matrix.Diagonal(Vector((scale, -scale, -scale, 1.0))))
    cloth.data.flip_normals()
    cloth.data.update()

    # Get bounding box
    verts = [Vector(v.co) for v in cloth.data.vertices]
    if not verts:
        return cloth

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]

    bbox_w = max(xs) - min(xs)
    bbox_d = max(ys) - min(ys)
    bbox_h = max(zs) - min(zs)

    # Center vertices at origin, bottom at Z=0
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = min(zs)
    cloth.data.transform(Matrix.Translation(Vector((-cx, -cy, -cz))))
    cloth.data.update()

    # Set Blender origin to geometry center (critical for correct pivots)
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Position on table
    table_top = NEWTON_SCENE["platform_top_z"]
    table_cx, table_cy = NEWTON_SCENE["platform_center"][0], NEWTON_SCENE["platform_center"][1]
    cloth.location = Vector((table_cx, table_cy, table_top + 0.005))

    # Material
    mat = bpy.data.materials.new(name="GarmentMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.85
    if cloth.data.materials:
        cloth.data.materials[0] = mat
    else:
        cloth.data.materials.append(mat)

    print(f"  Garment: {os.path.basename(obj_path)}")
    print(f"  Size (m): {bbox_w:.3f} x {bbox_d:.3f} x {bbox_h:.3f}")
    print(f"  Position: ({cloth.location.x:.3f}, {cloth.location.y:.3f}, {cloth.location.z:.3f})")

    cloth["bbox_size"] = [bbox_w, bbox_d, bbox_h]
    return cloth


# =============================================================================
# MAIN
# =============================================================================

def setup_scene():
    print("=" * 60)
    print("FRANKA FR3 — MESH IMPORT (RIGGING-READY)")
    print("=" * 60)

    print("\n[1/5] Cleaning scene...")
    clean_scene()

    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 300
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"

    print("[2/5] Importing garment...")
    cloth = None
    garment_bbox = None
    if GARMENT_OBJ:
        obj_path = os.path.join(DATASET_ROOT, GARMENT_OBJ)
        cloth = import_garment(obj_path)
        if cloth and "bbox_size" in cloth:
            garment_bbox = cloth["bbox_size"]

    print("[3/5] Creating environment (table sized to garment)...")
    create_ground()
    create_table(garment_bbox=garment_bbox)

    print("[4/5] Importing Franka meshes at zero-config positions...")
    root_empty = import_franka_meshes()

    print("[5/5] Camera + lighting...")
    bpy.ops.object.camera_add(location=(1.0, -2.0, 1.2))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(65), 0, math.radians(25))
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(2, -2, 3))
    light = bpy.context.active_object
    light.name = "KeyLight"
    light.data.energy = 3.0

    rx, ry, rz = NEWTON_SCENE["robot_base"]
    print(f"""
{'=' * 60}
READY — Franka FR3 meshes imported, ready for rigging
{'=' * 60}

Layout:
  FrankaRoot empty at ({rx}, {ry}, {rz})
  9 mesh objects (link0-7 + hand) at zero-config positions
  Joint empties (JNT_*) show rotation axis direction (arrow)
  Each JNT empty has custom properties: axis_world, limits, etc.

Rigging guide (Maya → Blender):
  1. Add > Armature > Single Bone (or empty armature)
  2. Tab into Edit Mode to add/position bones
  3. Each bone = one joint. Head at joint position, tail toward next joint.
  4. For a mechanical rig: DON'T connect bones (uncheck "Connected")
     Instead, parent each bone to the previous one.
  5. In Pose Mode, set rotation mode to XYZ Euler on each bone.
  6. Lock the 2 axes that shouldn't rotate:
     - Properties > Bone > Transform Locks
     - Or use a Limit Rotation constraint (like Maya's joint limits)
  7. Parent each mesh to its bone: select mesh, then armature,
     Ctrl+P > Bone (keep offset). Or use "Bone" parent type.
  8. For IK: add an Empty as target, then in Pose Mode add an
     IK constraint to the last bone in the chain.

The JNT_* empties show the world rotation axis for each joint.
Check their custom properties for exact axis vectors and limits.

When your rig is done, save the .blend file — export_trajectories.py
will read the bone rotations for Newton simulation.
""")


if __name__ == "__main__":
    setup_scene()
