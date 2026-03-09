"""
Shared Blender utilities for Newton robot import and rigging.
==============================================================
By David Clabaugh with Claude Code

Provides MJCF/URDF parsing, forward kinematics, mesh import, armature
generation, and scene setup functions used by all robot Blender scripts.

No Newton dependency required — all XML parsing done with stdlib ElementTree.
"""

import json
import math
import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

# =============================================================================
# CONSTANTS
# =============================================================================

# Standard Newton simulation scene layout
NEWTON_SCENE = {
    "robot_base": (-0.5, -0.5, -0.1),
    "platform_center": (0.0, -0.5, 0.1),
    "platform_size": (0.8, 0.8, 0.2),
    "platform_top_z": 0.2,
}


# =============================================================================
# ASSET PATH RESOLUTION
# =============================================================================


def _get_script_dir():
    """Resolve the directory containing this script, even inside Blender's text editor."""
    # 1. Normal Python execution (__file__ is defined)
    if "__file__" in dir():
        return os.path.dirname(os.path.abspath(__file__))
    # Blender text editor fallbacks:
    # 2. Text block has a filepath (was opened from disk via Text > Open)
    text = bpy.context.space_data and getattr(bpy.context.space_data, "text", None)
    if text and text.filepath:
        return os.path.dirname(os.path.abspath(text.filepath))
    # 3. Search all text blocks for one whose filepath is in scripts/blender
    for t in bpy.data.texts:
        if t.filepath and "scripts" in t.filepath:
            return os.path.dirname(os.path.abspath(t.filepath))
    # 4. .blend file saved in or near the scripts directory
    if bpy.data.filepath:
        return os.path.dirname(bpy.data.filepath)
    # 5. Hardcoded fallback
    return r"C:\_git\newton_zhao\scripts\blender"


def load_asset_paths(json_path=None):
    """Load robot cache paths from robot_asset_paths.json.

    Returns a dict mapping robot key -> local cache path.
    Returns empty dict if file not found.
    """
    if json_path is None:
        script_dir = _get_script_dir()
        json_path = os.path.join(script_dir, "robot_asset_paths.json")

    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


# =============================================================================
# SCENE SETUP
# =============================================================================


def clean_scene():
    """Remove all objects, meshes, materials, armatures, actions."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in [bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.actions]:
        for item in coll:
            coll.remove(item)


def create_table(garment_bbox=None):
    """Create work table, optionally sized to fit garment with 10cm margin."""
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
    """Create ground plane at Z=0."""
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)
    g.data.materials.append(mat)
    return g


def setup_camera_and_lights():
    """Add camera + sun light, set render FPS/units."""
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 300
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"

    bpy.ops.object.camera_add(location=(1.0, -2.0, 1.2))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(65), 0, math.radians(25))
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(2, -2, 3))
    light = bpy.context.active_object
    light.name = "KeyLight"
    light.data.energy = 3.0


# =============================================================================
# MJCF PARSING (MuJoCo Menagerie)
# =============================================================================


def _expand_mjcf_includes(root, base_dir):
    """Recursively expand <include> tags in MJCF XML.

    MuJoCo Menagerie models use <include file="..."/> to split across files.
    We need to inline them all into one tree for parsing.
    """
    includes = root.findall(".//include")
    for inc in includes:
        inc_file = inc.get("file")
        if inc_file is None:
            continue
        inc_path = os.path.join(base_dir, inc_file)
        if not os.path.isfile(inc_path):
            print(f"  WARNING: include file not found: {inc_path}")
            parent = root.find(f".//{inc.tag}/..")
            if parent is None:
                # Try to find parent by iterating
                for p in root.iter():
                    if inc in list(p):
                        parent = p
                        break
            if parent is not None:
                parent.remove(inc)
            continue

        inc_tree = ET.parse(inc_path)
        inc_root = inc_tree.getroot()

        # Recursively expand nested includes
        inc_dir = os.path.dirname(inc_path)
        _expand_mjcf_includes(inc_root, inc_dir)

        # Find parent of the <include> and replace it with the included content
        parent = None
        for p in root.iter():
            if inc in list(p):
                parent = p
                break
        if parent is not None:
            idx = list(parent).index(inc)
            parent.remove(inc)
            # Insert all children from included root
            for i, child in enumerate(inc_root):
                parent.insert(idx + i, child)
            # Merge attributes from included root onto parent
            for k, v in inc_root.attrib.items():
                if k not in parent.attrib:
                    parent.set(k, v)


def _normalize_quat(q):
    """Normalize a quaternion (w, x, y, z) tuple."""
    length = math.sqrt(sum(c * c for c in q))
    if length < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(c / length for c in q)


def _parse_mjcf_orientation(elem):
    """Parse MuJoCo orientation from quat/axisangle/euler/xyaxes attributes.

    Returns a Blender Quaternion. MuJoCo uses (w, x, y, z) format.
    """
    quat_str = elem.get("quat")
    if quat_str:
        vals = [float(v) for v in quat_str.split()]
        # MuJoCo: (w, x, y, z) — same as Blender Quaternion constructor
        return Quaternion(vals)

    axisangle_str = elem.get("axisangle")
    if axisangle_str:
        vals = [float(v) for v in axisangle_str.split()]
        axis = Vector(vals[:3])
        angle = vals[3]
        return Quaternion(axis, angle)

    euler_str = elem.get("euler")
    if euler_str:
        vals = [float(v) for v in euler_str.split()]
        return Euler(vals, "XYZ").to_quaternion()

    xyaxes_str = elem.get("xyaxes")
    if xyaxes_str:
        vals = [float(v) for v in xyaxes_str.split()]
        x_axis = Vector(vals[0:3]).normalized()
        y_axis = Vector(vals[3:6]).normalized()
        z_axis = x_axis.cross(y_axis).normalized()
        # Build rotation matrix from axes
        mat = Matrix((
            (x_axis.x, y_axis.x, z_axis.x),
            (x_axis.y, y_axis.y, z_axis.y),
            (x_axis.z, y_axis.z, z_axis.z),
        ))
        return mat.to_quaternion()

    return Quaternion()  # Identity


def parse_mjcf(xml_path):
    """Parse MJCF XML, return list of bodies and dict of mesh assets.

    Returns:
        bodies: list of dicts with keys: name, parent, pos, quat, joints, visuals, children
        mesh_assets: dict mapping mesh name -> file path
    """
    base_dir = os.path.dirname(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Expand includes
    _expand_mjcf_includes(root, base_dir)

    # Resolve meshdir from compiler tag
    meshdir = ""
    compiler = root.find(".//compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")

    if meshdir:
        mesh_base = os.path.join(base_dir, meshdir)
    else:
        mesh_base = base_dir

    # Parse mesh assets
    mesh_assets = {}
    for asset in root.iter("mesh"):
        name = asset.get("name", "")
        mesh_file = asset.get("file", "")
        if name and mesh_file:
            mesh_assets[name] = os.path.join(mesh_base, mesh_file)
        # Handle scale attribute
        scale_str = asset.get("scale")
        if scale_str and name:
            mesh_assets[name + "_scale"] = [float(v) for v in scale_str.split()]

    # Parse body hierarchy
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        return [], mesh_assets

    bodies = []
    for body_elem in worldbody:
        if body_elem.tag == "body":
            _parse_mjcf_body_recursive(body_elem, None, bodies)

    return bodies, mesh_assets


def _parse_mjcf_body_recursive(elem, parent_name, bodies):
    """Recursively parse MJCF body hierarchy.

    Each body gets: name, parent, pos, quat, joints (list), visuals (list of mesh names), children
    """
    name = elem.get("name", f"body_{len(bodies)}")

    # Position
    pos_str = elem.get("pos", "0 0 0")
    pos = [float(v) for v in pos_str.split()]

    # Orientation
    quat = _parse_mjcf_orientation(elem)

    # Joints in this body
    joints = []
    for j in elem.findall("joint"):
        joint_name = j.get("name", f"{name}_joint")
        joint_type = j.get("type", "hinge")  # MuJoCo default is hinge
        axis_str = j.get("axis", "0 0 1")
        axis = [float(v) for v in axis_str.split()]

        range_str = j.get("range", "")
        if range_str:
            limits = [math.radians(float(v)) for v in range_str.split()]
        else:
            limits = None

        # Joint position offset within body
        jpos_str = j.get("pos", "0 0 0")
        jpos = [float(v) for v in jpos_str.split()]

        joints.append({
            "name": joint_name,
            "type": joint_type,
            "axis": axis,
            "limits": limits,
            "pos": jpos,
        })

    # Visual geometries referencing meshes
    visuals = []
    for geom in elem.findall("geom"):
        if geom.get("type") == "mesh" or geom.get("mesh"):
            mesh_name = geom.get("mesh", "")
            if mesh_name:
                visuals.append(mesh_name)

    body = {
        "name": name,
        "parent": parent_name,
        "pos": pos,
        "quat": quat,
        "joints": joints,
        "visuals": visuals,
        "children": [],
    }
    bodies.append(body)

    # Recurse into child bodies
    for child in elem.findall("body"):
        child_name = child.get("name", f"body_{len(bodies)}")
        body["children"].append(child_name)
        _parse_mjcf_body_recursive(child, name, bodies)


def compute_mjcf_fk(bodies):
    """Compute world transforms for all bodies at zero config.

    Returns dict mapping body name -> 4x4 Matrix (world transform).
    """
    transforms = {}

    # Build parent lookup
    parent_map = {}
    for b in bodies:
        parent_map[b["name"]] = b["parent"]

    for b in bodies:
        pos = Vector(b["pos"])
        quat = b["quat"]

        local_T = Matrix.Translation(pos) @ quat.to_matrix().to_4x4()

        parent_name = b["parent"]
        if parent_name and parent_name in transforms:
            world_T = transforms[parent_name] @ local_T
        else:
            world_T = local_T

        transforms[b["name"]] = world_T

    return transforms


# =============================================================================
# URDF PARSING (Franka FR3)
# =============================================================================


def parse_urdf(urdf_path, asset_dir=None):
    """Parse URDF, return ordered joints and link mesh paths.

    Args:
        urdf_path: Path to URDF file.
        asset_dir: Base directory for resolving mesh paths. Defaults to URDF parent dir.

    Returns:
        joints: list of dicts with name, type, parent_link, child_link, xyz, rpy, axis, lower, upper
        link_meshes: dict mapping link name -> mesh file path
    """
    if asset_dir is None:
        asset_dir = os.path.dirname(urdf_path)

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

    # Resolve visual mesh paths
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
                    # Strip package:// prefix
                    if "package://" in raw:
                        parts = raw.split("package://", 1)
                        if len(parts) > 1:
                            # Remove the package name part
                            pkg_relative = parts[1]
                            # Try splitting on first /
                            slash_idx = pkg_relative.find("/")
                            if slash_idx >= 0:
                                rel = pkg_relative[slash_idx + 1:]
                            else:
                                rel = pkg_relative
                        else:
                            rel = raw
                    else:
                        rel = raw
                    link_meshes[name] = os.path.join(asset_dir, rel)

    return joints, link_meshes


def compute_urdf_fk(joints):
    """Compute world transforms for all links at zero configuration.

    Returns dict mapping link name -> 4x4 Matrix.
    """
    transforms = {"base": Matrix.Identity(4)}

    for j in joints:
        parent_T = transforms.get(j["parent_link"], Matrix.Identity(4))
        xyz = j["xyz"]
        rpy = j["rpy"]
        T_translate = Matrix.Translation(Vector(xyz))
        R = Euler((rpy[0], rpy[1], rpy[2]), "XYZ").to_matrix().to_4x4()
        transforms[j["child_link"]] = parent_T @ T_translate @ R

    return transforms


# =============================================================================
# MESH IMPORT
# =============================================================================


def import_stl(filepath, name, scale=None):
    """Import STL mesh with optional per-axis scale.

    Args:
        filepath: Path to STL file.
        name: Blender object name.
        scale: Optional [sx, sy, sz] scale factors.

    Returns:
        Blender object or None if file not found.
    """
    if not os.path.isfile(filepath):
        print(f"  WARNING: mesh not found: {filepath}")
        return None

    before = set(bpy.data.objects)
    bpy.ops.import_mesh.stl(filepath=filepath)
    new_objs = [o for o in bpy.data.objects if o not in before]

    if not new_objs:
        return None

    # Join if multiple objects
    bpy.ops.object.select_all(action="DESELECT")
    for o in new_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    obj.name = name

    # Apply mesh-level scale if provided
    if scale:
        scale_matrix = Matrix((
            (scale[0], 0, 0, 0),
            (0, scale[1], 0, 0),
            (0, 0, scale[2], 0),
            (0, 0, 0, 1),
        ))
        obj.data.transform(scale_matrix)
        obj.data.update()

    # Reset object transform
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)

    return obj


def parse_dae_node_transform(filepath):
    """Extract the per-node transform matrix from a COLLADA DAE file.

    Blender's COLLADA importer ignores these per-node <matrix> transforms,
    leaving geometry in the authoring tool's coordinate frame at mm scale.
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
    """Import a COLLADA .dae visual mesh and apply the embedded transform.

    Handles Blender's COLLADA importer limitations by extracting and applying
    the embedded per-node transform matrix that Blender ignores.
    """
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


def import_garment(obj_path, color=(0.9, 0.5, 0.6, 1.0), cloth_scale=0.01):
    """Import garment OBJ and place on table.

    Supports two modes:
      - *_clean.obj: already in meters, centered, oriented. Just import and place.
      - Raw *_sim.obj: scale cm->m, rotate 180 Y, center origin.

    Args:
        obj_path: Path to OBJ file.
        color: RGBA material color tuple.
        cloth_scale: Scale factor for raw meshes (default 0.01 = cm to m).

    Returns:
        dict with keys: obj (Blender object), bbox_size [w, d, h]
        or None if file not found.
    """
    if not os.path.isfile(obj_path):
        print(f"  WARNING: garment not found: {obj_path}")
        return None

    is_clean = "_clean.obj" in obj_path

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

    if not is_clean:
        # Raw mesh: scale cm->m, rotate 180 around Y, center origin
        cloth.data.transform(Matrix.Scale(cloth_scale, 4))
        cloth.data.transform(Matrix.Rotation(math.pi, 4, "Y"))

        # Center vertices: bbox center XY, bottom at Z=0
        verts = [Vector(v.co) for v in cloth.data.vertices]
        if verts:
            xs = [v.x for v in verts]
            ys = [v.y for v in verts]
            zs = [v.z for v in verts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            cz = min(zs)
            cloth.data.transform(Matrix.Translation(Vector((-cx, -cy, -cz))))

        cloth.data.update()
        print("  (raw mesh — applied scale + rotation + centering)")
    else:
        print("  (clean mesh — already in meters, centered)")

    cloth.data.update()

    # Set Blender origin to geometry center
    bpy.ops.object.select_all(action="DESELECT")
    cloth.select_set(True)
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Measure bbox for table sizing
    verts = [Vector(v.co) for v in cloth.data.vertices]
    if not verts:
        return {"obj": cloth, "bbox_size": [0, 0, 0]}
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    bbox_w = max(xs) - min(xs)
    bbox_d = max(ys) - min(ys)
    bbox_h = max(zs) - min(zs)

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

    return {"obj": cloth, "bbox_size": [bbox_w, bbox_d, bbox_h]}


# =============================================================================
# ROBOT MESH IMPORT
# =============================================================================


def import_mjcf_robot_meshes(bodies, mesh_assets, body_transforms, base_matrix,
                              root_name="RobotRoot", material_color=(0.5, 0.5, 0.5, 1.0)):
    """Import all visual meshes for an MJCF robot at zero-config positions.

    Creates a root empty, imports each body's visual meshes, positions them
    at their world transform, applies a shared material, and parents to root.

    Args:
        bodies: List of body dicts from parse_mjcf().
        mesh_assets: Dict of mesh name -> file path from parse_mjcf().
        body_transforms: Dict of body name -> 4x4 Matrix from compute_mjcf_fk().
        base_matrix: 4x4 Matrix for the robot base position/orientation.
        root_name: Name for the root empty.
        material_color: RGBA tuple for the robot material.

    Returns:
        (root_empty, imported_dict) where imported_dict maps body name -> list of mesh objects
    """
    # Create root empty
    bpy.ops.object.empty_add(type="ARROWS", location=base_matrix.translation, radius=0.08)
    root_empty = bpy.context.active_object
    root_empty.name = root_name

    # Create shared material
    mat = bpy.data.materials.new(name=f"{root_name}_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = material_color
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.3

    imported = {}

    for body in bodies:
        body_name = body["name"]
        body_T = body_transforms.get(body_name, Matrix.Identity(4))
        world_T = base_matrix @ body_T

        body_meshes = []
        for mesh_name in body.get("visuals", []):
            mesh_path = mesh_assets.get(mesh_name, "")
            if not mesh_path or not os.path.isfile(mesh_path):
                continue

            # Check for mesh-specific scale
            scale = mesh_assets.get(mesh_name + "_scale")

            # Determine import method by extension
            ext = os.path.splitext(mesh_path)[1].lower()
            obj_name = f"{body_name}_{mesh_name}"

            if ext == ".stl":
                mesh_obj = import_stl(mesh_path, obj_name, scale=scale)
            elif ext == ".dae":
                mesh_obj = import_dae(mesh_path, obj_name)
            elif ext == ".obj":
                # Simple OBJ import
                before = set(bpy.data.objects)
                try:
                    bpy.ops.wm.obj_import(filepath=mesh_path)
                except AttributeError:
                    bpy.ops.import_scene.obj(filepath=mesh_path)
                new_objs = [o for o in bpy.data.objects if o not in before]
                mesh_obj = new_objs[0] if new_objs else None
                if mesh_obj:
                    mesh_obj.name = obj_name
            else:
                print(f"  WARNING: unsupported mesh format {ext}: {mesh_path}")
                continue

            if mesh_obj is None:
                continue

            # Apply scale if STL didn't handle it already
            if scale and ext != ".stl":
                scale_matrix = Matrix((
                    (scale[0], 0, 0, 0),
                    (0, scale[1], 0, 0),
                    (0, 0, scale[2], 0),
                    (0, 0, 0, 1),
                ))
                mesh_obj.data.transform(scale_matrix)
                mesh_obj.data.update()

            # Position at world transform
            mesh_obj.matrix_world = world_T

            # Apply material
            if mesh_obj.data.materials:
                mesh_obj.data.materials[0] = mat
            else:
                mesh_obj.data.materials.append(mat)

            # Parent to root empty
            mesh_obj.parent = root_empty
            mesh_obj.matrix_parent_inverse = root_empty.matrix_world.inverted()

            body_meshes.append(mesh_obj)
            print(f"    {body_name}: {os.path.basename(mesh_path)}")

        if body_meshes:
            imported[body_name] = body_meshes

    return root_empty, imported


def create_mjcf_joint_empties(bodies, body_transforms, base_matrix, root_empty):
    """Create empties at each joint position with arrow pointing along rotation axis.

    Stores joint metadata as custom properties on each empty:
    axis_world, limits (degrees), parent_body.
    """
    for body in bodies:
        body_name = body["name"]
        body_T = body_transforms.get(body_name, Matrix.Identity(4))

        for joint in body.get("joints", []):
            if joint["type"] not in ("hinge", "slide", "revolute", "prismatic"):
                continue

            # Joint position is body position + joint offset
            jpos = Vector(joint.get("pos", (0, 0, 0)))
            joint_local_T = body_T @ Matrix.Translation(jpos)
            world_T = base_matrix @ joint_local_T
            world_pos = world_T.translation

            # Compute world rotation axis
            axis_local = Vector(joint["axis"])
            world_axis = (base_matrix @ body_T).to_3x3() @ axis_local
            world_axis.normalize()

            bpy.ops.object.empty_add(
                type="SINGLE_ARROW",
                location=world_pos,
                radius=0.04,
            )
            emp = bpy.context.active_object
            emp.name = f"JNT_{joint['name']}"

            # Point arrow along rotation axis
            emp.rotation_euler = world_axis.to_track_quat("Z", "Y").to_euler()

            emp.parent = root_empty
            emp.matrix_parent_inverse = root_empty.matrix_world.inverted()

            # Store metadata as custom properties
            emp["joint_type"] = joint["type"]
            emp["axis_world"] = [round(v, 4) for v in world_axis]
            emp["parent_body"] = body_name
            if joint["limits"]:
                emp["limit_lower_deg"] = round(math.degrees(joint["limits"][0]), 1)
                emp["limit_upper_deg"] = round(math.degrees(joint["limits"][1]), 1)
                emp["limit_lower_rad"] = round(joint["limits"][0], 4)
                emp["limit_upper_rad"] = round(joint["limits"][1], 4)


# =============================================================================
# ARMATURE GENERATION
# =============================================================================


def _collect_joint_chain_mjcf(bodies, body_transforms, base_matrix):
    """Extract flat chain of movable joints from MJCF bodies.

    Returns list of dicts: name, world_pos, world_axis, limits, parent_body
    """
    chain = []
    for body in bodies:
        body_name = body["name"]
        body_T = body_transforms.get(body_name, Matrix.Identity(4))

        for joint in body.get("joints", []):
            if joint["type"] not in ("hinge", "slide", "revolute", "prismatic"):
                continue

            jpos = Vector(joint.get("pos", (0, 0, 0)))
            joint_local_T = body_T @ Matrix.Translation(jpos)
            world_T = base_matrix @ joint_local_T
            world_pos = world_T.translation

            axis_local = Vector(joint["axis"])
            world_axis = (base_matrix @ body_T).to_3x3() @ axis_local
            world_axis.normalize()

            chain.append({
                "name": joint["name"],
                "world_pos": world_pos.copy(),
                "world_axis": world_axis.copy(),
                "limits": joint["limits"],
                "parent_body": body_name,
            })

    return chain


def _collect_joint_chain_urdf(joints, link_transforms, base_matrix):
    """Extract flat chain of movable joints from URDF.

    Returns list of dicts: name, world_pos, world_axis, limits, child_link
    """
    chain = []
    for j in joints:
        if j["type"] not in ("revolute", "prismatic"):
            continue

        child_T = link_transforms.get(j["child_link"], Matrix.Identity(4))
        world_T = base_matrix @ child_T
        world_pos = world_T.translation

        world_axis = child_T.to_3x3() @ Vector(j["axis"])
        world_axis.normalize()

        chain.append({
            "name": j["name"],
            "world_pos": world_pos.copy(),
            "world_axis": world_axis.copy(),
            "limits": [j["lower"], j["upper"]] if (j["lower"] != 0 or j["upper"] != 0) else None,
            "child_link": j["child_link"],
        })

    return chain


def create_armature(name, joint_chain, root_empty):
    """Create a Blender armature with bones at joint positions.

    Bones are placed at joint positions but NOT connected (robot joints have offsets).
    Joint metadata (axis, limits, parent body) stored as custom properties on pose bones.

    Args:
        name: Armature object name.
        joint_chain: List of joint dicts from _collect_joint_chain_*().
        root_empty: Parent empty for the armature.

    Returns:
        Armature object.
    """
    if not joint_chain:
        print("  WARNING: empty joint chain, no armature created")
        return None

    # Create armature data and object
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    # Parent to root empty
    arm_obj.parent = root_empty
    arm_obj.matrix_parent_inverse = root_empty.matrix_world.inverted()

    # Enter edit mode to create bones
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    MIN_BONE_LENGTH = 0.02  # Minimum bone length to avoid zero-length bones

    prev_bone = None
    for i, joint in enumerate(joint_chain):
        bone = arm_data.edit_bones.new(joint["name"])

        head = joint["world_pos"]
        # Bone tail extends along the joint axis
        tail = head + joint["world_axis"] * 0.05

        # Ensure minimum bone length
        if (tail - head).length < MIN_BONE_LENGTH:
            tail = head + Vector((0, 0, MIN_BONE_LENGTH))

        bone.head = head
        bone.tail = tail

        # Parent to previous bone (NOT connected — joints have offsets)
        if prev_bone is not None:
            bone.parent = prev_bone
            bone.use_connect = False

        prev_bone = bone

    bpy.ops.object.mode_set(mode="POSE")

    # Store metadata on pose bones
    for joint in joint_chain:
        pbone = arm_obj.pose.bones.get(joint["name"])
        if pbone is None:
            continue

        pbone["axis_world"] = [round(v, 4) for v in joint["world_axis"]]

        if joint.get("limits"):
            limits = joint["limits"]
            pbone["limit_lower_rad"] = round(limits[0], 4)
            pbone["limit_upper_rad"] = round(limits[1], 4)
            pbone["limit_lower_deg"] = round(math.degrees(limits[0]), 1)
            pbone["limit_upper_deg"] = round(math.degrees(limits[1]), 1)

        if "parent_body" in joint:
            pbone["parent_body"] = joint["parent_body"]
        if "child_link" in joint:
            pbone["child_link"] = joint["child_link"]

    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"  Created {name} with {len(joint_chain)} bones")
    return arm_obj


def create_armature_from_mjcf(name, bodies, body_transforms, base_matrix, root_empty):
    """Convenience: create armature from MJCF bodies."""
    chain = _collect_joint_chain_mjcf(bodies, body_transforms, base_matrix)
    return create_armature(name, chain, root_empty)


def create_armature_from_urdf(name, joints, link_transforms, base_matrix, root_empty):
    """Convenience: create armature from URDF joints."""
    chain = _collect_joint_chain_urdf(joints, link_transforms, base_matrix)
    return create_armature(name, chain, root_empty)


def create_combined_armature(name, arm_bodies, arm_transforms, hand_bodies, hand_transforms,
                              base_matrix, root_empty):
    """Create single armature for arm+hand combos.

    Collects joint chains from both arm and hand, concatenates them,
    and creates a single armature with all bones.
    """
    arm_chain = _collect_joint_chain_mjcf(arm_bodies, arm_transforms, base_matrix)
    hand_chain = _collect_joint_chain_mjcf(hand_bodies, hand_transforms, Matrix.Identity(4))
    combined = arm_chain + hand_chain
    return create_armature(name, combined, root_empty)


# =============================================================================
# HAND ATTACHMENT (MuJoCo Menagerie composition)
# =============================================================================


def compute_hand_base_transform(arm_transforms, ee_body_name, hand_xform_pos, hand_quat, base_matrix):
    """Compute world transform for hand base attached to arm end-effector.

    Args:
        arm_transforms: Dict of arm body transforms from compute_mjcf_fk().
        ee_body_name: Name of the arm end-effector body (e.g. "wrist_3_link").
        hand_xform_pos: [x, y, z] position offset for hand attachment.
        hand_quat: Quaternion rotation for hand attachment.
        base_matrix: Robot base world transform.

    Returns:
        4x4 Matrix: hand base transform in world space.
    """
    ee_local = arm_transforms.get(ee_body_name, Matrix.Identity(4))
    ee_world = base_matrix @ ee_local

    hand_offset = Matrix.Translation(Vector(hand_xform_pos))
    hand_rot = hand_quat.to_matrix().to_4x4()
    hand_local = hand_offset @ hand_rot

    return ee_world @ hand_local


def compute_hand_fk_in_world(hand_bodies, hand_base_world):
    """Transform hand FK results to world space.

    Takes the hand bodies' local FK transforms and applies the hand base
    world transform to get world positions for all hand bodies.

    Args:
        hand_bodies: List of hand body dicts from parse_mjcf().
        hand_base_world: 4x4 Matrix from compute_hand_base_transform().

    Returns:
        Dict mapping hand body name -> 4x4 Matrix (world space).
    """
    # First compute local FK for the hand
    local_transforms = compute_mjcf_fk(hand_bodies)

    # Transform to world space
    world_transforms = {}
    for name, local_T in local_transforms.items():
        world_transforms[name] = hand_base_world @ local_T

    return world_transforms


# =============================================================================
# SUMMARY HELPERS
# =============================================================================


def print_setup_complete(robot_name, base_pos, joint_count, mesh_count, armature_name):
    """Print summary message after setup is complete."""
    rx, ry, rz = base_pos
    print(f"""
{"=" * 65}
READY — {robot_name} meshes imported + armature generated
{"=" * 65}

Layout:
  Robot root at ({rx}, {ry}, {rz})
  {mesh_count} mesh objects at zero-config positions
  Joint empties (JNT_*) show rotation axis direction (arrow)
  {armature_name} with {joint_count} bones
  Each JNT empty / pose bone has custom properties: axis_world, limits, etc.

Rigging steps (you do these):
  1. Select a mesh, then Shift-select the armature
  2. Ctrl+P > Bone (for rigid robot parts, no weight painting)
  3. In Pose Mode, select the matching bone for each mesh
  4. Repeat for all link meshes
  5. Test rotations match expected joint axes

The JNT_* empties show the world rotation axis for each joint.
Check their custom properties for exact axis vectors and limits.

When your rig is done, save the .blend file — export_trajectories.py
will read the bone rotations for Newton simulation.
""")
