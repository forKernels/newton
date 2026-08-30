"""
Blender Franka FR3 — Armature Generation
==========================================
By David Clabaugh with Claude Code

Generates a Blender armature from the Franka FR3 URDF at world origin.
Each bone corresponds to a URDF link, positioned at its joint location.
Bones follow the URDF parent-child hierarchy with Limit Rotation constraints
derived from the URDF joint limits and axes.

No mesh parenting or binding — just the skeleton and constraints.
Run blender_franka_mesh.py first if you want the visual meshes in the scene.
"""

import math
import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Euler, Matrix, Vector

# =============================================================================
# CONFIGURATION
# =============================================================================

FRANKA_ASSET_DIR = (
    r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache"
    r"\newton-assets_franka_emika_panda_c9e36853\franka_emika_panda"
)
FRANKA_URDF = os.path.join(FRANKA_ASSET_DIR, "urdf", "fr3_franka_hand.urdf")


# =============================================================================
# URDF PARSING
# =============================================================================


def parse_urdf(urdf_path):
    """Parse URDF into joint list with full kinematic data."""
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

        mimic_elem = j.find("mimic")
        mimic = mimic_elem.get("joint") if mimic_elem is not None else None

        joints.append(
            {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent_link": j.find("parent").get("link"),
                "child_link": j.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "lower": lower,
                "upper": upper,
                "mimic": mimic,
            }
        )
    return joints


def compute_link_transforms(joints):
    """Compute world-space 4x4 transform for each link at zero config."""
    transforms = {"base": Matrix.Identity(4)}
    for j in joints:
        parent_T = transforms.get(j["parent_link"], Matrix.Identity(4))
        T = Matrix.Translation(Vector(j["xyz"]))
        R = Euler((j["rpy"][0], j["rpy"][1], j["rpy"][2]), "XYZ").to_matrix().to_4x4()
        transforms[j["child_link"]] = parent_T @ T @ R
    return transforms


def compute_joint_transforms(joints):
    """Compute world-space transform of each joint frame.

    The joint frame is: parent_link_world @ Translation(xyz) @ Rotation(rpy).
    This is the frame the joint axis is defined in.
    """
    link_transforms = {"base": Matrix.Identity(4)}
    joint_transforms = {}
    for j in joints:
        parent_T = link_transforms.get(j["parent_link"], Matrix.Identity(4))
        T = Matrix.Translation(Vector(j["xyz"]))
        R = Euler((j["rpy"][0], j["rpy"][1], j["rpy"][2]), "XYZ").to_matrix().to_4x4()
        child_T = parent_T @ T @ R
        link_transforms[j["child_link"]] = child_T
        joint_transforms[j["name"]] = child_T
    return joint_transforms, link_transforms


# =============================================================================
# ARMATURE GENERATION
# =============================================================================

# Bone length for visualization (doesn't affect kinematics)
BONE_LENGTH = 0.05


def _joint_world_axis(joint_transform, axis):
    """Get the joint rotation axis in world space."""
    world_axis = joint_transform.to_3x3() @ Vector(axis)
    world_axis.normalize()
    return world_axis


def _determine_constraint_axis(bone_matrix, world_axis):
    """Determine which bone-local axis best aligns with the joint rotation axis.

    Returns the axis name ('X', 'Y', or 'Z') and whether it's inverted.
    """
    bone_rot = bone_matrix.to_3x3()
    best_axis = "Z"
    best_dot = 0.0
    for i, name in enumerate(("X", "Y", "Z")):
        bone_axis = Vector(bone_rot.col[i])
        dot = bone_axis.dot(world_axis)
        if abs(dot) > abs(best_dot):
            best_dot = dot
            best_axis = name
    return best_axis, best_dot < 0


def generate_armature():
    print("=" * 60)
    print("FRANKA FR3 — ARMATURE GENERATION")
    print("=" * 60)

    if not os.path.isfile(FRANKA_URDF):
        print(f"\nERROR: Franka URDF not found at:\n  {FRANKA_URDF}")
        return

    # Remove any existing Franka armature
    for obj in list(bpy.data.objects):
        if obj.type == "ARMATURE" and "Franka" in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)
    for arm in list(bpy.data.armatures):
        if "Franka" in arm.name:
            bpy.data.armatures.remove(arm)

    joints = parse_urdf(FRANKA_URDF)
    joint_transforms, link_transforms = compute_joint_transforms(joints)

    # Build lookup: child_link -> joint, parent_link -> list of joints
    child_to_joint = {}
    link_to_children = {}
    for j in joints:
        child_to_joint[j["child_link"]] = j
        link_to_children.setdefault(j["parent_link"], []).append(j)

    # Only movable joints get bones
    movable_joints = [j for j in joints if j["type"] in ("revolute", "prismatic")]

    # Create armature at world origin
    arm_data = bpy.data.armatures.new("Franka_Armature")
    arm_data.display_type = "OCTAHEDRAL"
    arm_obj = bpy.data.objects.new("Franka_Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    # -- Edit mode: create bones --
    bpy.ops.object.mode_set(mode="EDIT")

    bone_map = {}  # joint_name -> edit bone

    for j in movable_joints:
        jt = joint_transforms[j["name"]]
        world_pos = jt.translation
        world_axis = _joint_world_axis(jt, j["axis"])

        bone = arm_data.edit_bones.new(j["name"])
        bone.head = world_pos
        bone.tail = world_pos + world_axis * BONE_LENGTH

        # Ensure minimum length
        if (bone.tail - bone.head).length < 0.01:
            bone.tail = bone.head + Vector((0, 0, 0.02))

        bone_map[j["name"]] = bone

    # Set parent-child relationships following URDF chain
    # For each movable joint, find its nearest movable ancestor
    for j in movable_joints:
        # Walk up the URDF tree from this joint's parent_link
        # until we find a link that is the child_link of another movable joint
        search_link = j["parent_link"]
        parent_bone = None
        while search_link and search_link != "base":
            # Is this link the child of a movable joint?
            if search_link in child_to_joint:
                parent_j = child_to_joint[search_link]
                if parent_j["name"] in bone_map:
                    parent_bone = bone_map[parent_j["name"]]
                    break
            # Walk further up: find which joint has this link as child
            if search_link in child_to_joint:
                search_link = child_to_joint[search_link]["parent_link"]
            else:
                break

        if parent_bone is not None:
            bone_map[j["name"]].parent = parent_bone
            bone_map[j["name"]].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    # -- Pose mode: add constraints --
    bpy.ops.object.mode_set(mode="POSE")

    for j in movable_joints:
        pbone = arm_obj.pose.bones.get(j["name"])
        if pbone is None:
            continue

        # Store URDF metadata as custom properties
        pbone["joint_type"] = j["type"]
        pbone["urdf_axis"] = list(j["axis"])
        pbone["child_link"] = j["child_link"]
        pbone["parent_link"] = j["parent_link"]
        if j["mimic"]:
            pbone["mimic_joint"] = j["mimic"]

        if j["type"] == "revolute":
            # Determine which bone-local axis the URDF joint axis maps to
            bone_world_matrix = arm_obj.matrix_world @ pbone.bone.matrix_local
            world_axis = _joint_world_axis(joint_transforms[j["name"]], j["axis"])
            constraint_axis, inverted = _determine_constraint_axis(bone_world_matrix, world_axis)

            lower = j["lower"]
            upper = j["upper"]
            if inverted:
                lower, upper = -upper, -lower

            # Add Limit Rotation constraint
            c = pbone.constraints.new("LIMIT_ROTATION")
            c.name = f"URDF_{j['name']}"
            c.owner_space = "LOCAL"

            if constraint_axis == "X":
                c.use_limit_x = True
                c.min_x = lower
                c.max_x = upper
            elif constraint_axis == "Y":
                c.use_limit_y = True
                c.min_y = lower
                c.max_y = upper
            else:  # Z
                c.use_limit_z = True
                c.min_z = lower
                c.max_z = upper

            pbone["constraint_axis"] = constraint_axis
            pbone["limit_lower_rad"] = round(lower, 4)
            pbone["limit_upper_rad"] = round(upper, 4)
            pbone["limit_lower_deg"] = round(math.degrees(lower), 1)
            pbone["limit_upper_deg"] = round(math.degrees(upper), 1)

            print(
                f"  {j['name']:25s} revolute  axis={constraint_axis}  "
                f"[{math.degrees(lower):+.1f}, {math.degrees(upper):+.1f}] deg"
            )

        elif j["type"] == "prismatic":
            # Add Limit Location constraint for prismatic joints
            bone_world_matrix = arm_obj.matrix_world @ pbone.bone.matrix_local
            world_axis = _joint_world_axis(joint_transforms[j["name"]], j["axis"])
            constraint_axis, inverted = _determine_constraint_axis(bone_world_matrix, world_axis)

            lower = j["lower"]
            upper = j["upper"]
            if inverted:
                lower, upper = -upper, -lower

            pbone["constraint_axis"] = constraint_axis
            pbone["limit_lower_m"] = round(lower, 4)
            pbone["limit_upper_m"] = round(upper, 4)

            print(f"  {j['name']:25s} prismatic axis={constraint_axis}  [{lower * 1000:+.1f}, {upper * 1000:+.1f}] mm")

    bpy.ops.object.mode_set(mode="OBJECT")

    rev_count = sum(1 for j in movable_joints if j["type"] == "revolute")
    pri_count = sum(1 for j in movable_joints if j["type"] == "prismatic")
    print(f"""
{"=" * 60}
ARMATURE READY — Franka FR3
{"=" * 60}

  Franka_Armature at world origin (0, 0, 0)
  {len(movable_joints)} bones ({rev_count} revolute + {pri_count} prismatic)
  Limit Rotation constraints from URDF joint limits
  Custom properties: joint_type, urdf_axis, limits, child_link

  Each bone is at the URDF joint position with tail along the rotation axis.
  Bone hierarchy follows the URDF kinematic chain.
""")


if __name__ == "__main__":
    generate_armature()
