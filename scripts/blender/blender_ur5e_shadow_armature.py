"""
Blender UR5e + Shadow DEX-EE — Armature Generation Only
=========================================================
By David Clabaugh with Claude Code

Generates armature for UR5e + Shadow already in the scene.
Run blender_ur5e_shadow_mesh.py FIRST.
"""

import math
import os
import sys

def _get_script_dir():
    """Resolve the directory containing this script, even inside Blender's text editor."""
    # 1. Normal Python execution (__file__ is defined)
    if "__file__" in dir():
        return os.path.dirname(os.path.abspath(__file__))
    # Blender text editor fallbacks:
    import bpy
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

SCRIPT_DIR = _get_script_dir()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy
from mathutils import Matrix, Quaternion, Vector

import robot_blender_utils as utils

_ASSET_PATHS = utils.load_asset_paths()
UR5E_ASSET_DIR = _ASSET_PATHS.get("universal_robots_ur5e", "")
SHADOW_ASSET_DIR = _ASSET_PATHS.get("shadow_dexee", "")
if not UR5E_ASSET_DIR:
    UR5E_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_universal_robots_ur5e_XXXXXXXX\universal_robots_ur5e"
if not SHADOW_ASSET_DIR:
    SHADOW_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_shadow_dexee_XXXXXXXX\shadow_dexee"
UR5E_MJCF = os.path.join(UR5E_ASSET_DIR, "ur5e.xml")
SHADOW_MJCF = os.path.join(SHADOW_ASSET_DIR, "scene.xml")
HAND_XFORM_POS = [0.0, 0.1, 0.0]
HAND_XFORM_ROT_AXIS = [1.0, 0.0, 0.0]
HAND_XFORM_ROT_ANGLE = -1.5708
EE_BODY_NAME = "wrist_3_link"
ROBOT_BASE_POS = utils.NEWTON_SCENE["robot_base"]


def generate_armature():
    print("=" * 65)
    print("UR5e + SHADOW DEX-EE — ARMATURE GENERATION")
    print("=" * 65)

    arm_root = bpy.data.objects.get("UR5e_Root")
    if arm_root is None:
        print("\nERROR: UR5e_Root not found. Run blender_ur5e_shadow_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and "Shadow" in obj.name:
            bpy.data.objects.remove(obj, do_unlink=True)

    arm_bodies, _ = utils.parse_mjcf(UR5E_MJCF)
    arm_transforms = utils.compute_mjcf_fk(arm_bodies)
    hand_bodies, _ = utils.parse_mjcf(SHADOW_MJCF)

    base_pos = Vector(ROBOT_BASE_POS)
    base_rot = Quaternion((0, 0, 1), math.pi)
    base_matrix = Matrix.Translation(base_pos) @ base_rot.to_matrix().to_4x4()

    hand_quat = Quaternion(Vector(HAND_XFORM_ROT_AXIS), HAND_XFORM_ROT_ANGLE)
    hand_base_world = utils.compute_hand_base_transform(
        arm_transforms, EE_BODY_NAME, HAND_XFORM_POS, hand_quat, base_matrix
    )
    hand_world_transforms = utils.compute_hand_fk_in_world(hand_bodies, hand_base_world)

    arm_chain = utils._collect_joint_chain_mjcf(arm_bodies, arm_transforms, base_matrix)
    hand_chain = utils._collect_joint_chain_mjcf(hand_bodies, hand_world_transforms, Matrix.Identity(4))
    combined = arm_chain + hand_chain

    utils.create_armature("UR5e_Shadow_Armature", combined, arm_root)

    print(f"""
{"=" * 65}
ARMATURE READY — UR5e + Shadow DEX-EE
{"=" * 65}

  UR5e_Shadow_Armature with {len(combined)} bones ({len(arm_chain)} arm + {len(hand_chain)} hand)
  NOTE: Re-parent finger bones for proper branching hand topology.
""")


if __name__ == "__main__":
    generate_armature()
