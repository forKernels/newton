"""
Blender LEAP Hand — Armature Generation Only
==============================================
By David Clabaugh with Claude Code

Generates an armature with 16 bones for a standalone LEAP hand already in the scene.
Run blender_leap_mesh.py FIRST.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
if not SCRIPT_DIR:
    import bpy
    SCRIPT_DIR = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy
from mathutils import Matrix, Vector

import robot_blender_utils as utils

_ASSET_PATHS = utils.load_asset_paths()
LEAP_ASSET_DIR = _ASSET_PATHS.get("leap_hand", "")
if not LEAP_ASSET_DIR:
    LEAP_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_leap_hand_XXXXXXXX\leap_hand"
LEAP_MJCF = os.path.join(LEAP_ASSET_DIR, "left_hand.xml")
HAND_BASE_POS = (0.0, 0.0, 0.0)


def generate_armature():
    print("=" * 65)
    print("LEAP HAND (standalone) — ARMATURE GENERATION")
    print("=" * 65)

    hand_root = bpy.data.objects.get("LEAP_Root")
    if hand_root is None:
        print("\nERROR: LEAP_Root not found. Run blender_leap_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == "LEAP_Armature":
            bpy.data.objects.remove(obj, do_unlink=True)

    if not os.path.isfile(LEAP_MJCF):
        print(f"\nERROR: LEAP MJCF not found at:\n  {LEAP_MJCF}")
        return

    hand_bodies, _ = utils.parse_mjcf(LEAP_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))
    hand_chain = utils._collect_joint_chain_mjcf(hand_bodies, hand_transforms, base_matrix)
    utils.create_armature("LEAP_Armature", hand_chain, hand_root)

    print(f"""
{"=" * 65}
ARMATURE READY — LEAP Hand (standalone)
{"=" * 65}

  LEAP_Armature with {len(hand_chain)} bones (16 DOF, 4 finger chains)

  NOTE: Bone chain is linear — re-parent finger bones to palm for proper rig.
  Rigging: Select mesh > Shift-select armature > Ctrl+P > Bone
""")


if __name__ == "__main__":
    generate_armature()
