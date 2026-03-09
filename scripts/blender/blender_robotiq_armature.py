"""
Blender Robotiq 2F-85 — Armature Generation Only
==================================================
By David Clabaugh with Claude Code

Generates an armature for a standalone Robotiq 2F-85 gripper already in the scene.
Run blender_robotiq_mesh.py FIRST.
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
ROBOTIQ_ASSET_DIR = _ASSET_PATHS.get("robotiq_2f85", "")
if not ROBOTIQ_ASSET_DIR:
    ROBOTIQ_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_robotiq_2f85_XXXXXXXX\robotiq_2f85"
ROBOTIQ_MJCF = os.path.join(ROBOTIQ_ASSET_DIR, "2f85.xml")
HAND_BASE_POS = (0.0, 0.0, 0.0)


def generate_armature():
    print("=" * 65)
    print("ROBOTIQ 2F-85 (standalone) — ARMATURE GENERATION")
    print("=" * 65)

    hand_root = bpy.data.objects.get("Robotiq_Root")
    if hand_root is None:
        print("\nERROR: Robotiq_Root not found. Run blender_robotiq_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == "Robotiq_Armature":
            bpy.data.objects.remove(obj, do_unlink=True)

    if not os.path.isfile(ROBOTIQ_MJCF):
        print(f"\nERROR: Robotiq MJCF not found at:\n  {ROBOTIQ_MJCF}")
        return

    hand_bodies, _ = utils.parse_mjcf(ROBOTIQ_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))
    hand_chain = utils._collect_joint_chain_mjcf(hand_bodies, hand_transforms, base_matrix)
    utils.create_armature("Robotiq_Armature", hand_chain, hand_root)

    print(f"""
{"=" * 65}
ARMATURE READY — Robotiq 2F-85 (standalone)
{"=" * 65}

  Robotiq_Armature with {len(hand_chain)} bones

  Rigging: Select mesh > Shift-select armature > Ctrl+P > Bone
""")


if __name__ == "__main__":
    generate_armature()
