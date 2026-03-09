"""
Blender LEAP Hand — Mesh Import Only
======================================
By David Clabaugh with Claude Code

Imports the LEAP dexterous hand (left) meshes at zero-config positions,
standing upright at the scene origin. No arm attached. 16 DOF.

NO armature is created — run blender_leap_armature.py separately.
"""

import json
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


def setup_meshes():
    print("=" * 65)
    print("LEAP HAND (standalone) — MESH IMPORT (no armature)")
    print("=" * 65)

    if not os.path.isfile(LEAP_MJCF):
        print(f"\nERROR: LEAP MJCF not found at:\n  {LEAP_MJCF}")
        print("Run first:  uv run python scripts/download_robot_assets.py")
        return

    print("\n[1/3] Cleaning scene...")
    utils.clean_scene()

    print("[2/3] Setting up scene...")
    utils.setup_camera_and_lights()

    print("[3/3] Importing LEAP hand meshes...")
    hand_bodies, hand_mesh_assets = utils.parse_mjcf(LEAP_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))

    hand_root, hand_imported = utils.import_mjcf_robot_meshes(
        hand_bodies, hand_mesh_assets, hand_transforms, base_matrix,
        root_name="LEAP_Root",
        material_color=(0.85, 0.75, 0.65, 1.0),
    )
    utils.create_mjcf_joint_empties(hand_bodies, hand_transforms, base_matrix, hand_root)

    total_meshes = sum(len(v) for v in hand_imported.values())
    print(f"""
{"=" * 65}
MESHES READY — LEAP Hand (standalone, 16-DOF)
{"=" * 65}

  LEAP_Root at origin (0, 0, 0)
  {total_meshes} mesh objects imported at zero-config

  NEXT STEP: Run blender_leap_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
