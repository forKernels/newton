"""
Blender Robotiq 2F-85 — Mesh Import Only
==========================================
By David Clabaugh with Claude Code

Imports the Robotiq 2F-85 parallel jaw gripper meshes at zero-config,
standing upright at the scene origin. No arm attached.

NO armature is created — run blender_robotiq_armature.py separately.
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
ROBOTIQ_ASSET_DIR = _ASSET_PATHS.get("robotiq_2f85", "")
if not ROBOTIQ_ASSET_DIR:
    ROBOTIQ_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_robotiq_2f85_XXXXXXXX\robotiq_2f85"
ROBOTIQ_MJCF = os.path.join(ROBOTIQ_ASSET_DIR, "2f85.xml")
HAND_BASE_POS = (0.0, 0.0, 0.0)


def setup_meshes():
    print("=" * 65)
    print("ROBOTIQ 2F-85 (standalone) — MESH IMPORT (no armature)")
    print("=" * 65)

    if not os.path.isfile(ROBOTIQ_MJCF):
        print(f"\nERROR: Robotiq MJCF not found at:\n  {ROBOTIQ_MJCF}")
        print("Run first:  uv run python scripts/download_robot_assets.py")
        return

    print("\n[1/3] Cleaning scene...")
    utils.clean_scene()

    print("[2/3] Setting up scene...")
    utils.setup_camera_and_lights()

    print("[3/3] Importing Robotiq 2F-85 meshes...")
    hand_bodies, hand_mesh_assets = utils.parse_mjcf(ROBOTIQ_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))

    hand_root, hand_imported = utils.import_mjcf_robot_meshes(
        hand_bodies, hand_mesh_assets, hand_transforms, base_matrix,
        root_name="Robotiq_Root",
        material_color=(0.1, 0.1, 0.1, 1.0),
    )
    utils.create_mjcf_joint_empties(hand_bodies, hand_transforms, base_matrix, hand_root)

    total_meshes = sum(len(v) for v in hand_imported.values())
    print(f"""
{"=" * 65}
MESHES READY — Robotiq 2F-85 (standalone)
{"=" * 65}

  Robotiq_Root at origin (0, 0, 0)
  {total_meshes} mesh objects imported at zero-config

  NEXT STEP: Run blender_robotiq_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
