"""
Blender UR5e + LEAP Hand — Mesh Import Only
============================================
By David Clabaugh with Claude Code

Imports UR5e arm and LEAP dexterous hand (left) meshes at zero-config.
NO armature — run blender_ur5e_leap_armature.py separately.
"""

import json
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

DATASET_ROOT = r"D:\_blender\_myBlender\SimulationWork\ClothDataset\_Maria_Set"
GARMENT_OBJ = "dress_sleeveless_2550/dress_sleeveless_000YCTJ9HS/dress_sleeveless_000YCTJ9HS_sim.obj"
CLOTH_SCALE = 0.01

_ASSET_PATHS = utils.load_asset_paths()
UR5E_ASSET_DIR = _ASSET_PATHS.get("universal_robots_ur5e", "")
LEAP_ASSET_DIR = _ASSET_PATHS.get("leap_hand", "")
if not UR5E_ASSET_DIR:
    UR5E_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_universal_robots_ur5e_XXXXXXXX\universal_robots_ur5e"
if not LEAP_ASSET_DIR:
    LEAP_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_leap_hand_XXXXXXXX\leap_hand"
UR5E_MJCF = os.path.join(UR5E_ASSET_DIR, "ur5e.xml")
LEAP_MJCF = os.path.join(LEAP_ASSET_DIR, "left_hand.xml")

HAND_XFORM_POS = [-0.065, 0.28, 0.10]
EE_BODY_NAME = "wrist_3_link"
ROBOT_BASE_POS = utils.NEWTON_SCENE["robot_base"]


def _compute_leap_hand_quat():
    quat_z = Quaternion((0, 0, 1), math.pi / 2)
    quat_y = Quaternion((0, 1, 0), math.pi)
    return quat_y @ quat_z


def setup_meshes():
    print("=" * 65)
    print("UR5e + LEAP HAND (LEFT) — MESH IMPORT (no armature)")
    print("=" * 65)

    for label, path in [("UR5e MJCF", UR5E_MJCF), ("LEAP MJCF", LEAP_MJCF)]:
        if not os.path.isfile(path):
            print(f"\nERROR: {label} not found at:\n  {path}")
            return

    print("\n[1/6] Cleaning scene...")
    utils.clean_scene()
    print("[2/6] Setting up scene...")
    utils.setup_camera_and_lights()

    print("[3/6] Importing garment...")
    garment_bbox = None
    if GARMENT_OBJ and DATASET_ROOT:
        obj_path = os.path.join(DATASET_ROOT, GARMENT_OBJ)
        cloth = utils.import_garment(obj_path, cloth_scale=CLOTH_SCALE)
        if cloth and "bbox_size" in cloth:
            garment_bbox = cloth["bbox_size"]

    print("[4/6] Creating environment...")
    utils.create_ground()
    utils.create_table(garment_bbox=garment_bbox)

    base_pos = Vector(ROBOT_BASE_POS)
    base_rot = Quaternion((0, 0, 1), math.pi)
    base_matrix = Matrix.Translation(base_pos) @ base_rot.to_matrix().to_4x4()

    print("[5/6] Importing UR5e arm meshes...")
    arm_bodies, arm_mesh_assets = utils.parse_mjcf(UR5E_MJCF)
    arm_transforms = utils.compute_mjcf_fk(arm_bodies)
    arm_root, arm_imported = utils.import_mjcf_robot_meshes(
        arm_bodies, arm_mesh_assets, arm_transforms, base_matrix,
        root_name="UR5e_Root", material_color=(0.15, 0.15, 0.18, 1.0),
    )
    utils.create_mjcf_joint_empties(arm_bodies, arm_transforms, base_matrix, arm_root)

    print("\n[6/6] Importing LEAP hand meshes...")
    hand_bodies, hand_mesh_assets = utils.parse_mjcf(LEAP_MJCF)
    hand_quat = _compute_leap_hand_quat()
    hand_base_world = utils.compute_hand_base_transform(
        arm_transforms, EE_BODY_NAME, HAND_XFORM_POS, hand_quat, base_matrix
    )
    hand_world_transforms = utils.compute_hand_fk_in_world(hand_bodies, hand_base_world)
    hand_root, hand_imported = utils.import_mjcf_robot_meshes(
        hand_bodies, hand_mesh_assets, hand_world_transforms, Matrix.Identity(4),
        root_name="LEAP_Root", material_color=(0.85, 0.75, 0.65, 1.0),
    )
    hand_root.parent = arm_root
    hand_root.matrix_parent_inverse = arm_root.matrix_world.inverted()
    utils.create_mjcf_joint_empties(hand_bodies, hand_world_transforms, Matrix.Identity(4), arm_root)

    total = sum(len(v) for v in arm_imported.values()) + sum(len(v) for v in hand_imported.values())
    rx, ry, rz = ROBOT_BASE_POS
    print(f"""
{"=" * 65}
MESHES READY — UR5e + LEAP Hand (16-DOF)
{"=" * 65}

  UR5e_Root at ({rx}, {ry}, {rz}), {total} meshes

  NEXT STEP: Run blender_ur5e_leap_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
