"""
Blender Shadow DEX-EE — Mesh Import Only
==========================================
By David Clabaugh with Claude Code

Imports the Shadow DEX-EE dexterous hand meshes at zero-config positions,
standing upright at the scene origin. No arm attached. 20+ DOF.

NO armature is created — run blender_shadow_armature.py separately.
"""

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

import robot_blender_utils as utils
from mathutils import Matrix, Vector

_ASSET_PATHS = utils.load_asset_paths()
SHADOW_ASSET_DIR = _ASSET_PATHS.get("shadow_dexee", "")
if not SHADOW_ASSET_DIR:
    SHADOW_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_shadow_dexee_XXXXXXXX\shadow_dexee"
SHADOW_MJCF = os.path.join(SHADOW_ASSET_DIR, "scene.xml")
HAND_BASE_POS = (0.0, 0.0, 0.0)


def setup_meshes():
    print("=" * 65)
    print("SHADOW DEX-EE (standalone) — MESH IMPORT (no armature)")
    print("=" * 65)

    if not os.path.isfile(SHADOW_MJCF):
        print(f"\nERROR: Shadow MJCF not found at:\n  {SHADOW_MJCF}")
        print("Run first:  uv run python scripts/download_robot_assets.py")
        return

    print("\n[1/3] Cleaning scene...")
    utils.clean_scene()

    print("[2/3] Setting up scene...")
    utils.setup_camera_and_lights()

    print("[3/3] Importing Shadow DEX-EE meshes...")
    hand_bodies, hand_mesh_assets = utils.parse_mjcf(SHADOW_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))

    hand_root, hand_imported = utils.import_mjcf_robot_meshes(
        hand_bodies,
        hand_mesh_assets,
        hand_transforms,
        base_matrix,
        root_name="Shadow_Root",
        material_color=(0.2, 0.2, 0.25, 1.0),
    )
    utils.create_mjcf_joint_empties(hand_bodies, hand_transforms, base_matrix, hand_root)

    total_meshes = sum(len(v) for v in hand_imported.values())
    print(f"""
{"=" * 65}
MESHES READY — Shadow DEX-EE (standalone, 20+ DOF)
{"=" * 65}

  Shadow_Root at origin (0, 0, 0)
  {total_meshes} mesh objects imported at zero-config

  NEXT STEP: Run blender_shadow_armature.py to generate the rig.
""")


if __name__ == "__main__":
    setup_meshes()
