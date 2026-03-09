"""
Blender Shadow DEX-EE — Armature Generation Only
==================================================
By David Clabaugh with Claude Code

Generates an armature for a standalone Shadow DEX-EE hand already in the scene.
Run blender_shadow_mesh.py FIRST.
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

import bpy
from mathutils import Matrix, Vector

import robot_blender_utils as utils

_ASSET_PATHS = utils.load_asset_paths()
SHADOW_ASSET_DIR = _ASSET_PATHS.get("shadow_dexee", "")
if not SHADOW_ASSET_DIR:
    SHADOW_ASSET_DIR = r"C:\Users\DiscoKid\AppData\Local\newton-physics\newton\Cache\mujoco_menagerie_shadow_dexee_XXXXXXXX\shadow_dexee"
SHADOW_MJCF = os.path.join(SHADOW_ASSET_DIR, "scene.xml")
HAND_BASE_POS = (0.0, 0.0, 0.0)


def generate_armature():
    print("=" * 65)
    print("SHADOW DEX-EE (standalone) — ARMATURE GENERATION")
    print("=" * 65)

    hand_root = bpy.data.objects.get("Shadow_Root")
    if hand_root is None:
        print("\nERROR: Shadow_Root not found. Run blender_shadow_mesh.py first.")
        return

    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name == "Shadow_Armature":
            bpy.data.objects.remove(obj, do_unlink=True)

    if not os.path.isfile(SHADOW_MJCF):
        print(f"\nERROR: Shadow MJCF not found at:\n  {SHADOW_MJCF}")
        return

    hand_bodies, _ = utils.parse_mjcf(SHADOW_MJCF)
    hand_transforms = utils.compute_mjcf_fk(hand_bodies)
    base_matrix = Matrix.Translation(Vector(HAND_BASE_POS))
    hand_chain = utils._collect_joint_chain_mjcf(hand_bodies, hand_transforms, base_matrix)
    utils.create_armature("Shadow_Armature", hand_chain, hand_root)

    print(f"""
{"=" * 65}
ARMATURE READY — Shadow DEX-EE (standalone)
{"=" * 65}

  Shadow_Armature with {len(hand_chain)} bones (20+ DOF)

  NOTE: Re-parent finger bones for proper branching hand topology.
  Rigging: Select mesh > Shift-select armature > Ctrl+P > Bone
""")


if __name__ == "__main__":
    generate_armature()
