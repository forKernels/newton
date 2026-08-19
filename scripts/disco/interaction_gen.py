"""
newton_zhao_corp :: interaction_gen.py
Procedural physics-based interaction generator for Blender (headless or GUI).

Usage:
    # Pre-rigged scene (George.rig already in .blend)
    blender -b base_scene.blend -P interaction_gen.py -- --rig George.rig --interaction knockover --seeds 0-49

    # Import CC5/AccuRig FBX on the fly (auto-wires IK constraints)
    blender -b base_scene.blend -P interaction_gen.py -- --import-fbx rigged_male.fbx --rig rigged_male --interaction grasp_lift --seeds 0-19

    # Batch mode
    blender -b base_scene.blend -P interaction_gen.py -- --batch batch_config.json

Supported rigs:
    - George.rig (Rig_HumanMan.blend) — pre-wired IK
    - CC5/AccuRig FBX exports — IK auto-wired on import via --import-fbx
    - Any rig added to BONE_MAPS with IK targets
"""

import json
import math
import os
import random
import sys

import bpy
from mathutils import Euler, Vector

# ---------------------------------------------------------------------------
# STANDARD NAMING CONVENTION
# ---------------------------------------------------------------------------
# All rigs must follow this naming convention for their control bones.
# When building a control rig, name the controllers using these prefixes:
#
#   CTRL_{canonical_name}
#
# Example: CTRL_hand_r, CTRL_hand_l, CTRL_foot_r, CTRL_spine, CTRL_head
#
# The pipeline will look for controllers first (CTRL_ prefix), then fall
# back to deform/FK bones if no controller is found.
#
# Required canonical names for humanoid rigs:
#   hand_r, hand_l           — hand IK targets
#   foot_r, foot_l           — foot IK targets
#   elbow_r, elbow_l         — elbow pole targets (optional)
#   knee_r, knee_l           — knee pole targets (optional)
#   upper_arm_r, upper_arm_l — FK upper arm
#   forearm_r, forearm_l     — FK forearm
#   wrist_r, wrist_l         — FK wrist
#   spine, head              — torso
#   grip_master_r, grip_master_l — master grip controller (optional)
#   fingers_r, fingers_l     — list of first-knuckle bones

STANDARD_CTRL_PREFIX = "CTRL_"


def resolve_bone_name(armature_obj, canonical_name, bone_map):
    """Resolve a canonical bone name to an actual bone in the armature.

    Lookup order:
        1. CTRL_{canonical_name}  (standard controller)
        2. bone_map[canonical_name]  (rig-specific mapping)
        3. canonical_name  (direct match)

    Returns the actual bone name string, or None if not found.
    """
    bone_names = {b.name for b in armature_obj.data.bones}

    ctrl_name = f"{STANDARD_CTRL_PREFIX}{canonical_name}"
    if ctrl_name in bone_names:
        return ctrl_name

    mapped = bone_map.get(canonical_name)
    if isinstance(mapped, str) and mapped in bone_names:
        return mapped

    if canonical_name in bone_names:
        return canonical_name

    return None


def resolve_bone_list(armature_obj, canonical_name, bone_map):
    """Resolve a canonical name that maps to a list of bones.

    Lookup order per entry:
        1. CTRL_{entry}
        2. entry as-is

    Returns list of actual bone name strings found in the armature.
    """
    bone_names = {b.name for b in armature_obj.data.bones}
    mapped = bone_map.get(canonical_name, [])
    if not isinstance(mapped, list):
        return []

    result = []
    for entry in mapped:
        ctrl_name = f"{STANDARD_CTRL_PREFIX}{entry}"
        if ctrl_name in bone_names:
            result.append(ctrl_name)
        elif entry in bone_names:
            result.append(entry)
    return result


def validate_rig_naming(armature_obj, bone_map):
    """Check a rig against the standard naming convention.

    Returns a dict with 'valid', 'controllers_found', 'fallbacks_used',
    and 'missing' lists for debugging.
    """
    required = ["hand_r", "hand_l", "foot_r", "foot_l", "spine", "head"]
    optional = ["elbow_r", "elbow_l", "knee_r", "knee_l", "wrist_r", "wrist_l", "grip_master_r", "grip_master_l"]

    report = {"controllers_found": [], "fallbacks_used": [], "missing": []}

    bone_names = {b.name for b in armature_obj.data.bones}

    for canon in required + optional:
        ctrl_name = f"{STANDARD_CTRL_PREFIX}{canon}"
        if ctrl_name in bone_names:
            report["controllers_found"].append(canon)
        elif resolve_bone_name(armature_obj, canon, bone_map):
            report["fallbacks_used"].append(canon)
        elif canon in required:
            report["missing"].append(canon)

    report["valid"] = len(report["missing"]) == 0
    return report


# ---------------------------------------------------------------------------
# BONE MAPS
# ---------------------------------------------------------------------------
# Map canonical names -> actual bone names per rig.
# The pipeline checks for CTRL_ prefixed controllers first, then falls back
# to these mappings.

BONE_MAPS = {
    # Rig_HumanMan.blend — armature object "George.rig" (186 bones)
    # Full IK setup: hand/foot/elbow/knee IK targets + master grip controls
    "George.rig": {
        "type": "humanoid",
        "needs_ik_setup": False,
        # IK targets (root-level bones that drive the chains)
        "hand_r": "right_hand_ik",
        "hand_l": "left_hand_ik",
        "foot_r": "right_foot_ik",
        "foot_l": "left_foot_ik",
        "elbow_r": "right_elbow_ik",
        "elbow_l": "left_elbow_ik",
        "knee_r": "right_knee_ik",
        "knee_l": "left_knee_ik",
        # FK chain bones (for direct rotation if needed)
        "upper_arm_r": "upperarm01.R",
        "upper_arm_l": "upperarm01.L",
        "forearm_r": "lowerarm01.R",
        "forearm_l": "lowerarm01.L",
        "wrist_r": "wrist.R",
        "wrist_l": "wrist.L",
        # Spine / head
        "spine": "spine01",
        "head": "head",
        # Grip controls — rotate these to close/open hands
        "grip_master_r": "right_master_grip",
        "grip_master_l": "left_master_grip",
        # Per-finger grip bones (COPY_ROTATION from master)
        "finger_grips_r": [
            "right_finger1_grip",
            "right_finger2_grip",
            "right_finger3_grip",
            "right_finger4_grip",
            "right_finger5_grip",
        ],
        "finger_grips_l": [
            "left_finger1_grip",
            "left_finger2_grip",
            "left_finger3_grip",
            "left_finger4_grip",
            "left_finger5_grip",
        ],
        # First knuckle of each finger (for direct FK if needed)
        "fingers_r": [
            "finger1-1.R",
            "finger2-1.R",
            "finger3-1.R",
            "finger4-1.R",
            "finger5-1.R",
        ],
        "fingers_l": [
            "finger1-1.L",
            "finger2-1.L",
            "finger3-1.L",
            "finger4-1.L",
            "finger5-1.L",
        ],
    },
}

# ---------------------------------------------------------------------------
# CC5 / AccuRig TEMPLATE — shared by all Character Creator 5 exports
# ---------------------------------------------------------------------------
# AccuRig FBX exports have IK target bones but NO IK constraints wired up.
# setup_accurig_ik() adds them automatically after FBX import.

ACCURIG_BONE_MAP = {
    "type": "humanoid",
    "needs_ik_setup": True,
    # IK targets (root-level orphan bones — need constraints wired)
    "hand_r": "ik_hand_r",
    "hand_l": "ik_hand_l",
    "foot_r": "ik_foot_r",
    "foot_l": "ik_foot_l",
    # FK chain bones
    "upper_arm_r": "upperarm_r",
    "upper_arm_l": "upperarm_l",
    "forearm_r": "lowerarm_r",
    "forearm_l": "lowerarm_l",
    "wrist_r": "hand_r",
    "wrist_l": "hand_l",
    "thigh_r": "thigh_r",
    "thigh_l": "thigh_l",
    "calf_r": "calf_r",
    "calf_l": "calf_l",
    # Spine / head
    "spine": "spine_01",
    "head": "head",
    # No master grip bone — use per-finger FK rotation for grasping
    "fingers_r": [
        "thumb_01_r",
        "index_01_r",
        "middle_01_r",
        "ring_01_r",
        "pinky_01_r",
    ],
    "fingers_l": [
        "thumb_01_l",
        "index_01_l",
        "middle_01_l",
        "ring_01_l",
        "pinky_01_l",
    ],
    # IK chain wiring config (used by setup_accurig_ik)
    "_ik_chains": {
        # target bone: (chain tip bone, chain length, pole target bone or None)
        "ik_hand_r": ("hand_r", 3, None),  # hand_r <- lowerarm_r <- upperarm_r
        "ik_hand_l": ("hand_l", 3, None),
        "ik_foot_r": ("foot_r", 3, None),  # foot_r <- calf_r <- thigh_r
        "ik_foot_l": ("foot_l", 3, None),
    },
}

# AccuRig characters imported via FBX get armature named "Armature".
# If you rename in Blender, update the key here.
ACCURIG_RIG_NAMES = ["rigged_male", "rigged_female"]


def _register_accurig_rigs():
    """Register all known AccuRig rigs with the shared bone map."""
    for name in ACCURIG_RIG_NAMES:
        BONE_MAPS[name] = dict(ACCURIG_BONE_MAP)


_register_accurig_rigs()

# ---------------------------------------------------------------------------
# AccuRig IK SETUP
# ---------------------------------------------------------------------------


def setup_accurig_ik(armature_obj, bone_map):
    """Wire IK constraints on AccuRig FBX imports.

    AccuRig exports include ik_hand_r/l and ik_foot_r/l bones but no
    IK constraints. This adds them so set_loc() on IK targets drives
    the arm/leg chains.
    """
    ik_chains = bone_map.get("_ik_chains", {})
    if not ik_chains:
        return

    for target_bone, (tip_bone, chain_len, pole_bone) in ik_chains.items():
        pb = armature_obj.pose.bones.get(tip_bone)
        if pb is None:
            print(f"[interaction_gen] WARNING: bone '{tip_bone}' not found, skipping IK setup")
            continue

        # Check if IK constraint already exists
        has_ik = any(c.type == "IK" for c in pb.constraints)
        if has_ik:
            continue

        ik = pb.constraints.new("IK")
        ik.target = armature_obj
        ik.subtarget = target_bone
        ik.chain_count = chain_len
        if pole_bone:
            ik.pole_target = armature_obj
            ik.pole_subtarget = pole_bone

        print(f"[interaction_gen] IK: {tip_bone} -> {target_bone} (chain={chain_len})")

    print(f"[interaction_gen] AccuRig IK setup complete for '{armature_obj.name}'")


def import_accurig_fbx(fbx_path: str, rig_name: str = None):
    """Import a CC5/AccuRig FBX and set up IK constraints.

    Args:
        fbx_path: Path to the .fbx file.
        rig_name: Name to register in BONE_MAPS. If None, uses fbx filename stem.

    Returns:
        The armature object name (use this as --rig argument).
    """
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find the imported armature (FBX importer names it "Armature" by default)
    armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            # Check for AccuRig IK bones
            bone_names = {b.name for b in obj.data.bones}
            if "ik_hand_r" in bone_names and "ik_foot_r" in bone_names:
                armature = obj
                break

    if armature is None:
        raise RuntimeError(f"No AccuRig armature found after importing {fbx_path}")

    # Rename armature to something meaningful if it's generic
    if rig_name is None:
        rig_name = os.path.splitext(os.path.basename(fbx_path))[0]

    if armature.name != rig_name:
        armature.name = rig_name

    # Register bone map and wire IK
    if rig_name not in BONE_MAPS:
        BONE_MAPS[rig_name] = dict(ACCURIG_BONE_MAP)
    setup_accurig_ik(armature, BONE_MAPS[rig_name])

    print(f"[interaction_gen] Imported AccuRig rig '{rig_name}' from {fbx_path}")
    return rig_name


# ---------------------------------------------------------------------------
# RIG ACCESSOR
# ---------------------------------------------------------------------------


class RigController:
    """Wraps a rig object with its bone map for interaction scripts.

    Bone resolution order:
        1. CTRL_{canonical_name} — standard control rig controller
        2. bone_map[canonical_name] — rig-specific fallback
        3. canonical_name — direct match
    """

    def __init__(self, rig_name: str):
        self.rig = bpy.data.objects[rig_name]
        if rig_name not in BONE_MAPS:
            raise ValueError(
                f"No bone map for '{rig_name}'. "
                f"Available rigs: {list(BONE_MAPS.keys())}. "
                f"Add it to BONE_MAPS if it has IK targets."
            )
        self.map = BONE_MAPS[rig_name]
        self.rig_type = self.map["type"]

        # Auto-setup IK for AccuRig rigs on first use
        if self.map.get("needs_ik_setup"):
            setup_accurig_ik(self.rig, self.map)
            self.map["needs_ik_setup"] = False

        # Validate naming and log results
        report = validate_rig_naming(self.rig, self.map)
        if report["controllers_found"]:
            print(f"[interaction_gen] CTRL_ controllers found: {report['controllers_found']}")
        if report["fallbacks_used"]:
            print(f"[interaction_gen] Fallback bones used: {report['fallbacks_used']}")
        if report["missing"]:
            print(f"[interaction_gen] WARNING: missing required bones: {report['missing']}")

    def _resolve(self, canonical_name: str) -> str:
        """Resolve canonical name to actual bone name via standard lookup order."""
        actual = resolve_bone_name(self.rig, canonical_name, self.map)
        if actual is None:
            raise KeyError(
                f"Bone '{canonical_name}' not found in rig '{self.rig.name}'. "
                f"Expected CTRL_{canonical_name} or a bone_map entry."
            )
        return actual

    def _resolve_list(self, canonical_name: str) -> list:
        """Resolve canonical name to a list of actual bone names."""
        return resolve_bone_list(self.rig, canonical_name, self.map)

    def bone(self, canonical_name: str):
        """Get pose bone by canonical name."""
        mapped = self.map.get(canonical_name)
        if isinstance(mapped, list):
            names = self._resolve_list(canonical_name)
            return [self.rig.pose.bones[n] for n in names]
        actual = self._resolve(canonical_name)
        return self.rig.pose.bones[actual]

    def bone_name(self, canonical_name: str) -> str:
        """Get the resolved actual bone name string (not the pose bone object)."""
        return self._resolve(canonical_name)

    def set_loc(self, canonical_name: str, loc: tuple, frame: int):
        """Set bone location in WORLD space. Auto-converts to bone-local offset.

        Handles both identity and non-identity armature transforms.
        """
        b = self.bone(canonical_name)
        world_pos = Vector(loc)
        arm_local = self.rig.matrix_world.inverted() @ world_pos
        b.location = arm_local - b.bone.head_local
        b.keyframe_insert("location", frame=frame)

    def set_rot(self, canonical_name: str, rot_euler: tuple, frame: int):
        b = self.bone(canonical_name)
        b.rotation_euler = Euler(rot_euler)
        b.keyframe_insert("rotation_euler", frame=frame)

    def is_robotic(self):
        return self.rig_type == "robotic"

    def is_humanoid(self):
        return self.rig_type == "humanoid"

    def has_bone(self, canonical_name: str):
        """Check if a canonical bone exists (controller or fallback)."""
        if isinstance(self.map.get(canonical_name), list):
            return len(self._resolve_list(canonical_name)) > 0
        return resolve_bone_name(self.rig, canonical_name, self.map) is not None

    def close_grip(self, hand: str, amount: float, frame: int):
        """Close grip via master grip bone or per-finger FK rotation.

        Args:
            hand: "r" or "l"
            amount: 0.0 (open) to 1.0 (fully closed)
            frame: keyframe
        """
        grip_key = f"grip_master_{hand}"
        if self.has_bone(grip_key):
            grip_rot = (amount * 1.2, 0, 0)
            self.set_rot(grip_key, grip_rot, frame)
        else:
            finger_key = f"fingers_{hand}"
            if self.has_bone(finger_key):
                fingers = self.bone(finger_key)
                curl = amount * 1.2
                for fb in fingers:
                    fb.rotation_euler = Euler((curl, 0, 0))
                    fb.keyframe_insert("rotation_euler", frame=frame)


# ---------------------------------------------------------------------------
# SCENE UTILITIES
# ---------------------------------------------------------------------------


def clear_keyframes(obj):
    """Remove all animation data from object."""
    if obj.animation_data:
        obj.animation_data_clear()


def clear_physics_cache():
    """Free all physics bakes."""
    for scene in bpy.data.scenes:
        if scene.rigidbody_world:
            for pt in scene.rigidbody_world.point_cache.point_caches:
                pt.frame_start = 1
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass


def set_rigid_body_kinematic(obj, kinematic: bool, frame: int):
    """Toggle kinematic state of a rigid body at a specific frame."""
    obj.rigid_body.kinematic = kinematic
    obj.keyframe_insert("rigid_body.kinematic", frame=frame)


def add_child_of_constraint(child_obj, parent_obj, bone_name: str, frame_on: int):
    """Add a Child Of constraint that activates at frame_on."""
    c = child_obj.constraints.new("CHILD_OF")
    c.target = parent_obj
    c.subtarget = bone_name
    c.influence = 0
    c.keyframe_insert("influence", frame=frame_on - 1)
    c.influence = 1
    c.keyframe_insert("influence", frame=frame_on)
    return c


def randomize_table_objects(obj_names: list, seed: int, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), z_base=0.8):
    """Scatter named objects randomly on a table surface."""
    random.seed(seed)
    for name in obj_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.location = Vector(
            (
                random.uniform(*x_range),
                random.uniform(*y_range),
                z_base,
            )
        )
        obj.rotation_euler = Euler((0, 0, random.uniform(0, math.tau)))


# ---------------------------------------------------------------------------
# INTERACTIONS
# ---------------------------------------------------------------------------


def knockover(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Arm sweeps across table, knocking objects over.
    Works with both robotic and humanoid rigs.

    Kwargs:
        hand: "r" or "l" (default "r")
        table_y: Y center of the table in world space (default 0.4)
        table_z: Z of the table top surface (default 0.77)
        sweep_frames: duration of the sweep in frames (default 30-50)
        start_frame: frame when sweep begins (default 30, gives objects time to settle)
        settle_frames: extra frames after sweep for physics to play out (default 60)
        skip_randomize: if True, don't re-scatter objects (default False)
    """
    random.seed(seed)
    sweep_hand = kwargs.get("hand", "r")
    table_y = kwargs.get("table_y", 0.4)
    table_z = kwargs.get("table_z", 0.77)
    sweep_frames = kwargs.get("sweep_frames", random.randint(30, 50))
    start_frame = kwargs.get("start_frame", 30)
    settle_frames = kwargs.get("settle_frames", 60)
    skip_randomize = kwargs.get("skip_randomize", False)

    z_height = table_z + 0.08  # just above table surface

    if not skip_randomize:
        randomize_table_objects(
            objects,
            seed,
            x_range=(-0.25, 0.25),
            y_range=(table_y - 0.12, table_y + 0.12),
            z_base=table_z + 0.03,
        )

    if rc.is_robotic():
        sweep_angle = random.uniform(0.4, 1.0)
        rc.set_rot("base", (0, 0, -sweep_angle / 2), start_frame)
        rc.set_rot("base", (0, 0, sweep_angle / 2), start_frame + sweep_frames)
    else:
        hand = f"hand_{sweep_hand}"

        # Approach: hand starts to the side, at table height
        start_x = random.uniform(-0.35, -0.2)
        end_x = random.uniform(0.25, 0.4)
        y_reach = table_y + random.uniform(-0.05, 0.1)

        # Keyframe 1: resting pose (arm at side)
        rc.set_loc(hand, (start_x - 0.1, table_y - 0.2, z_height + 0.15), 1)

        # Keyframe 2: hand moves to sweep start position
        rc.set_loc(hand, (start_x, y_reach, z_height), start_frame)

        # Keyframe 3: sweep across table
        rc.set_loc(hand, (end_x, y_reach, z_height), start_frame + sweep_frames)

        # Keyframe 4: hand lifts away
        rc.set_loc(hand, (end_x + 0.1, y_reach - 0.1, z_height + 0.2), start_frame + sweep_frames + 10)

    bpy.context.scene.frame_end = start_frame + sweep_frames + settle_frames


def grasp_lift(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Reach to object, grasp (constraint-based), lift.
    """
    random.seed(seed)
    target_obj_name = random.choice(objects)
    target_obj = bpy.data.objects[target_obj_name]
    obj_loc = target_obj.location.copy()

    approach_frame = 10
    grasp_frame = 30
    lift_frame = 50
    hold_frame = 70

    hand = kwargs.get("hand", "r")
    hand_key = f"hand_{hand}"
    wrist_key = f"wrist_{hand}"

    # Resolve the bone name for the child-of constraint (wrist preferred, hand fallback)
    if rc.has_bone(wrist_key):
        constraint_bone = rc.bone_name(wrist_key)
    elif rc.has_bone(hand_key):
        constraint_bone = rc.bone_name(hand_key)
    else:
        raise KeyError(
            f"Neither '{wrist_key}' nor '{hand_key}' found in rig '{rc.rig.name}'. Cannot create grasp constraint."
        )

    # Approach from above
    pre_grasp = obj_loc + Vector((0, -0.1, 0.15))
    rc.set_loc(hand_key, tuple(pre_grasp), approach_frame)

    # Move to grasp position
    rc.set_loc(hand_key, tuple(obj_loc + Vector((0, -0.02, 0))), grasp_frame)

    # Close grip at grasp frame
    rc.close_grip(hand, 0.0, approach_frame)
    rc.close_grip(hand, 0.9, grasp_frame)

    # Constraint-based grasp: object becomes kinematic child of wrist
    set_rigid_body_kinematic(target_obj, False, 1)
    set_rigid_body_kinematic(target_obj, True, grasp_frame)
    add_child_of_constraint(target_obj, rc.rig, constraint_bone, grasp_frame)

    # Lift
    lift_loc = obj_loc + Vector((0, -0.1, random.uniform(0.2, 0.4)))
    rc.set_loc(hand_key, tuple(lift_loc), lift_frame)
    rc.set_loc(hand_key, tuple(lift_loc), hold_frame)

    bpy.context.scene.frame_end = hold_frame + 10


def cloth_pickup(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Grab cloth by a point, lift. Uses hook modifier + vertex pinning.
    Expects a cloth object named in objects[0] with a 'grasp_pins' vertex group
    and a 'grasp_hook' empty parented to the hand bone.
    """
    random.seed(seed)
    cloth_name = objects[0] if objects else "cloth_plane"
    cloth_obj = bpy.data.objects[cloth_name]
    cloth_loc = cloth_obj.location.copy()

    # Random grasp point offset on cloth surface
    grasp_offset = Vector((random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0))
    grasp_point = cloth_loc + grasp_offset

    approach_frame = 10
    contact_frame = 25
    lift_frame = 45
    hold_frame = 65

    hand = kwargs.get("hand", "r")
    hand_key = f"hand_{hand}"

    # Approach
    rc.set_loc(hand_key, tuple(grasp_point + Vector((0, 0, 0.15))), approach_frame)
    # Contact
    rc.set_loc(hand_key, tuple(grasp_point + Vector((0, 0, 0.01))), contact_frame)

    # Close grip at contact
    rc.close_grip(hand, 0.0, approach_frame)
    rc.close_grip(hand, 1.0, contact_frame)

    # Activate pin stiffness at contact
    cloth_mod = cloth_obj.modifiers.get("Cloth")
    if cloth_mod:
        cloth_mod.settings.pin_stiffness = 0
        cloth_mod.settings.keyframe_insert("pin_stiffness", frame=contact_frame - 1)
        cloth_mod.settings.pin_stiffness = 1
        cloth_mod.settings.keyframe_insert("pin_stiffness", frame=contact_frame)

    # Lift
    lift_height = random.uniform(0.3, 0.6)
    rc.set_loc(hand_key, tuple(grasp_point + Vector((0, 0, lift_height))), lift_frame)
    rc.set_loc(hand_key, tuple(grasp_point + Vector((0, 0, lift_height))), hold_frame)

    bpy.context.scene.frame_end = hold_frame + 10


def cloth_fold(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Two-handed cloth fold. Grabs opposite edges, brings them together.
    Expects two hook empties: 'fold_hook_l' and 'fold_hook_r'.
    """
    random.seed(seed)
    cloth_name = objects[0] if objects else "cloth_plane"
    cloth_obj = bpy.data.objects[cloth_name]
    cloth_loc = cloth_obj.location.copy()

    half_w = kwargs.get("cloth_half_width", 0.2)

    grab_frame = 20
    fold_frame = 50
    settle_frame = 70

    if rc.is_humanoid():
        edge_l = cloth_loc + Vector((-half_w, 0, 0.01))
        edge_r = cloth_loc + Vector((half_w, 0, 0.01))

        # Approach
        rc.set_loc("hand_l", tuple(edge_l + Vector((0, 0, 0.15))), 5)
        rc.set_loc("hand_r", tuple(edge_r + Vector((0, 0, 0.15))), 5)

        # Close grips
        rc.close_grip("l", 0.0, 5)
        rc.close_grip("r", 0.0, 5)

        # Grab edges
        rc.set_loc("hand_l", tuple(edge_l), grab_frame)
        rc.set_loc("hand_r", tuple(edge_r), grab_frame)
        rc.close_grip("l", 1.0, grab_frame)
        rc.close_grip("r", 1.0, grab_frame)

        # Fold: bring left edge to right
        fold_target = edge_r + Vector((0, 0, 0.05))
        rc.set_loc("hand_l", tuple(fold_target), fold_frame)
        rc.set_loc("hand_r", tuple(edge_r + Vector((0, 0, 0.02))), fold_frame)

        # Release: lift away
        rc.set_loc("hand_l", tuple(fold_target + Vector((0, 0, 0.1))), settle_frame)
        rc.set_loc("hand_r", tuple(edge_r + Vector((0, 0, 0.1))), settle_frame)
        rc.close_grip("l", 0.0, settle_frame)
        rc.close_grip("r", 0.0, settle_frame)
    else:
        # Robotic / single-arm fold
        fold_hand = "hand_r" if rc.has_bone("hand_r") else "hand_l"
        rc.set_loc(fold_hand, tuple(cloth_loc + Vector((-half_w, 0, 0.15))), 5)
        rc.set_loc(fold_hand, tuple(cloth_loc + Vector((-half_w, 0, 0.01))), grab_frame)
        rc.set_loc(fold_hand, tuple(cloth_loc + Vector((half_w, 0, 0.05))), fold_frame)
        rc.set_loc(fold_hand, tuple(cloth_loc + Vector((half_w, 0, 0.15))), settle_frame)

    bpy.context.scene.frame_end = settle_frame + 20


def spill(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Knock over a container, causing a spill.
    Container goes from kinematic -> active at bump frame.
    Optionally triggers a particle/fluid emitter inside the container.
    """
    random.seed(seed)
    container_name = objects[0] if objects else "cup"
    container = bpy.data.objects[container_name]
    container_loc = container.location.copy()

    approach_frame = 10
    bump_frame = 25

    bump_direction = Vector(
        (
            random.uniform(-0.15, 0.15),
            random.uniform(0.05, 0.15),
            random.uniform(0, 0.05),
        )
    )

    hand = kwargs.get("hand", "r")
    hand_key = f"hand_{hand}"

    # Approach from side
    approach_pos = container_loc + Vector((-0.15, 0, 0.02))
    rc.set_loc(hand_key, tuple(approach_pos), approach_frame)
    # Push through
    push_pos = container_loc + bump_direction
    rc.set_loc(hand_key, tuple(push_pos), bump_frame)
    # Follow through
    rc.set_loc(hand_key, tuple(push_pos + bump_direction), bump_frame + 8)

    # Container: kinematic until bump, then active
    set_rigid_body_kinematic(container, True, 1)
    set_rigid_body_kinematic(container, False, bump_frame)

    # If there's a fluid emitter child, enable it at bump frame
    for child in container.children:
        if child.type == "MESH" and child.modifiers.get("FluidFlow"):
            child.hide_render = True
            child.keyframe_insert("hide_render", frame=bump_frame - 1)
            child.hide_render = False
            child.keyframe_insert("hide_render", frame=bump_frame)

    bpy.context.scene.frame_end = bump_frame + 60


def push_slide(rc: RigController, seed: int, objects: list, **kwargs):
    """
    Push an object across a surface. Object slides under friction.
    """
    random.seed(seed)
    target_name = random.choice(objects)
    target = bpy.data.objects[target_name]
    target_loc = target.location.copy()

    push_frame = 15
    release_frame = 30

    push_dir = Vector((random.uniform(-1, 1), random.uniform(-1, 1), 0)).normalized()
    push_dist = random.uniform(0.1, 0.3)

    hand = kwargs.get("hand", "r")
    hand_key = f"hand_{hand}"

    # Contact
    contact_pos = target_loc + (-push_dir * 0.05)
    rc.set_loc(hand_key, tuple(contact_pos), push_frame)
    # Push
    push_end = target_loc + (push_dir * push_dist)
    rc.set_loc(hand_key, tuple(push_end), release_frame)
    # Pull away
    rc.set_loc(hand_key, tuple(push_end + Vector((0, 0, 0.15))), release_frame + 10)

    bpy.context.scene.frame_end = release_frame + 40


# ---------------------------------------------------------------------------
# INTERACTION REGISTRY
# ---------------------------------------------------------------------------

INTERACTIONS = {
    "knockover": knockover,
    "grasp_lift": grasp_lift,
    "cloth_pickup": cloth_pickup,
    "cloth_fold": cloth_fold,
    "spill": spill,
    "push_slide": push_slide,
}

# ---------------------------------------------------------------------------
# ANNOTATION EXPORT
# ---------------------------------------------------------------------------


def export_annotations(output_dir: str, rig_name: str, interaction: str, seed: int):
    """
    Export per-frame 6DOF poses for all relevant objects.
    Outputs JSON with object poses, contact events, and metadata.
    """
    scene = bpy.context.scene
    frames = range(scene.frame_start, scene.frame_end + 1)
    data = {
        "rig": rig_name,
        "interaction": interaction,
        "seed": seed,
        "fps": scene.render.fps,
        "frames": {},
    }

    # Collect all rigid body objects + cloth objects
    tracked = [o for o in bpy.data.objects if o.rigid_body or any(m.type == "CLOTH" for m in o.modifiers)]

    for f in frames:
        scene.frame_set(f)
        frame_data = {}
        for obj in tracked:
            frame_data[obj.name] = {
                "location": list(obj.matrix_world.translation),
                "rotation": list(obj.matrix_world.to_euler()),
                "scale": list(obj.scale),
            }
            if obj.rigid_body:
                frame_data[obj.name]["kinematic"] = obj.rigid_body.kinematic
        data["frames"][str(f)] = frame_data

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{rig_name}_{interaction}_s{seed:04d}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as fp:
        json.dump(data, fp, indent=2)
    print(f"[interaction_gen] Annotations -> {filepath}")


# ---------------------------------------------------------------------------
# BATCH RUNNER
# ---------------------------------------------------------------------------


def run_single(
    rig_name: str,
    interaction: str,
    seed: int,
    objects: list = None,
    output_dir: str = "//output",
    render: bool = True,
    **kwargs,
):
    """Run a single interaction variant."""
    if objects is None:
        objects = [o.name for o in bpy.data.objects if o.rigid_body and o.name != rig_name]

    rc = RigController(rig_name)

    # Clear previous animation
    clear_keyframes(rc.rig)
    clear_physics_cache()

    # Run interaction
    fn = INTERACTIONS[interaction]
    fn(rc, seed, objects, **kwargs)

    # Bake physics (only if rigid body world exists)
    if bpy.context.scene.rigidbody_world:
        bpy.ops.ptcache.bake_all(bake=True)

    # Export annotations
    abs_output = bpy.path.abspath(output_dir)
    export_annotations(abs_output, rig_name, interaction, seed)

    # Render
    if render:
        scene = bpy.context.scene
        scene.render.filepath = os.path.join(abs_output, f"{rig_name}_{interaction}_s{seed:04d}_")
        bpy.ops.render.render(animation=True)

    print(f"[interaction_gen] Done: {rig_name}/{interaction}/seed={seed}")


def parse_seeds(seed_str: str) -> list:
    """Parse '0-49' or '0,5,10' into list of ints."""
    if "-" in seed_str and "," not in seed_str:
        start, end = seed_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seed_str.split(",")]


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    # Parse args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        print("[interaction_gen] No args. Use: -- --rig NAME --interaction TYPE --seeds RANGE")
        print(f"[interaction_gen] Available rigs: {list(BONE_MAPS.keys())}")
        print(f"[interaction_gen] Available interactions: {list(INTERACTIONS.keys())}")
        return

    import argparse

    parser = argparse.ArgumentParser(description="interaction_gen — procedural interaction generator")
    parser.add_argument("--rig", required=True, help=f"Rig object name. Available: {list(BONE_MAPS.keys())}")
    parser.add_argument("--interaction", required=True, choices=list(INTERACTIONS.keys()))
    parser.add_argument("--seeds", default="0-9", help="Seed range: '0-49' or '0,5,10'")
    parser.add_argument("--objects", nargs="*", help="Object names to interact with")
    parser.add_argument("--output", default="//output", help="Output directory")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hand", default="r", choices=["l", "r"], help="Which hand (humanoid)")
    parser.add_argument("--batch", help="Path to batch config JSON")
    parser.add_argument("--import-fbx", help="Import AccuRig FBX and wire IK before running")
    args = parser.parse_args(argv)

    # Import AccuRig FBX if requested
    if args.import_fbx:
        rig_name = import_accurig_fbx(args.import_fbx, rig_name=args.rig)
        # Override rig name with the imported armature name
        args.rig = rig_name

    if args.batch:
        with open(args.batch) as f:
            batch = json.load(f)
        for job in batch["jobs"]:
            seeds = parse_seeds(str(job.get("seeds", "0-9")))
            for s in seeds:
                run_single(
                    rig_name=job["rig"],
                    interaction=job["interaction"],
                    seed=s,
                    objects=job.get("objects"),
                    output_dir=job.get("output", args.output),
                    render=not job.get("no_render", False),
                    hand=job.get("hand", "r"),
                )
    else:
        seeds = parse_seeds(args.seeds)
        for s in seeds:
            run_single(
                rig_name=args.rig,
                interaction=args.interaction,
                seed=s,
                objects=args.objects,
                output_dir=args.output,
                render=not args.no_render,
                hand=args.hand,
            )


if __name__ == "__main__":
    main()
