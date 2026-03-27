"""
Test knockover scene: character sweeps objects off a table.

Usage:
    blender -b --factory-startup -P scripts/disco/test_knockover.py -- --seed 42
    blender --factory-startup -P scripts/disco/test_knockover.py -- --seed 42
"""

import bpy
import json
import math
import os
import random
import sys
from mathutils import Vector, Euler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "sim_config.json")
RIG_FBX = "D:/_blender/_myBlender/SimulationWork/Rigged/rigged_male.fbx"

TABLE_HEIGHT = 0.95
TABLE_W = 1.0
TABLE_D = 0.6


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg["environments"][os.environ.get("SIM_ENV", "disco_machine")]


def find_tabletop_objects(dtc_dir, max_dim=0.20):
    results = []
    for name in os.listdir(dtc_dir):
        meta_path = os.path.join(dtc_dir, name, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        dims = meta.get("dimensions_m", [1, 1, 1])
        tags = meta.get("tags", [])
        if "tabletop_prop" in tags and max(dims) < max_dim:
            glb = os.path.join(dtc_dir, name, "visual_textured.glb")
            if not os.path.exists(glb):
                glb = os.path.join(dtc_dir, name, "visual.glb")
            if os.path.exists(glb):
                results.append({
                    "name": name, "glb": glb, "dims": dims,
                    "mass": meta.get("mass_kg", 0.1),
                    "friction": meta.get("friction_static", 0.5),
                    "restitution": meta.get("restitution", 0.3),
                })
    return results


# ---------------------------------------------------------------------------
# SCENE
# ---------------------------------------------------------------------------

def add_rigid_body(obj, rb_type, shape, friction=0.6, mass=1.0):
    """Add rigid body with margin=0 to any object."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type=rb_type)
    rb = obj.rigid_body
    rb.collision_shape = shape
    rb.collision_margin = 0.0
    rb.use_margin = True
    rb.friction = friction
    rb.mass = mass
    return rb


def create_table(cx, cy):
    """Visual table (no collision) + invisible collision cube on top."""
    mat = bpy.data.materials.new("TableMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.35, 0.22, 0.12, 1)
    bsdf.inputs["Roughness"].default_value = 0.7

    # Table top — set origin to geometry, then passive BOX rigid body
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, TABLE_HEIGHT))
    vis_top = bpy.context.active_object
    vis_top.name = "Table_Top"
    vis_top.scale = (TABLE_W, TABLE_D, 0.025)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    vis_top.data.materials.append(mat)
    add_rigid_body(vis_top, "PASSIVE", "BOX")

    # Legs
    leg_h = TABLE_HEIGHT - 0.0125
    ix = TABLE_W / 2 - 0.04
    iy = TABLE_D / 2 - 0.04
    for i, (lx, ly) in enumerate([
        (cx - ix, cy - iy), (cx + ix, cy - iy),
        (cx - ix, cy + iy), (cx + ix, cy + iy),
    ]):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(lx, ly, leg_h / 2))
        leg = bpy.context.active_object
        leg.name = f"Table_Leg_{i}"
        leg.scale = (0.04, 0.04, leg_h)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        leg.data.materials.append(mat)

    # Ground
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    gmat = bpy.data.materials.new("GroundMat")
    gmat.use_nodes = True
    gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.7, 0.7, 0.65, 1)
    ground.data.materials.append(gmat)
    add_rigid_body(ground, "PASSIVE", "BOX")

    return vis_top


def import_rig(fbx_path):
    sys.path.insert(0, SCRIPT_DIR)
    from interaction_gen import import_accurig_fbx
    rig_name = import_accurig_fbx(fbx_path)
    return rig_name, bpy.data.objects[rig_name]


def create_ik_controller(rig, bone_name, label, color=(1, 0.2, 0.2), size=0.06):
    """Create a visible control empty that drives an IK bone.

    Animate the controller, bone follows via COPY_LOCATION,
    collision sphere follows the controller too.

    Returns (controller_empty, collider_sphere).
    """
    # Get bone world position
    pb = rig.pose.bones[bone_name]
    bone_world = rig.matrix_world @ pb.head

    # 1. Controller empty — what you select and keyframe
    bpy.ops.object.empty_add(type="CUBE", location=bone_world, radius=size)
    ctrl = bpy.context.active_object
    ctrl.name = f"CTRL_{label}"
    ctrl.empty_display_size = size
    ctrl.show_in_front = True
    # Color it
    ctrl.color = (*color, 1.0)
    ctrl.show_name = True

    # 2. IK bone follows the controller
    bone_con = pb.constraints.new("COPY_LOCATION")
    bone_con.target = ctrl
    bone_con.name = f"Follow_{label}"

    # 3. Collision sphere follows the controller
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, segments=8, ring_count=6,
                                          location=bone_world)
    col = bpy.context.active_object
    col.name = f"COL_{label}"
    col.display_type = "WIRE"
    col.hide_render = True

    col_con = col.constraints.new("COPY_LOCATION")
    col_con.target = ctrl

    add_rigid_body(col, "ACTIVE", "SPHERE", friction=0.8, mass=5.0)
    col.rigid_body.kinematic = True
    col.keyframe_insert("rigid_body.kinematic", frame=1)

    return ctrl, col


def import_dtc_object(obj_info, location):
    """Import GLB, add MESH rigid body with margin=0."""
    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=obj_info["glb"])
    after = set(o.name for o in bpy.data.objects)
    meshes = [bpy.data.objects[n] for n in (after - before) if bpy.data.objects[n].type == "MESH"]
    if not meshes:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = obj_info["name"]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = Vector(location)

    # Props get MESH collision, margin=0
    add_rigid_body(obj, "ACTIVE", "MESH", friction=obj_info["friction"], mass=obj_info["mass"])
    obj.rigid_body.restitution = obj_info["restitution"]
    return obj


def setup_scene(frame_end):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.fps = 60

    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.point_cache.frame_start = 1
    scene.rigidbody_world.point_cache.frame_end = frame_end
    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 10

    bpy.ops.object.camera_add(location=(1.5, -1.2, 1.3))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = Euler((math.radians(60), 0, math.radians(48)))
    scene.camera = cam

    bpy.ops.object.light_add(type="SUN", location=(2, -1, 4))
    bpy.context.active_object.data.energy = 3.0

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480


# ---------------------------------------------------------------------------
# ANIMATION — direct keyframing, no matrix conversion needed
# ---------------------------------------------------------------------------

def animate_knockover(rig, ik_bone_name, table_cy, table_z, seed):
    """Keyframe the IK hand target to sweep across the table.

    Character is at origin with NO rotation, so bone local space = world space.
    We keyframe the IK target bone directly.
    """
    random.seed(seed)
    pb = rig.pose.bones[ik_bone_name]

    # Rest position of this IK target
    rest = pb.bone.head_local.copy()

    def key(frame, world_pos):
        """Set bone to a world position and keyframe it."""
        offset = Vector(world_pos) - rest
        pb.location = offset
        pb.keyframe_insert("location", frame=frame)

    # Timing at 60fps
    settle_end = 60        # 1s for objects to settle
    approach_end = 120     # 1s to reach sweep start
    sweep_end = 240        # 2s sweep across table
    lift_end = 270         # 0.5s lift away

    sweep_z = table_z + 0.06  # just above table surface
    start_x = random.uniform(-0.30, -0.20)
    end_x = random.uniform(0.25, 0.35)
    reach_y = table_cy + random.uniform(-0.05, 0.05)

    # Frame 1: arm at rest (zero offset)
    pb.location = Vector((0, 0, 0))
    pb.keyframe_insert("location", frame=1)

    # Approach: hand moves to sweep start
    key(approach_end, (start_x, reach_y, sweep_z))

    # Sweep across table
    key(sweep_end, (end_x, reach_y, sweep_z))

    # Lift away
    key(lift_end, (end_x, reach_y - 0.15, sweep_z + 0.25))

    # Set interpolation to linear for predictable motion
    if rig.animation_data and rig.animation_data.action:
        for fc in rig.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    return lift_end + 60  # extra frames for physics settle


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--output", default="C:/_data/SimObj/Output/test_knockover")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    cfg = load_config()
    bpy.ops.wm.read_homefile(use_empty=True)

    # Character at origin, NO rotation — bone local = world space
    # AccuRig faces -Y, so table goes at -Y
    TABLE_CY = -0.40

    setup_scene(frame_end=360)

    # 1. Table
    print("[test_knockover] Creating table...", flush=True)
    create_table(cx=0, cy=TABLE_CY)
    top_surface = TABLE_HEIGHT + 0.0125

    # 2. Character — NO rotation
    print("[test_knockover] Importing rig...", flush=True)
    rig_name, rig = import_rig(RIG_FBX)
    rig.location = Vector((0, 0, 0))
    rig.rotation_euler = Euler((0, 0, 0))
    bpy.context.view_layer.update()

    # IK controllers — visible empties you select & keyframe
    print("[test_knockover] Adding IK controllers...", flush=True)
    sys.path.insert(0, SCRIPT_DIR)
    from interaction_gen import BONE_MAPS
    bone_map = BONE_MAPS[rig_name]

    create_ik_controller(rig, bone_map["hand_r"], "Hand_R", color=(1, 0.2, 0.2))
    create_ik_controller(rig, bone_map["hand_l"], "Hand_L", color=(0.2, 0.4, 1))

    # 3. Props on table
    print("[test_knockover] Finding tabletop objects...", flush=True)
    all_tabletop = find_tabletop_objects(cfg["dtc_dir"], max_dim=0.15)
    print(f"[test_knockover] {len(all_tabletop)} candidates", flush=True)

    chosen = random.sample(all_tabletop, min(args.num_objects, len(all_tabletop)))
    for obj_info in chosen:
        x = random.uniform(-0.10, 0.10)
        y = TABLE_CY + random.uniform(0.05, 0.15)
        z = top_surface + 0.01 + max(obj_info["dims"]) / 2
        print(f"[test_knockover]   {obj_info['name']}", flush=True)
        obj = import_dtc_object(obj_info, (x, y, z))
        if obj:
            obj.rotation_euler = Euler((0, 0, random.uniform(0, math.tau)))

    # 4. No animation — user will keyframe manually
    print("[test_knockover] Scene ready — no keyframes, animate yourself", flush=True)

    # 5. Save
    os.makedirs(args.output, exist_ok=True)
    blend_path = os.path.join(args.output, f"knockover_s{args.seed:04d}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[test_knockover] Saved -> {blend_path}", flush=True)
    print("[test_knockover] DONE")


if __name__ == "__main__":
    main()
