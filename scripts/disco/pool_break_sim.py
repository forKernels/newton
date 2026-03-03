"""
Pool Break Simulation v6 — Blender 4.5+
Fixes: rack freeze, shorter kinematic ramp, net toggle
"""

import bpy
import math
import os
import random
from mathutils import Vector


# ── Configuration ─────────────────────────────────────────────────────

SIM_FRAMES       = 150
BREAK_SPEED_MIN  = 3.0
BREAK_SPEED_MAX  = 6.0
AIM_RANDOMNESS   = 8          # degrees
CUE_BALL_NAME    = "PoolTable_02_CueBall"
KINEMATIC_FRAMES = 3          # DOWN from 4 — less ramp, more room before rack
ENABLE_NETS      = False       # flip True for final bake


# ── Physics values for ~0.88x scale table ─────────────────────────────

BALL_MASS        = 0.8
BALL_FRICTION    = 0.25
BALL_RESTITUTION = 0.6        # slightly higher for better energy transfer
BALL_LIN_DAMP    = 0.25
BALL_ANG_DAMP    = 0.65


# ── Helpers ───────────────────────────────────────────────────────────

def find_balls(scene):
    cue = bpy.data.objects.get(CUE_BALL_NAME)
    others = [o for o in scene.objects
              if o.type == 'MESH'
              and 'Ball_' in o.name
              and o.name != CUE_BALL_NAME]
    return cue, others


def compute_break_direction(cue_ball, target_balls):
    if target_balls:
        center = sum((b.location for b in target_balls), Vector()) / len(target_balls)
        base = (center - cue_ball.location)
        base.z = 0
        base = base.normalized()
    else:
        base = Vector((1, 0, 0))

    angle = random.uniform(
        -math.radians(AIM_RANDOMNESS),
        math.radians(AIM_RANDOMNESS)
    )
    c, s = math.cos(angle), math.sin(angle)
    d = Vector((base.x*c - base.y*s, base.x*s + base.y*c, 0.0)).normalized()
    return d, math.degrees(angle)


def clear_all_keyframes(objects):
    """Remove animation data from all listed objects."""
    for obj in objects:
        if obj.animation_data:
            obj.animation_data_clear()
        # Also clear rigid body keyframes
        if obj.rigid_body and obj.animation_data:
            obj.animation_data_clear()


def freeze_rack(rack_balls, unfreeze_frame):
    """
    Lock rack balls in place until unfreeze_frame.
    Keyframes BOTH location AND kinematic with CONSTANT interpolation.
    """
    for ball in rack_balls:
        rb = ball.rigid_body
        if not rb:
            continue

        pos = ball.location.copy()

        # Frame 1: locked in place, kinematic
        ball.location = pos
        ball.keyframe_insert(data_path="location", frame=1)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=1)

        # Hold until unfreeze
        ball.location = pos
        ball.keyframe_insert(data_path="location", frame=unfreeze_frame - 1)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=unfreeze_frame - 1)

        # Unfreeze: go dynamic
        rb.kinematic = False
        rb.keyframe_insert(data_path="kinematic", frame=unfreeze_frame)

        # Set interpolation
        if ball.animation_data and ball.animation_data.action:
            for fc in ball.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'

    print(f"  Froze {len(rack_balls)} rack balls until frame {unfreeze_frame}")


def setup_cue_break(scene, cue_ball, direction, speed):
    """
    Short kinematic ramp — 2 frames only.
    Keeps cue well away from rack when it goes dynamic.
    """
    fps = scene.render.fps
    step = speed / fps

    rb = cue_ball.rigid_body
    start_pos = cue_ball.location.copy()

    for f in range(1, KINEMATIC_FRAMES + 1):
        scene.frame_set(f)
        cue_ball.location = start_pos + direction * step * (f - 1)
        cue_ball.keyframe_insert(data_path="location", frame=f)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=f)

    release = KINEMATIC_FRAMES + 1
    scene.frame_set(release)
    rb.kinematic = False
    rb.keyframe_insert(data_path="kinematic", frame=release)

    # Interpolation
    if cue_ball.animation_data and cue_ball.animation_data.action:
        for fc in cue_ball.animation_data.action.fcurves:
            if "kinematic" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'CONSTANT'
            if "location" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    travel = step * (KINEMATIC_FRAMES - 1)
    print(f"  Speed: {speed:.1f} m/s ({step:.4f} m/frame)")
    print(f"  Kinematic ramp: {KINEMATIC_FRAMES} frames, {travel:.3f}m travel")
    print(f"  Goes dynamic at frame {release}")
    return travel


def update_ball_physics(balls):
    """Update rigid body values on existing physics — don't remove/re-add."""
    for ball in balls:
        rb = ball.rigid_body
        if not rb:
            continue
        rb.mass = BALL_MASS
        rb.friction = BALL_FRICTION
        rb.restitution = BALL_RESTITUTION
        rb.linear_damping = BALL_LIN_DAMP
        rb.angular_damping = BALL_ANG_DAMP
        rb.collision_shape = 'SPHERE'
    print(f"  Updated physics on {len(balls)} balls")


def toggle_nets(enable):
    count = 0
    for obj in bpy.context.scene.objects:
        if 'Net' not in obj.name:
            continue
        for mod in obj.modifiers:
            if mod.type == 'CLOTH':
                mod.show_viewport = enable
                mod.show_render = enable
                count += 1
    print(f"  {count} cloth sims {'ENABLED' if enable else 'DISABLED'}")


def bake_sim(scene, frames):
    scene.frame_start = 1
    scene.frame_end = frames
    rbw = scene.rigidbody_world
    if rbw:
        rbw.point_cache.frame_start = 1
        rbw.point_cache.frame_end = frames
    try:
        bpy.ops.ptcache.free_bake_all()
    except:
        pass
    print(f"  Baking {frames} frames...")
    bpy.ops.ptcache.bake_all(bake=True)
    print("  Done.")


def export_usda():
    blend_path = bpy.data.filepath
    if not blend_path:
        print("  ERROR: Save .blend first!")
        return None
    usd_path = os.path.join(
        os.path.dirname(blend_path),
        os.path.splitext(os.path.basename(blend_path))[0] + "_sim.usda"
    )
    print(f"  Exporting: {usd_path}")
    bpy.ops.wm.usd_export(
        filepath=usd_path,
        selected_objects_only=False,
        export_animation=True,
        export_meshes=True,
        export_materials=True,
        export_normals=True,
        export_uvmaps=True,
        evaluation_mode='RENDER',
    )
    size = os.path.getsize(usd_path) / (1024*1024)
    print(f"  Exported: {size:.1f} MB")
    return usd_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    scene = bpy.context.scene
    cue, rack = find_balls(scene)
    if not cue or not cue.rigid_body:
        print("ERROR: Cue ball not found or no rigid body!")
        return

    all_balls = [cue] + rack
    rack_dist = 0
    if rack:
        center = sum((b.location for b in rack), Vector()) / len(rack)
        rack_dist = (center - cue.location).length

    print(f"\n{'='*55}")
    print(f"  POOL BREAK v6")
    print(f"{'='*55}")
    print(f"  Balls: {len(all_balls)} (1 cue + {len(rack)} rack)")
    print(f"  Rack distance: {rack_dist:.3f}m")
    print(f"  Nets: {'ON' if ENABLE_NETS else 'OFF'}")

    # 1. Clear ALL keyframes from all balls
    print(f"\n-- Clearing keyframes --")
    clear_all_keyframes(all_balls)

    # 2. Update physics values
    print(f"\n-- Updating ball physics --")
    update_ball_physics(all_balls)

    # 3. Compute break
    direction, aim_offset = compute_break_direction(cue, rack)
    speed = random.uniform(BREAK_SPEED_MIN, BREAK_SPEED_MAX)
    print(f"  Aim offset: {aim_offset:+.1f}°")

    # 4. Set up cue ball kinematic ramp
    print(f"\n-- Cue ball --")
    travel = setup_cue_break(scene, cue, direction, speed)

    # 5. Freeze rack until cue arrives
    #    Estimate impact frame from distance and speed
    fps = scene.render.fps
    remaining = rack_dist - travel
    if remaining > 0 and speed > 0:
        frames_to_impact = remaining / (speed / fps)
        impact_frame = KINEMATIC_FRAMES + 1 + int(frames_to_impact)
    else:
        impact_frame = KINEMATIC_FRAMES + 2

    # Unfreeze 1 frame before impact so Bullet sees them as dynamic
    unfreeze = max(impact_frame - 1, KINEMATIC_FRAMES + 2)

    print(f"\n-- Rack freeze --")
    print(f"  Estimated impact: frame {impact_frame}")
    print(f"  Unfreeze rack at: frame {unfreeze}")
    freeze_rack(rack, unfreeze)

    # 6. Nets
    print(f"\n-- Nets --")
    toggle_nets(ENABLE_NETS)

    # 7. Bake
    print(f"\n-- Simulation --")
    bake_sim(scene, SIM_FRAMES)

    # 8. Export
    print(f"\n-- Export --")
    usd_path = export_usda()

    print(f"\n{'='*55}")
    print(f"  Speed: {speed:.1f} m/s | Aim: {aim_offset:+.1f}°")
    print(f"  Nets: {'ON' if ENABLE_NETS else 'OFF'}")
    if usd_path:
        print(f"  USDA: {usd_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()