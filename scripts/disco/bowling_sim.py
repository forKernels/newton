"""
Bowling Simulation — Blender 4.5+
Rolls a bowling ball toward 10 pins with randomised aim and speed.
Physics properties are already on the objects in the .blend file.

Usage (headless):
  blender --background "D:/_blender/.../Bowling_sim.blend" --python scripts/disco/bowling_sim.py

  # Override speed / aim:
  blender --background "Bowling_sim.blend" --python bowling_sim.py -- \
      --speed 3.5 --aim-offset 2.0 --frames 200

  # Batch (multiple seeds):
  blender --background "Bowling_sim.blend" --python bowling_sim.py -- \
      --seed 7 --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


# ── Configuration ────────────────────────────────────────────────────

SIM_FRAMES       = 200
ROLL_SPEED_MIN   = 2.5   # m/s — gentle roll
ROLL_SPEED_MAX   = 4.5   # m/s — firm roll
AIM_RANDOMNESS   = 4     # degrees max random offset
KINEMATIC_FRAMES = 4     # short ramp so ball picks up velocity

BALL_NAME = "BowlingBall"
PIN_PREFIX = "BowlingPin_"

# Physics tuning (applied on top of existing rigid bodies)
BALL_MASS        = 6.8    # kg  (real ≈ 6–7 kg)
BALL_FRICTION    = 0.30
BALL_RESTITUTION = 0.05
BALL_LIN_DAMP    = 0.08   # low — let it roll the full lane
BALL_ANG_DAMP    = 0.25   # moderate — realistic rolling drag

PIN_MASS         = 1.5    # kg  (real ≈ 1.5 kg)
PIN_FRICTION     = 0.35
PIN_RESTITUTION  = 0.15
PIN_LIN_DAMP     = 0.40
PIN_ANG_DAMP     = 0.50


# ── Helpers ──────────────────────────────────────────────────────────

def find_objects(scene):
    """Return (ball, [pins], floor, wall)."""
    ball = bpy.data.objects.get(BALL_NAME)
    pins = sorted(
        [o for o in scene.objects if o.type == "MESH" and o.name.startswith(PIN_PREFIX)],
        key=lambda o: o.name,
    )
    floor = bpy.data.objects.get("Floor")
    wall = bpy.data.objects.get("Wall")
    return ball, pins, floor, wall


def clear_all_keyframes(objects):
    """Remove animation data from all listed objects."""
    for obj in objects:
        if obj.animation_data:
            obj.animation_data_clear()


def compute_roll_direction(ball, pins, aim_offset_deg=None, rng=None):
    """Compute direction from ball toward head pin with random aim offset."""
    if not rng:
        rng = random.Random()

    # Aim at the head pin (closest to ball along Y)
    head_pin = min(pins, key=lambda p: (p.location - ball.location).length)
    base = (head_pin.location - ball.location)
    base.z = 0
    base = base.normalized()

    if aim_offset_deg is None:
        aim_offset_deg = rng.uniform(-AIM_RANDOMNESS, AIM_RANDOMNESS)

    angle = math.radians(aim_offset_deg)
    c, s = math.cos(angle), math.sin(angle)
    d = Vector((base.x * c - base.y * s, base.x * s + base.y * c, 0.0)).normalized()
    return d, aim_offset_deg


def update_ball_physics(ball):
    rb = ball.rigid_body
    if not rb:
        return
    rb.mass = BALL_MASS
    rb.friction = BALL_FRICTION
    rb.restitution = BALL_RESTITUTION
    rb.linear_damping = BALL_LIN_DAMP
    rb.angular_damping = BALL_ANG_DAMP
    rb.collision_shape = "SPHERE"


def update_pin_physics(pins):
    for pin in pins:
        rb = pin.rigid_body
        if not rb:
            continue
        rb.mass = PIN_MASS
        rb.friction = PIN_FRICTION
        rb.restitution = PIN_RESTITUTION
        rb.linear_damping = PIN_LIN_DAMP
        rb.angular_damping = PIN_ANG_DAMP
        rb.collision_shape = "CONVEX_HULL"


def setup_ball_roll(scene, ball, direction, speed):
    """Short kinematic ramp to give the ball initial rolling velocity."""
    fps = scene.render.fps
    step = speed / fps

    rb = ball.rigid_body
    start_pos = ball.location.copy()

    for f in range(1, KINEMATIC_FRAMES + 1):
        scene.frame_set(f)
        ball.location = start_pos + direction * step * (f - 1)
        ball.keyframe_insert(data_path="location", frame=f)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=f)

    release = KINEMATIC_FRAMES + 1
    scene.frame_set(release)
    rb.kinematic = False
    rb.keyframe_insert(data_path="kinematic", frame=release)

    # Interpolation
    if ball.animation_data and ball.animation_data.action:
        for fc in ball.animation_data.action.fcurves:
            if "kinematic" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
            if "location" in fc.data_path:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"

    travel = step * (KINEMATIC_FRAMES - 1)
    return travel, release


def freeze_pins(pins, unfreeze_frame):
    """Lock pins in place until unfreeze_frame, then release to dynamic."""
    for pin in pins:
        rb = pin.rigid_body
        if not rb:
            continue

        pos = pin.location.copy()

        pin.location = pos
        pin.keyframe_insert(data_path="location", frame=1)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=1)

        pin.location = pos
        pin.keyframe_insert(data_path="location", frame=unfreeze_frame - 1)
        rb.kinematic = True
        rb.keyframe_insert(data_path="kinematic", frame=unfreeze_frame - 1)

        rb.kinematic = False
        rb.keyframe_insert(data_path="kinematic", frame=unfreeze_frame)

        if pin.animation_data and pin.animation_data.action:
            for fc in pin.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


def bake_sim(scene, frames):
    scene.frame_start = 1
    scene.frame_end = frames
    rbw = scene.rigidbody_world
    if rbw:
        rbw.point_cache.frame_start = 1
        rbw.point_cache.frame_end = frames
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass
    print(f"  Baking {frames} frames...")
    bpy.ops.ptcache.bake_all(bake=True)
    print("  Bake complete.")


def export_usda(output_path=None):
    """Export animated USD. Uses blend filepath if output_path is None."""
    if output_path is None:
        blend_path = bpy.data.filepath
        if not blend_path:
            print("  ERROR: Save .blend first or pass --output")
            return None
        output_path = os.path.splitext(blend_path)[0] + "_sim.usda"

    print(f"  Exporting: {output_path}")
    bpy.ops.wm.usd_export(
        filepath=output_path,
        selected_objects_only=False,
        export_animation=True,
        export_meshes=True,
        export_materials=True,
        export_normals=True,
        export_uvmaps=True,
        evaluation_mode="RENDER",
    )
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported: {size:.1f} MB")
    return output_path


def write_metadata(json_path, metadata):
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {json_path}")


def save_blend(blend_path):
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
    print(f"  Saved: {blend_path}")


# ── Single sim run ───────────────────────────────────────────────────

def run_single_sim(
    *,
    seed: int = 42,
    speed: float | None = None,
    aim_offset: float | None = None,
    frames: int = SIM_FRAMES,
    output_dir: str | None = None,
    sample_idx: int = 1,
) -> dict:
    """Run one bowling simulation and export results."""
    rng = random.Random(seed)
    scene = bpy.context.scene
    t0 = time.time()

    ball, pins, floor_obj, wall_obj = find_objects(scene)
    if not ball or not ball.rigid_body:
        return {"error": "BowlingBall not found or no rigid body"}
    if not pins:
        return {"error": "No BowlingPin_* objects found"}

    all_objs = [ball] + pins
    head_pin = min(pins, key=lambda p: (p.location - ball.location).length)
    lane_dist = (head_pin.location - ball.location).length

    # 1. Clear keyframes
    clear_all_keyframes(all_objs)

    # 2. Update physics
    update_ball_physics(ball)
    update_pin_physics(pins)

    # 3. Compute roll direction
    if speed is None:
        speed = rng.uniform(ROLL_SPEED_MIN, ROLL_SPEED_MAX)
    direction, aim_deg = compute_roll_direction(ball, pins, aim_offset, rng)

    # 4. Kinematic ramp
    travel, release_frame = setup_ball_roll(scene, ball, direction, speed)

    # 5. Freeze pins until ball is close
    fps = scene.render.fps
    remaining = lane_dist - travel
    if remaining > 0 and speed > 0:
        frames_to_impact = remaining / (speed / fps)
        impact_frame = KINEMATIC_FRAMES + 1 + int(frames_to_impact)
    else:
        impact_frame = KINEMATIC_FRAMES + 2
    unfreeze = max(impact_frame - 1, KINEMATIC_FRAMES + 2)
    freeze_pins(pins, unfreeze)

    # 6. Bake
    bake_sim(scene, frames)
    bake_time = time.time() - t0

    # 7. Export
    metadata = {
        "mode": "bowling",
        "seed": seed,
        "speed_mps": round(speed, 2),
        "aim_offset_deg": round(aim_deg, 1),
        "lane_distance": round(lane_dist, 3),
        "impact_frame": impact_frame,
        "unfreeze_frame": unfreeze,
        "frames": frames,
        "bake_time_s": round(bake_time, 1),
        "pins": len(pins),
    }

    if output_dir:
        sim_dir = Path(output_dir) / "bowling"
        sim_dir.mkdir(parents=True, exist_ok=True)
        base = f"bowl_{sample_idx:03d}"

        blend_path = sim_dir / f"{base}.blend"
        usda_path = sim_dir / f"{base}.usda"
        json_path = sim_dir / f"{base}.json"

        save_blend(str(blend_path))
        usd = export_usda(str(usda_path))
        metadata["blend_file"] = blend_path.name
        metadata["usda_file"] = usda_path.name if usd else None
        write_metadata(str(json_path), metadata)
    else:
        save_blend(bpy.data.filepath)
        usd = export_usda()

    return metadata


# ── CLI main ─────────────────────────────────────────────────────────

def parse_blender_args(argv):
    """Extract args after '--' (Blender convention)."""
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def main():
    argv = parse_blender_args(sys.argv)
    parser = argparse.ArgumentParser(description="Bowling simulation (Blender)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=None, help="Ball speed m/s (default: random 2.5-4.5)")
    parser.add_argument("--aim-offset", type=float, default=None, help="Aim offset degrees (default: random ±4)")
    parser.add_argument("--frames", type=int, default=SIM_FRAMES)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: next to .blend)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of sims with different seeds")
    args = parser.parse_args(argv)

    print(f"\n{'=' * 55}")
    print(f"  BOWLING SIMULATION")
    print(f"{'=' * 55}")

    for i in range(1, args.num_samples + 1):
        seed = args.seed + i - 1
        print(f"\n-- Sample {i}/{args.num_samples} (seed={seed}) --")
        result = run_single_sim(
            seed=seed,
            speed=args.speed,
            aim_offset=args.aim_offset,
            frames=args.frames,
            output_dir=args.output_dir,
            sample_idx=i,
        )
        print(f"  Speed: {result.get('speed_mps', '?')} m/s | Aim: {result.get('aim_offset_deg', '?')}°")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")

    print(f"\n{'=' * 55}")
    print(f"  DONE — {args.num_samples} bowling sim(s) complete")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
