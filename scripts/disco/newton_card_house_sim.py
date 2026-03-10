"""
Newton Card House Simulation
================================
Builds a randomized house of cards (leaning pairs + flat bridges) then
lets it collapse under gravity + optional perturbation.

Cards are thin rigid boxes (not cloth) for crisp stacking and collapse.

Usage:
    uv run python scripts/disco/newton_card_house_sim.py
    uv run python scripts/disco/newton_card_house_sim.py --test
    uv run python scripts/disco/newton_card_house_sim.py --seed 42 --levels 4 --perturb
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import warp as wp

import newton

from newton_sim_utils import (
    TEST_FRAMES,
    add_ground_plane,
    add_table,
    create_viewer,
    load_config,
    write_sim_metadata,
)

# Card dimensions (rigid box)
CARD_HX = 0.0004    # ~0.8mm thick (half)
CARD_HY = 0.0317    # 6.35cm wide (half)
CARD_HZ = 0.0444    # 8.89cm tall (half)
CARD_DENSITY = 700.0  # cardstock


def _lean_angle():
    """Lean angle for a pair of cards forming an inverted V."""
    return math.radians(75.0)  # steep lean


def build_card_house_scene(
    seed: int = 0,
    levels: int = 3,
    bays_base: int = 3,
    table_height: float = 0.4,
    lean_jitter: float = 0.02,
    position_jitter: float = 0.001,
    perturb: bool = False,
    perturb_speed: float = 0.3,
    solver_iters: int = 10,
):
    """Build a house of cards: leaning pairs per level, flat bridge cards on top.

    Structure per level:
      - N bays of leaning card pairs (inverted V)
      - (N-1) flat bridge cards spanning adjacent bays

    Each level has one fewer bay than the level below.
    """
    rng = random.Random(seed)

    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    table_half = (0.4, 0.4, 0.02)
    add_table(builder, position=(0.0, 0.0, table_height), size=table_half)
    base_z = table_height + table_half[2]

    card_cfg = newton.ModelBuilder.ShapeConfig()
    card_cfg.density = CARD_DENSITY
    card_cfg.mu = 0.35
    card_cfg.restitution = 0.05

    lean = _lean_angle()
    card_full_height = CARD_HZ * 2
    bay_width = card_full_height * math.sin(math.pi - lean) * 1.05
    level_height = card_full_height * math.cos(math.pi / 2 - lean)

    total_cards = 0

    for level in range(levels):
        n_bays = bays_base - level
        if n_bays <= 0:
            break

        row_z = base_z + level * (level_height + CARD_HX * 2) + CARD_HZ * math.cos(math.pi / 2 - lean)
        row_x_start = -(n_bays - 1) * bay_width / 2

        # Leaning pairs
        for bay in range(n_bays):
            cx = row_x_start + bay * bay_width
            cy = 0.0

            for side in [-1, 1]:
                angle = side * lean + rng.uniform(-lean_jitter, lean_jitter)
                px = cx + side * CARD_HZ * math.sin(math.pi / 2 - lean) * 0.5
                px += rng.uniform(-position_jitter, position_jitter)
                py = cy + rng.uniform(-position_jitter, position_jitter)
                pz = row_z

                q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)

                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(px, py, pz), q=q),
                )
                builder.add_shape_box(body=body, hx=CARD_HX, hy=CARD_HY, hz=CARD_HZ, cfg=card_cfg)
                total_cards += 1

        # Bridge cards (flat, spanning adjacent bays)
        if level < levels - 1:
            bridge_z = base_z + (level + 1) * (level_height + CARD_HX * 2) - CARD_HX
            for b in range(n_bays - 1):
                bx = row_x_start + (b + 0.5) * bay_width
                bx += rng.uniform(-position_jitter, position_jitter)
                by = rng.uniform(-position_jitter, position_jitter)

                # Flat card (rotated 90 around Y so it lies horizontal)
                q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2)

                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(bx, by, bridge_z), q=q),
                )
                builder.add_shape_box(body=body, hx=CARD_HX, hy=CARD_HY, hz=CARD_HZ, cfg=card_cfg)
                total_cards += 1

    # Optional perturbation sphere
    sphere_body_idx = None
    if perturb:
        sphere_z = base_z + CARD_HZ * 0.5
        sphere_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.4, 0.0, sphere_z), q=wp.quat_identity()),
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 2000.0
        sphere_cfg.mu = 0.3
        sphere_cfg.restitution = 0.2
        builder.add_shape_sphere(body=sphere_body, radius=0.015, cfg=sphere_cfg)
        sphere_body_idx = sphere_body

    print(f"  Cards: {total_cards}, levels: {levels}, base bays: {bays_base}")

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iters)

    # Set initial sphere velocity if perturbing
    initial_state = model.state()
    if perturb and sphere_body_idx is not None:
        body_qd = initial_state.body_qd.numpy()
        body_qd[sphere_body_idx] = [0.0, 0.0, 0.0, perturb_speed, 0.0, 0.0]
        initial_state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector)

    scene_info = {
        "total_cards": total_cards,
        "levels": levels,
        "bays_base": bays_base,
        "perturb": perturb,
        "perturb_speed": perturb_speed if perturb else 0.0,
        "lean_jitter": lean_jitter,
        "position_jitter": position_jitter,
        "table_height": table_height,
    }

    return model, solver, initial_state, scene_info


def run_card_house_sim(
    model,
    solver,
    initial_state,
    num_frames: int = 300,
    substeps: int = 10,
    dt: float = 1.0 / 60.0,
    viewer=None,
):
    state_0 = initial_state
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, None, state_0)
    sub_dt = dt / substeps

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    for frame in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

        if (frame + 1) % 50 == 0:
            print(f"  Frame {frame + 1}/{num_frames}")

    if viewer:
        viewer.close()

    return state_0


def count_collapsed(model, state, total_cards: int, base_z: float) -> int:
    """Count cards that have fallen flat (center Z near table surface)."""
    if model.body_count < total_cards:
        return 0
    body_q = state.body_q.numpy()
    collapsed = 0
    for i in range(total_cards):
        z = body_q[i][2]
        if z < base_z + CARD_HY * 1.2:  # roughly flat on surface
            collapsed += 1
    return collapsed


def main():
    parser = argparse.ArgumentParser(description="Newton Card House Simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--bays", type=int, default=3, help="Number of bays at base level")
    parser.add_argument("--perturb", action="store_true",
                        help="Roll a sphere into the house")
    parser.add_argument("--perturb-speed", type=float, default=0.3)
    parser.add_argument("--lean-jitter", type=float, default=0.02,
                        help="Random lean angle variation [radians]")
    parser.add_argument("--position-jitter", type=float, default=0.001,
                        help="Random position offset [m]")
    parser.add_argument("--table-height", type=float, default=0.4)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--solver-iters", type=int, default=10)
    parser.add_argument("--viewer", type=str, default="null",
                        choices=["null", "usd", "gl"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Test run: few frames, GL viewer")
    args = parser.parse_args()

    if args.test:
        args.num_frames = TEST_FRAMES
        args.viewer = args.viewer if args.viewer != "null" else "gl"

    wp.init()
    if args.device:
        device = wp.get_device(args.device)
    else:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        config = load_config(args.config)

        output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_dir", "./output"))
        sim_name = f"card_house_s{args.seed:06d}"
        sim_dir = output_dir / "card_house" / sim_name
        sim_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Newton Card House (seed={args.seed}) ===")

        model, solver, initial_state, scene_info = build_card_house_scene(
            seed=args.seed,
            levels=args.levels,
            bays_base=args.bays,
            table_height=args.table_height,
            lean_jitter=args.lean_jitter,
            position_jitter=args.position_jitter,
            perturb=args.perturb,
            perturb_speed=args.perturb_speed,
            solver_iters=args.solver_iters,
        )

        usd_path = sim_dir / f"{sim_name}.usd" if args.viewer == "usd" else None
        viewer = create_viewer(output_path=usd_path, viewer_type=args.viewer)

        final_state = run_card_house_sim(
            model, solver, initial_state,
            num_frames=args.num_frames,
            substeps=args.substeps,
            viewer=viewer,
        )

        base_z = args.table_height + 0.02
        collapsed = count_collapsed(model, final_state, scene_info["total_cards"], base_z)
        print(f"  Collapsed: {collapsed}/{scene_info['total_cards']}")

        write_sim_metadata(
            sim_dir / "metadata.json",
            sim_type="card_house",
            seed=args.seed,
            cards_collapsed=collapsed,
            **scene_info,
        )

        print(f"  Done: {sim_dir}")


if __name__ == "__main__":
    main()
