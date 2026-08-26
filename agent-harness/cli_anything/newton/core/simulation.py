# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulation runner for Newton CLI.

Wraps Newton's simulation loop (clear_forces -> collide -> solver.step)
into a configurable, headless simulation pipeline.
"""

from __future__ import annotations

import time

# newton does NOT publish a constant default for rigid_contact_max: it
# ESTIMATES one per model, and the estimate is small - 1000 for a single box on
# a ground plane. Overflowing it does not raise. It prints
#
#     Warning: Contact buffer overflowed 17580 > 11000.
#
# and DROPS the excess contacts, so the symptom is a body passing through
# something it should have hit, at a threshold that moves with scene size.
#
# The Blender add-on that consumes this engine sizes it explicitly, and a
# harness that does not is measuring a world no consumer runs. That mistake
# produced four confident, reproducible and entirely fictional findings in one
# session: "the contact buffer overflows at 200 bricks", "more substeps is
# worse", "HULL beats BOX", and "the metric is hopelessly noisy". All four
# evaporated once the pipeline was sized the way the product sizes it.
CONTACT_MAX_FLOOR = 16384
CONTACT_MAX_PER_BODY = 512


def contact_capacity(model):
    """How many rigid contacts to make room for. Matches the add-on's rule."""
    bodies = int(getattr(model, "body_count", 0) or 0)
    return max(CONTACT_MAX_FLOOR, bodies * CONTACT_MAX_PER_BODY)


def make_pipeline(newton, model, **kwargs):
    """A CollisionPipeline sized for the scene, with the model told to match.

    Both halves matter. The pipeline allocates the contact buffer; the model
    field is what a solver reads when sizing its own. Use this instead of
    newton.CollisionPipeline - see CONTACT_MAX_FLOOR.
    """
    capacity = contact_capacity(model)
    try:
        model.rigid_contact_max = capacity
    except AttributeError:
        pass
    try:
        return newton.CollisionPipeline(
            model, rigid_contact_max=capacity, **kwargs)
    except TypeError:
        # An older build without the keyword still gets the model field.
        return newton.CollisionPipeline(model, **kwargs)


SOLVER_TYPES = {
    "vbd": "SolverVBD",
    "xpbd": "SolverXPBD",
    "mujoco": "SolverMuJoCo",
    "featherstone": "SolverFeatherstone",
    "semi_implicit": "SolverSemiImplicit",
    "style3d": "SolverStyle3D",
    "mpm": "SolverImplicitMPM",
}


def create_solver(model, solver_type: str = "xpbd", iterations: int = 10, **kwargs):
    """Create a Newton solver for the given model.

    Args:
        model: Finalized Newton Model.
        solver_type: One of: vbd, xpbd, mujoco, featherstone, semi_implicit, style3d, mpm.
        iterations: Solver iterations per substep.
        **kwargs: Additional solver-specific parameters.

    Returns:
        Solver instance.
    """
    if solver_type not in SOLVER_TYPES:
        raise ValueError(f"Unknown solver: {solver_type}. Choose from: {list(SOLVER_TYPES.keys())}")

    import newton.solvers as solvers

    cls_name = SOLVER_TYPES[solver_type]
    cls = getattr(solvers, cls_name)

    solver_kwargs = {"iterations": iterations}
    solver_kwargs.update(kwargs)

    # Some solvers don't accept iterations
    if solver_type in ("mujoco", "featherstone", "semi_implicit"):
        solver_kwargs.pop("iterations", None)

    return cls(model, **solver_kwargs)


def run_simulation(
    model,
    solver,
    num_frames: int = 100,
    substeps: int = 8,
    dt: float = 1.0 / 60.0,
    collision_margin: float = 0.01,
    viewer=None,
    progress_callback=None,
) -> dict:
    """Run a headless simulation loop.

    Args:
        model: Finalized Newton Model.
        solver: Solver instance.
        num_frames: Total frames to simulate.
        substeps: Physics substeps per frame.
        dt: Frame timestep in seconds.
        collision_margin: Collision margin for contact detection.
        viewer: Optional viewer for recording (ViewerUSD, ViewerFile, etc.).
        progress_callback: Optional callable(frame, num_frames) for progress.

    Returns:
        Dict with simulation results: timing, final state stats, etc.
    """
    import newton
    import numpy as np

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = make_pipeline(
        newton, model, soft_contact_margin=collision_margin)
    contacts = collision_pipeline.contacts()

    if viewer:
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state_0)
        viewer.end_frame()

    sub_dt = dt / substeps
    t_start = time.perf_counter()

    for frame in range(num_frames):
        for sub in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        if viewer:
            viewer.begin_frame((frame + 1) * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

        if progress_callback:
            progress_callback(frame + 1, num_frames)

    t_elapsed = time.perf_counter() - t_start

    if viewer:
        viewer.close()

    # Gather final state stats
    result = {
        "num_frames": num_frames,
        "substeps": substeps,
        "dt": dt,
        "total_sim_time": num_frames * dt,
        "wall_time_s": round(t_elapsed, 3),
        "fps": round(num_frames / t_elapsed, 1) if t_elapsed > 0 else 0,
        "steps_per_second": round((num_frames * substeps) / t_elapsed, 1) if t_elapsed > 0 else 0,
    }

    # Extract final particle/body positions if available
    if model.particle_count > 0:
        try:
            pos = state_0.particle_q.numpy()
            result["particle_bbox_min"] = pos.min(axis=0).tolist()
            result["particle_bbox_max"] = pos.max(axis=0).tolist()
            result["particle_center"] = pos.mean(axis=0).tolist()
        except Exception:
            pass

    if model.body_count > 0:
        try:
            bq = state_0.body_q.numpy()
            result["body_positions"] = bq[:, :3].tolist()
        except Exception:
            pass

    return result


def list_solvers() -> list[dict]:
    """List available Newton solvers with descriptions.

    Returns:
        List of dicts with solver info.
    """
    descriptions = {
        "vbd": "Vertex Block Descent — implicit, GPU-optimized for cloth + rigid coupling",
        "xpbd": "Extended Position-Based Dynamics — implicit, fast, good default",
        "mujoco": "MuJoCo Warp — generalized coords, Euler/RK4/implicit, primary robotics backend",
        "featherstone": "Featherstone — CRBA + semi-implicit Euler, articulated rigid bodies",
        "semi_implicit": "Semi-Implicit Euler — simple explicit integration",
        "style3d": "Style3D — BVH self-collision, cloth simulation",
        "mpm": "Implicit MPM — Material Point Method, granular/fluid materials",
    }

    return [
        {"name": name, "class": cls, "description": descriptions.get(name, "")}
        for name, cls in SOLVER_TYPES.items()
    ]
