"""
Export Blender Robot Arm Animations → Newton / Isaac Lab
=========================================================
Run this script in Blender AFTER keyframing your 11 cloth group animations.

Outputs (per cloth group):
  1. joint_trajectories.json   — Joint angles at each timestep (for joint-space policies)
  2. ee_trajectories.json      — End-effector poses (for IK/Cartesian policies)
  3. gripper_commands.json      — Gripper open/close timeline
  4. dmp_params.json           — Fitted DMP weights (for adaptive replay)
  5. isaac_lab_demos.hdf5      — HDF5 dataset formatted for Isaac Lab Mimic

Usage:
  1. In Blender, open Scripting workspace
  2. Load this script
  3. Set OUTPUT_DIR below
  4. Run script
"""

import json
import os

import bpy
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "//franka_cloth_animations/"  # Blender-relative path (// = .blend file dir)
# For absolute path, use something like:
# OUTPUT_DIR = "/home/david/projects/cloth_manipulation/animations/"

EXPORT_FPS = 10.0  # Control frequency for the robot (Hz) — Newton default
SCENE_FPS = 30  # Blender scene FPS

# Franka joint info (must match setup script)
JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
JOINT_AXES = ["Z", "Y", "Z", "-Y", "Z", "Y", "Z"]

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def get_joint_angle(pose_bone, axis):
    """Extract the active joint angle from a pose bone given its rotation axis."""
    rot = pose_bone.rotation_euler
    if axis == "Z":
        return rot.z
    elif axis == "-Z":
        return -rot.z
    elif axis == "Y":
        return rot.y
    elif axis == "-Y":
        return -rot.y
    return 0.0


def get_ee_pose(arm_obj):
    """
    Compute end-effector pose in world space.
    Returns (x, y, z, roll, pitch, yaw) in meters and radians.
    """
    # Get the last joint bone's world-space tail position
    last_joint = arm_obj.pose.bones[JOINT_NAMES[-1]]

    # World-space matrix of the end-effector
    world_matrix = arm_obj.matrix_world @ last_joint.matrix

    # Extract position
    pos = world_matrix.translation

    # Extract rotation as Euler
    rot = world_matrix.to_euler("XYZ")

    return {
        "position": [pos.x, pos.y, pos.z],
        "orientation_euler": [rot.x, rot.y, rot.z],
        "orientation_quat": list(world_matrix.to_quaternion()),
    }


def extract_action_trajectory(arm_obj, action):
    """
    Extract a complete trajectory from a Blender action.
    Samples at EXPORT_FPS rate regardless of scene FPS.

    Returns dict with joint angles, ee poses, and gripper state per timestep.
    """
    # Store current state
    original_action = arm_obj.animation_data.action
    original_frame = bpy.context.scene.frame_current

    # Set the action
    arm_obj.animation_data.action = action

    # Determine frame range from action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])

    # Calculate sample frames based on export FPS
    scene_fps = bpy.context.scene.render.fps
    frame_step = scene_fps / EXPORT_FPS  # e.g., 30fps scene / 10Hz export = every 3 frames

    trajectory = {
        "metadata": {
            "cloth_group_id": action.get("cloth_group_id", -1),
            "cloth_group_name": action.get("cloth_group_name", "unknown"),
            "cloth_stiffness": action.get("cloth_stiffness", 0.0),
            "cloth_geometry": action.get("cloth_geometry", "unknown"),
            "cloth_dataset_folder": action.get("cloth_dataset_folder", "unknown"),
            "control_frequency_hz": EXPORT_FPS,
            "num_timesteps": 0,
            "duration_seconds": 0.0,
            "joint_names": JOINT_NAMES,
        },
        "timesteps": [],
    }

    frame = float(frame_start)
    timestep_idx = 0

    while frame <= frame_end:
        # Set frame (fractional frames are interpolated by Blender)
        bpy.context.scene.frame_set(int(round(frame)))

        # Force dependency graph update
        bpy.context.view_layer.update()

        # Extract joint angles
        joint_positions = []
        joint_velocities = []  # approximate from finite differences

        for i, name in enumerate(JOINT_NAMES):
            pose_bone = arm_obj.pose.bones[name]
            angle = get_joint_angle(pose_bone, JOINT_AXES[i])
            joint_positions.append(angle)

        # Extract end-effector pose
        ee_pose = get_ee_pose(arm_obj)

        # Extract gripper state
        gripper_width = arm_obj.get("gripper_width", 0.04)
        gripper_open = gripper_width > 0.01  # threshold for open/closed

        timestep_data = {
            "timestep": timestep_idx,
            "time": timestep_idx / EXPORT_FPS,
            "joint_positions": joint_positions,
            "ee_position": ee_pose["position"],
            "ee_orientation_euler": ee_pose["orientation_euler"],
            "ee_orientation_quat": ee_pose["orientation_quat"],
            "gripper_width": gripper_width,
            "gripper_open": gripper_open,
        }

        trajectory["timesteps"].append(timestep_data)

        frame += frame_step
        timestep_idx += 1

    # Compute velocities via finite differences
    dt = 1.0 / EXPORT_FPS
    for i in range(len(trajectory["timesteps"])):
        if i == 0:
            trajectory["timesteps"][i]["joint_velocities"] = [0.0] * 7
        else:
            prev = trajectory["timesteps"][i - 1]["joint_positions"]
            curr = trajectory["timesteps"][i]["joint_positions"]
            vels = [(c - p) / dt for c, p in zip(curr, prev)]
            trajectory["timesteps"][i]["joint_velocities"] = vels

    # Update metadata
    trajectory["metadata"]["num_timesteps"] = len(trajectory["timesteps"])
    trajectory["metadata"]["duration_seconds"] = len(trajectory["timesteps"]) / EXPORT_FPS

    # Restore original state
    arm_obj.animation_data.action = original_action
    bpy.context.scene.frame_set(original_frame)

    return trajectory


def fit_dmp_parameters(trajectory, n_basis=25):
    """
    Fit Dynamic Movement Primitive parameters to the extracted trajectory.
    This allows adaptive replay — same motion shape, different start/goal.

    Returns DMP weights per joint that can be loaded in Python/C++ on the robot.
    """
    timesteps = trajectory["timesteps"]
    n_steps = len(timesteps)

    if n_steps < 3:
        return None

    dmp_params = {
        "n_basis": n_basis,
        "dt": 1.0 / trajectory["metadata"]["control_frequency_hz"],
        "tau": trajectory["metadata"]["duration_seconds"],
        "alpha_z": 25.0,  # spring constant
        "beta_z": 6.25,  # damping (alpha_z / 4)
        "alpha_x": 5.0,  # canonical system decay
        "joints": {},
    }

    for j_idx, j_name in enumerate(JOINT_NAMES):
        # Extract joint trajectory
        positions = [ts["joint_positions"][j_idx] for ts in timesteps]
        y = np.array(positions)

        y0 = y[0]  # start
        goal = y[-1]  # goal

        # Phase variable (canonical system)
        phase = np.exp(-dmp_params["alpha_x"] * np.linspace(0, 1, n_steps))

        # Compute forcing function from trajectory
        # f(t) = tau^2 * y_dd - alpha * (beta * (goal - y) - tau * y_d)
        dy = np.gradient(y, dmp_params["dt"])
        ddy = np.gradient(dy, dmp_params["dt"])

        tau = dmp_params["tau"]
        alpha = dmp_params["alpha_z"]
        beta = dmp_params["beta_z"]

        f_target = tau**2 * ddy - alpha * (beta * (goal - y) - tau * dy)

        # Gaussian basis functions
        centers = np.exp(-dmp_params["alpha_x"] * np.linspace(0, 1, n_basis))
        widths = 1.0 / (0.65 * np.diff(centers) ** 2)
        widths = np.append(widths, widths[-1])

        # Fit weights using least squares
        Phi = np.zeros((n_steps, n_basis))
        for k in range(n_basis):
            psi = np.exp(-widths[k] * (phase - centers[k]) ** 2)
            Phi[:, k] = (
                psi
                * phase
                / (np.sum([np.exp(-widths[m] * (phase - centers[m]) ** 2) for m in range(n_basis)], axis=0) + 1e-10)
            )

        # Least squares fit
        weights, _, _, _ = np.linalg.lstsq(Phi, f_target, rcond=None)

        dmp_params["joints"][j_name] = {
            "start": float(y0),
            "goal": float(goal),
            "weights": weights.tolist(),
        }

    # Also fit gripper trajectory
    gripper_positions = [ts["gripper_width"] for ts in timesteps]
    dmp_params["gripper_timeline"] = {
        "times": [ts["time"] for ts in timesteps],
        "widths": gripper_positions,
        "open_close_events": [],
    }

    # Detect open/close transitions
    for i in range(1, len(gripper_positions)):
        prev_open = gripper_positions[i - 1] > 0.01
        curr_open = gripper_positions[i] > 0.01
        if prev_open != curr_open:
            dmp_params["gripper_timeline"]["open_close_events"].append(
                {
                    "time": timesteps[i]["time"],
                    "action": "open" if curr_open else "close",
                }
            )

    return dmp_params


def export_isaac_lab_format(all_trajectories, output_dir):
    """
    Export trajectories in a format compatible with Isaac Lab's
    demonstration loading for behavior cloning / Isaac Lab Mimic.

    Isaac Lab expects HDF5 with specific structure:
      /data/demo_0/obs          — observations array
      /data/demo_0/actions      — actions array
      /data/demo_0/rewards      — rewards (can be dummy for demos)
      /data/demo_0/dones        — episode termination flags
    """
    try:
        import h5py
    except ImportError:
        print("WARNING: h5py not available in Blender's Python. Skipping HDF5 export.")
        print("Install with: pip install h5py")
        print("Exporting as JSON instead...")

        # Fallback: export as JSON that can be converted externally
        isaac_data = {"demos": []}

        for traj in all_trajectories:
            demo = {
                "metadata": traj["metadata"],
                "obs": [],
                "actions": [],
            }

            for i, ts in enumerate(traj["timesteps"]):
                # Observation: joint_pos + joint_vel + gripper_width
                obs = ts["joint_positions"] + ts["joint_velocities"] + [ts["gripper_width"]]
                demo["obs"].append(obs)

                # Action: next joint position delta (or absolute, depending on policy type)
                if i < len(traj["timesteps"]) - 1:
                    next_ts = traj["timesteps"][i + 1]
                    action = [next_ts["joint_positions"][j] - ts["joint_positions"][j] for j in range(7)] + [
                        next_ts["gripper_width"] - ts["gripper_width"]
                    ]
                else:
                    action = [0.0] * 8

                demo["actions"].append(action)

            isaac_data["demos"].append(demo)

        isaac_path = os.path.join(output_dir, "isaac_lab_demos.json")
        with open(isaac_path, "w") as f:
            json.dump(isaac_data, f, indent=2)
        print(f"  Exported Isaac Lab JSON: {isaac_path}")
        return

    # HDF5 export
    hdf5_path = os.path.join(output_dir, "isaac_lab_demos.hdf5")

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")

        for demo_idx, traj in enumerate(all_trajectories):
            demo_grp = data_grp.create_group(f"demo_{demo_idx}")

            # Build observation and action arrays
            obs_list = []
            action_list = []

            for i, ts in enumerate(traj["timesteps"]):
                obs = ts["joint_positions"] + ts["joint_velocities"] + [ts["gripper_width"]]
                obs_list.append(obs)

                if i < len(traj["timesteps"]) - 1:
                    next_ts = traj["timesteps"][i + 1]
                    action = [next_ts["joint_positions"][j] - ts["joint_positions"][j] for j in range(7)] + [
                        next_ts["gripper_width"] - ts["gripper_width"]
                    ]
                else:
                    action = [0.0] * 8
                action_list.append(action)

            demo_grp.create_dataset("obs", data=np.array(obs_list))
            demo_grp.create_dataset("actions", data=np.array(action_list))
            demo_grp.create_dataset("rewards", data=np.zeros(len(obs_list)))

            dones = np.zeros(len(obs_list))
            dones[-1] = 1.0
            demo_grp.create_dataset("dones", data=dones)

            # Store metadata
            demo_grp.attrs["cloth_group_id"] = traj["metadata"]["cloth_group_id"]
            demo_grp.attrs["cloth_group_name"] = traj["metadata"]["cloth_group_name"]
            demo_grp.attrs["num_timesteps"] = traj["metadata"]["num_timesteps"]

        # Global metadata
        f.attrs["num_demos"] = len(all_trajectories)
        f.attrs["control_frequency_hz"] = EXPORT_FPS
        f.attrs["joint_names"] = json.dumps(JOINT_NAMES)

    print(f"  Exported Isaac Lab HDF5: {hdf5_path}")


# =============================================================================
# MAIN EXPORT
# =============================================================================


def export_all():
    """Export all cloth group animations."""
    print("=" * 60)
    print("EXPORTING ROBOT ARM ANIMATIONS")
    print("=" * 60)

    # Find the arm
    arm_obj = bpy.data.objects.get("FrankaPanda")
    if not arm_obj:
        print("ERROR: FrankaPanda armature not found. Run setup script first.")
        return

    if not arm_obj.animation_data:
        print("ERROR: No animation data on FrankaPanda.")
        return

    # Resolve output directory
    output_dir = bpy.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Find all cloth group actions
    cloth_actions = []
    for action in bpy.data.actions:
        if action.name.startswith("ClothGroup_"):
            cloth_actions.append(action)

    cloth_actions.sort(key=lambda a: a.get("cloth_group_id", 0))
    print(f"Found {len(cloth_actions)} cloth group actions.\n")

    all_trajectories = []
    all_dmp_params = []

    for action in cloth_actions:
        group_id = action.get("cloth_group_id", -1)
        group_name = action.get("cloth_group_name", "unknown")
        print(f"Processing Group {group_id}: {group_name}...")

        # Create per-group output directory
        group_dir = os.path.join(output_dir, f"group_{group_id:02d}_{group_name}")
        os.makedirs(group_dir, exist_ok=True)

        # Extract trajectory
        trajectory = extract_action_trajectory(arm_obj, action)
        all_trajectories.append(trajectory)

        # Save joint trajectory
        joint_path = os.path.join(group_dir, "joint_trajectory.json")
        with open(joint_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        print(
            f"  Joint trajectory: {trajectory['metadata']['num_timesteps']} timesteps, "
            f"{trajectory['metadata']['duration_seconds']:.1f}s"
        )

        # Fit and save DMP parameters
        try:
            dmp_params = fit_dmp_parameters(trajectory)
            if dmp_params:
                all_dmp_params.append(dmp_params)
                dmp_path = os.path.join(group_dir, "dmp_params.json")
                with open(dmp_path, "w") as f:
                    json.dump(dmp_params, f, indent=2)
                print(f"  DMP parameters: {dmp_params['n_basis']} basis functions per joint")
        except Exception as e:
            print(f"  WARNING: DMP fitting failed: {e}")

        # Save a compact summary for quick loading
        summary = {
            "group_id": group_id,
            "group_name": group_name,
            "stiffness": action.get("cloth_stiffness", 0.0),
            "geometry": action.get("cloth_geometry", "unknown"),
            "dataset_folder": action.get("cloth_dataset_folder", "unknown"),
            "duration": trajectory["metadata"]["duration_seconds"],
            "num_timesteps": trajectory["metadata"]["num_timesteps"],
            "start_joints": trajectory["timesteps"][0]["joint_positions"],
            "grasp_event": None,
            "peak_ee_height": 0.0,
        }

        # Find grasp event and peak height
        max_height = 0.0
        for ts in trajectory["timesteps"]:
            h = ts["ee_position"][2]
            if h > max_height:
                max_height = h
        summary["peak_ee_height"] = max_height

        # Find first gripper close event
        for i in range(1, len(trajectory["timesteps"])):
            prev = trajectory["timesteps"][i - 1]["gripper_open"]
            curr = trajectory["timesteps"][i]["gripper_open"]
            if prev and not curr:
                summary["grasp_event"] = {
                    "time": trajectory["timesteps"][i]["time"],
                    "ee_position": trajectory["timesteps"][i]["ee_position"],
                }
                break

        summary_path = os.path.join(group_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Export combined Isaac Lab format
    print("\nExporting combined Isaac Lab format...")
    export_isaac_lab_format(all_trajectories, output_dir)

    # Export master manifest
    manifest = {
        "num_groups": len(cloth_actions),
        "control_frequency_hz": EXPORT_FPS,
        "joint_names": JOINT_NAMES,
        "groups": [],
    }
    for action in cloth_actions:
        manifest["groups"].append(
            {
                "id": action.get("cloth_group_id", -1),
                "name": action.get("cloth_group_name", "unknown"),
                "action_name": action.name,
                "stiffness": action.get("cloth_stiffness", 0.0),
                "geometry": action.get("cloth_geometry", "unknown"),
                "dataset_folder": action.get("cloth_dataset_folder", "unknown"),
            }
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Output: {output_dir}")
    print(f"  Groups exported: {len(cloth_actions)}")
    print("  Files per group: joint_trajectory.json, dmp_params.json, summary.json")
    print("  Combined: isaac_lab_demos.json/hdf5, manifest.json")
    print("\nNext steps:")
    print("  1. Copy output to your Isaac Lab workspace")
    print("  2. Use load_blender_demos.py to replay in Newton")
    print("  3. Or use DMP params for adaptive online execution")


# Run export
if __name__ == "__main__":
    export_all()
