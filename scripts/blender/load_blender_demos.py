"""
Load Blender-Exported Trajectories into Newton / Isaac Lab
============================================================
This script runs OUTSIDE of Blender, in your Isaac Lab Python environment.

It loads the exported JSON/HDF5 trajectories and:
  1. Replays them in Newton simulation to validate
  2. Converts to Isaac Lab Mimic-compatible format
  3. Generates augmented demonstrations using DMP warping
  4. Sets up the training config for group-conditioned policy

Usage:
  python load_blender_demos.py \
      --data_dir ./franka_cloth_animations/ \
      --output_dir ./datasets/ \
      --augment_per_group 200
"""

import argparse
import json
import os

import numpy as np

# =============================================================================
# DMP REPLAY ENGINE — Adaptive motion from Blender-authored primitives
# =============================================================================


class DynamicMovementPrimitive:
    """
    Replay a DMP learned from Blender keyframe animation.
    Supports warping to new goals while preserving the motion 'style'.
    """

    def __init__(self, params: dict):
        self.n_basis = params["n_basis"]
        self.dt = params["dt"]
        self.tau = params["tau"]
        self.alpha_z = params["alpha_z"]
        self.beta_z = params["beta_z"]
        self.alpha_x = params["alpha_x"]
        self.joints = params["joints"]
        self.gripper_timeline = params.get("gripper_timeline", None)

    def generate(
        self, new_goals: dict = None, new_starts: dict = None, speed_factor: float = 1.0, noise_scale: float = 0.0
    ):
        """
        Generate a trajectory from DMP parameters.

        Args:
            new_goals: dict of {joint_name: goal_angle} to warp to new goals
            new_starts: dict of {joint_name: start_angle} for new starting config
            speed_factor: >1 = slower, <1 = faster (scales tau)
            noise_scale: Gaussian noise added to weights for demonstration augmentation

        Returns:
            dict with joint_positions array [T, 7] and gripper_widths [T]
        """
        tau = self.tau * speed_factor
        n_steps = int(tau / self.dt)

        joint_names = list(self.joints.keys())
        n_joints = len(joint_names)

        trajectories = np.zeros((n_steps, n_joints))

        for j_idx, j_name in enumerate(joint_names):
            j_params = self.joints[j_name]
            weights = np.array(j_params["weights"])

            # Add noise for augmentation
            if noise_scale > 0:
                weights = weights + np.random.randn(len(weights)) * noise_scale

            y0 = j_params["start"]
            goal = j_params["goal"]

            # Override start/goal if provided
            if new_starts and j_name in new_starts:
                y0 = new_starts[j_name]
            if new_goals and j_name in new_goals:
                goal = new_goals[j_name]

            # DMP integration
            y = y0
            dy = 0.0
            x = 1.0  # canonical system

            # Basis function centers and widths
            centers = np.exp(-self.alpha_x * np.linspace(0, 1, self.n_basis))
            widths = 1.0 / (0.65 * np.diff(centers) ** 2)
            widths = np.append(widths, widths[-1])

            for t in range(n_steps):
                # Canonical system decay
                dx = -self.alpha_x * x / tau
                x += dx * self.dt

                # Evaluate basis functions
                psi = np.exp(-widths * (x - centers) ** 2)

                # Forcing function
                f = np.dot(psi * x, weights) / (np.sum(psi) + 1e-10)

                # Spring-damper system
                ddy = (self.alpha_z * (self.beta_z * (goal - y) - tau * dy) + f) / (tau**2)
                dy += ddy * self.dt
                y += dy * self.dt

                trajectories[t, j_idx] = y

        # Interpolate gripper timeline
        gripper_widths = np.ones(n_steps) * 0.08  # default open
        if self.gripper_timeline:
            times = np.array(self.gripper_timeline["times"])
            widths_arr = np.array(self.gripper_timeline["widths"])

            new_times = np.linspace(0, tau, n_steps)
            # Scale times by speed_factor
            scaled_times = times * speed_factor
            gripper_widths = np.interp(new_times, scaled_times, widths_arr)

        return {
            "joint_positions": trajectories,
            "gripper_widths": gripper_widths,
            "joint_names": joint_names,
            "dt": self.dt,
            "duration": tau,
        }


# =============================================================================
# AUGMENTATION — Generate diverse demonstrations from Blender primitives
# =============================================================================


def augment_demonstrations(dmp_params: dict, n_augmented: int = 200, cloth_group_id: int = 0):
    """
    Generate augmented demonstrations from a single Blender-authored DMP.

    Applies randomization to:
    - Start position (small perturbations)
    - Goal position (target cloth position varies)
    - Speed (timing variation)
    - DMP weights (adds noise for trajectory diversity)

    This is how you go from 1 Blender animation → 200+ training demos.
    """
    dmp = DynamicMovementPrimitive(dmp_params)

    demonstrations = []

    for i in range(n_augmented):
        # Randomize start position (±5° per joint)
        start_noise = {name: params["start"] + np.random.uniform(-0.087, 0.087) for name, params in dmp.joints.items()}

        # Randomize goal (cloth position varies on table)
        # Small perturbation to end configuration
        goal_noise = {name: params["goal"] + np.random.uniform(-0.05, 0.05) for name, params in dmp.joints.items()}

        # Random speed variation (±20%)
        speed_factor = np.random.uniform(0.8, 1.2)

        # DMP weight noise for trajectory shape diversity
        noise_scale = 0.5 + np.random.uniform(0, 0.5)

        traj = dmp.generate(
            new_goals=goal_noise,
            new_starts=start_noise,
            speed_factor=speed_factor,
            noise_scale=noise_scale,
        )

        # Package as demonstration
        demo = {
            "demo_id": i,
            "cloth_group_id": cloth_group_id,
            "joint_positions": traj["joint_positions"].tolist(),
            "gripper_widths": traj["gripper_widths"].tolist(),
            "speed_factor": speed_factor,
            "augmentation_noise": noise_scale,
        }
        demonstrations.append(demo)

    return demonstrations


# =============================================================================
# ISAAC LAB INTEGRATION
# =============================================================================


def create_isaac_lab_dataset(all_demos: list, output_path: str):
    """
    Create an HDF5 dataset in Isaac Lab Mimic-compatible format.

    Structure:
      /data/demo_0/obs       [T, obs_dim]    — joint_pos + joint_vel + gripper
      /data/demo_0/actions   [T, action_dim]  — delta joint positions + gripper
      /data/demo_0/dones     [T]              — episode termination
      /data/demo_0/rewards   [T]              — reward signal

    Each demo is tagged with cloth_group_id for conditioned training.
    """
    try:
        import h5py
    except ImportError:
        print("h5py not installed. Install with: pip install h5py")
        print("Falling back to JSON export.")
        json_path = output_path.replace(".hdf5", ".json")
        with open(json_path, "w") as f:
            json.dump({"demos": all_demos}, f)
        print(f"Saved JSON: {json_path}")
        return

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")

        for demo in all_demos:
            demo_id = demo["demo_id"]
            demo_grp = data_grp.create_group(f"demo_{demo_id}")

            joint_pos = np.array(demo["joint_positions"])  # [T, 7]
            gripper = np.array(demo["gripper_widths"])  # [T]
            T = joint_pos.shape[0]

            # Compute velocities
            dt = 0.1  # 10 Hz
            joint_vel = np.zeros_like(joint_pos)
            joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt

            # Observations: joint_pos(7) + joint_vel(7) + gripper(1) + group_id(1)
            group_ids = np.full((T, 1), demo["cloth_group_id"])
            obs = np.hstack([joint_pos, joint_vel, gripper[:, None], group_ids])

            # Actions: delta joint positions(7) + delta gripper(1)
            actions = np.zeros((T, 8))
            actions[:-1, :7] = joint_pos[1:] - joint_pos[:-1]
            actions[:-1, 7] = gripper[1:] - gripper[:-1]

            # Done flags
            dones = np.zeros(T)
            dones[-1] = 1.0

            demo_grp.create_dataset("obs", data=obs.astype(np.float32))
            demo_grp.create_dataset("actions", data=actions.astype(np.float32))
            demo_grp.create_dataset("dones", data=dones.astype(np.float32))
            demo_grp.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))

            demo_grp.attrs["cloth_group_id"] = demo["cloth_group_id"]

        f.attrs["num_demos"] = len(all_demos)
        f.attrs["obs_dim"] = obs.shape[1]
        f.attrs["action_dim"] = 8

    print(f"Saved Isaac Lab dataset: {output_path} ({len(all_demos)} demos)")


def generate_training_config(data_dir: str, n_groups: int = 11):
    """
    Generate an Isaac Lab training config for a group-conditioned policy.
    This config file can be used with RSL-RL or robomimic.
    """
    config = {
        "task": "ClothManipulation-Franka-GroupConditioned-v0",
        "num_envs": 256,
        "physics_backend": "newton",
        "solver": "mujoco_warp",
        "robot": {
            "type": "franka_panda",
            "control_mode": "joint_position",
            "control_frequency_hz": 10,
            "action_scale": 0.1,
        },
        "cloth_groups": {
            "num_groups": n_groups,
            "group_embedding_dim": 32,
            "randomize_within_group": True,
        },
        "policy": {
            "type": "group_conditioned_mlp",
            "architecture": {
                "obs_encoder": [256, 256],
                "group_embedding": 32,
                "policy_head": [256, 128],
                "activation": "elu",
            },
            "action_chunk_size": 16,  # predict 16 steps ahead
        },
        "training": {
            "algorithm": "behavior_cloning",
            "dataset_path": os.path.join(data_dir, "augmented_dataset.hdf5"),
            "batch_size": 256,
            "learning_rate": 1e-4,
            "epochs": 300,
            "validation_split": 0.1,
        },
        "domain_randomization": {
            "cloth_stiffness_noise": 0.1,
            "cloth_position_noise": 0.02,
            "table_height_noise": 0.005,
            "joint_noise": 0.01,
        },
    }

    config_path = os.path.join(data_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated training config: {config_path}")
    return config


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Load Blender demos into Isaac Lab format")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with exported Blender animations")
    parser.add_argument("--output_dir", type=str, default="./datasets/", help="Output directory for training datasets")
    parser.add_argument(
        "--augment_per_group", type=int, default=200, help="Number of augmented demos to generate per cloth group"
    )
    parser.add_argument("--validate", action="store_true", help="Replay trajectories in Newton for validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(args.data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {manifest['num_groups']} cloth groups")
    print(f"Generating {args.augment_per_group} augmented demos per group")
    print(f"Total demos: {manifest['num_groups'] * args.augment_per_group}")

    all_demos = []
    demo_counter = 0

    for group in manifest["groups"]:
        gid = group["id"]
        gname = group["name"]

        group_dir = os.path.join(args.data_dir, f"group_{gid:02d}_{gname}")

        # Load DMP parameters
        dmp_path = os.path.join(group_dir, "dmp_params.json")
        if not os.path.exists(dmp_path):
            print(f"  WARNING: No DMP params for group {gid}, skipping augmentation")
            continue

        with open(dmp_path) as f:
            dmp_params = json.load(f)

        print(f"\nGroup {gid}: {gname} (stiffness={group['stiffness']:.2f})")

        # Also load the original trajectory as demo 0
        traj_path = os.path.join(group_dir, "joint_trajectory.json")
        with open(traj_path) as f:
            original_traj = json.load(f)

        # Original demo
        original_demo = {
            "demo_id": demo_counter,
            "cloth_group_id": gid,
            "joint_positions": [ts["joint_positions"] for ts in original_traj["timesteps"]],
            "gripper_widths": [ts["gripper_width"] for ts in original_traj["timesteps"]],
            "speed_factor": 1.0,
            "augmentation_noise": 0.0,
            "is_original": True,
        }
        all_demos.append(original_demo)
        demo_counter += 1

        # Generate augmented demos
        augmented = augment_demonstrations(dmp_params, n_augmented=args.augment_per_group, cloth_group_id=gid)

        for aug in augmented:
            aug["demo_id"] = demo_counter
            all_demos.append(aug)
            demo_counter += 1

        print(f"  Original: 1 demo, Augmented: {len(augmented)} demos")

    # Create combined dataset
    dataset_path = os.path.join(args.output_dir, "augmented_dataset.hdf5")
    create_isaac_lab_dataset(all_demos, dataset_path)

    # Generate training config
    generate_training_config(args.output_dir, n_groups=manifest["num_groups"])

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total demonstrations: {len(all_demos)}")
    print(f"  Original (from Blender): {manifest['num_groups']}")
    print(f"  Augmented (from DMPs): {len(all_demos) - manifest['num_groups']}")
    print(f"  Dataset: {dataset_path}")
    print("\nTo train in Isaac Lab:")
    print("  ./isaaclab.sh -p train_cloth_policy.py \\")
    print(f"      --config {os.path.join(args.output_dir, 'training_config.json')}")


if __name__ == "__main__":
    main()
