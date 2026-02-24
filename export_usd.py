"""Interactive script to select a Newton forward-simulation example and export it as USD."""

import os
import subprocess

# Forward simulation examples only (excludes diffsim, IK, selection, sensor, recording/replay)
EXAMPLES = [
    # Basic
    "basic_joints",
    "basic_pendulum",
    "basic_shapes",
    "basic_urdf",
    # Cable
    "cable_ball_joints",
    "cable_bend",
    "cable_bend_damping",
    "cable_bundle_hysteresis",
    "cable_fixed_joints",
    "cable_helix",
    "cable_pile",
    "cable_twist",
    "cable_y_junction",
    # Cloth
    "cloth_bending",
    "cloth_franka",
    "cloth_h1",
    "cloth_hanging",
    "cloth_style3d",
    "cloth_twist",
    "rolling_cloth",
    # Contact
    "nut_bolt_hydro",
    "nut_bolt_sdf",
    # MPM
    "mpm_anymal",  # requires torch
    "mpm_grain_rendering",
    "mpm_granular",
    "mpm_multi_material",
    "mpm_twoway_coupling",
    # Misc
    "falling_gift",
    "poker_cards_stacking",
    "softbody_dropping_to_cloth",
    "softbody_hanging",
    # Robot
    "robot_allegro_hand",
    "robot_anymal_c_walk",  # requires torch
    "robot_anymal_d",
    "robot_cartpole",
    "robot_g1",
    "robot_h1",
    "robot_humanoid",
    "robot_panda_hydro",
    "robot_ur10",
]

# Examples that require PyTorch (torch)
TORCH_REQUIRED = {
    "mpm_anymal",
    "robot_anymal_c_walk",
}


def main():
    print("\n=== Newton USD Export ===\n")
    print("Available forward-simulation examples:\n")

    # Print numbered list in columns
    col_width = 40
    per_row = 2
    for i, name in enumerate(EXAMPLES):
        tag = " [torch]" if name in TORCH_REQUIRED else ""
        entry = f"  [{i + 1:2d}] {name}{tag}"
        if (i + 1) % per_row == 0 or i == len(EXAMPLES) - 1:
            print(entry)
        else:
            print(entry.ljust(col_width), end="")

    print()
    while True:
        choice = input("Select example number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(EXAMPLES):
                break
            print(f"  Please enter a number between 1 and {len(EXAMPLES)}.")
        except ValueError:
            print("  Invalid input. Enter a number.")

    example_name = EXAMPLES[idx]

    # Configure num-frames
    num_frames_str = input("Number of frames [100]: ").strip()
    num_frames = int(num_frames_str) if num_frames_str else 100

    # Configure device
    device = input("Device [cuda:0]: ").strip() or "cuda:0"

    # Output path
    output_dir = "usd_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{example_name}.usd")
    output_path_input = input(f"Output path [{output_path}]: ").strip()
    if output_path_input:
        output_path = output_path_input

    # Build command — use uv run so dependencies are resolved properly
    cmd = ["uv", "run", "--extra", "examples"]
    if example_name in TORCH_REQUIRED:
        cmd.extend(["--extra", "torch-cu12"])
    cmd.extend(
        [
            "-m",
            "newton.examples",
            example_name,
            "--viewer",
            "usd",
            "--output-path",
            output_path,
            "--num-frames",
            str(num_frames),
            "--device",
            device,
        ]
    )

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        abs_path = os.path.abspath(output_path)
        print(f"\nDone! USD file saved to: {abs_path}")
    else:
        print(f"\nExample exited with code {result.returncode}.")


if __name__ == "__main__":
    main()
