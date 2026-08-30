#!/usr/bin/env python3
# By David Clabaugh with Claude Code
#
# Download all Newton-supported robot assets to local cache.
#
# Usage:
#   uv run python scripts/download_robot_assets.py
#
# This pre-caches every robot model so Blender setup scripts can reference
# the local mesh files without needing Newton installed inside Blender.
# After running, copy the printed paths into the ASSET_DIR configs of each
# Blender setup script.

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure Newton is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newton._src.utils.download_assets import download_asset, download_git_folder

_MENAGERIE_GIT_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"

# All Newton-supported robot assets
ASSETS = {
    "franka_emika_panda": {
        "source": "newton-assets",
        "asset_name": "franka_emika_panda",
        "description": "Franka Emika Panda FR3 — 7-DOF arm + 2-DOF parallel gripper",
        "format": "URDF",
        "urdf_path": "urdf/fr3_franka_hand.urdf",
    },
    "universal_robots_ur5e": {
        "source": "menagerie",
        "folder": "universal_robots_ur5e",
        "description": "Universal Robots UR5e — 6-DOF industrial arm",
        "format": "MJCF",
        "mjcf_file": "ur5e.xml",
    },
    "robotiq_2f85": {
        "source": "menagerie",
        "folder": "robotiq_2f85",
        "description": "Robotiq 2F-85 — 2-DOF parallel jaw gripper",
        "format": "MJCF",
        "mjcf_file": "2f85.xml",
    },
    "leap_hand": {
        "source": "menagerie",
        "folder": "leap_hand",
        "description": "LEAP Hand (left) — 16-DOF dexterous hand",
        "format": "MJCF",
        "mjcf_file": "left_hand.xml",
    },
    "shadow_dexee": {
        "source": "menagerie",
        "folder": "shadow_dexee",
        "description": "Shadow DEX-EE — 20+ DOF dexterous hand",
        "format": "MJCF",
        "mjcf_file": "scene.xml",
    },
}


def download_all():
    """Download all robot assets and return a path map."""
    paths = {}
    print("=" * 65)
    print("DOWNLOADING ALL NEWTON-SUPPORTED ROBOT ASSETS")
    print("=" * 65)

    for key, info in ASSETS.items():
        print(f"\n  [{key}] {info['description']}")
        print(f"  Format: {info['format']}")

        try:
            if info["source"] == "newton-assets":
                asset_path = download_asset(info["asset_name"])
            else:
                asset_path = download_git_folder(_MENAGERIE_GIT_URL, info["folder"])

            paths[key] = str(asset_path)
            print(f"  Cached: {asset_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            paths[key] = None

    return paths


def main():
    paths = download_all()

    # Write paths to JSON config for Blender scripts
    config_path = Path(__file__).resolve().parent / "blender" / "robot_asset_paths.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    for key, path in paths.items():
        if path:
            config[key] = path

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 65}")
    print("DONE — Asset paths saved to:")
    print(f"  {config_path}")
    print("\nBlender scripts will auto-load from this JSON,")
    print("or you can paste these paths into the ASSET_DIR configs manually:")
    print(f"{'=' * 65}")
    for key, path in paths.items():
        if path:
            print(f"  {key:30s} → {path}")
        else:
            print(f"  {key:30s} → FAILED (re-run to retry)")


if __name__ == "__main__":
    main()
