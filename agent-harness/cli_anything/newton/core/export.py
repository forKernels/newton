# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export pipeline for Newton CLI.

Supports exporting simulation results to USD, JSON, and CBOR formats
using Newton's built-in viewer backends.
"""

from __future__ import annotations

import json
import os


def create_viewer(
    viewer_type: str = "usd",
    output_path: str | None = None,
    fps: float = 60.0,
    up_axis: str = "y",
):
    """Create a Newton viewer for recording simulation output.

    Args:
        viewer_type: Viewer type — "usd", "file", "null".
        output_path: Output file path.
        fps: Frames per second for recording.
        up_axis: Up axis ("y" or "z").

    Returns:
        Viewer instance.
    """
    if viewer_type not in ("usd", "file", "null"):
        raise ValueError(f"Unknown viewer type: {viewer_type}. Use 'usd', 'file', or 'null'.")

    import newton.viewer as nv

    if viewer_type == "usd":
        if output_path is None:
            output_path = "output.usd"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        return nv.ViewerUSD(
            path=output_path,
            fps=int(fps),
            up_axis=up_axis.upper(),
        )
    elif viewer_type == "file":
        if output_path is None:
            output_path = "output.json"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        return nv.ViewerFile(path=output_path)
    else:  # null — already validated above
        return nv.ViewerNull()


def export_state_json(state, model, output_path: str, pretty: bool = True) -> dict:
    """Export the current simulation state to a JSON file.

    Args:
        state: Newton State object.
        model: Newton Model object.
        output_path: Path to write JSON file.
        pretty: Pretty-print JSON with indentation.

    Returns:
        Dict with export info (path, size, counts).
    """

    data = {
        "body_count": model.body_count,
        "particle_count": model.particle_count,
        "joint_count": model.joint_count,
    }

    if model.particle_count > 0:
        try:
            data["particle_positions"] = state.particle_q.numpy().tolist()
            data["particle_velocities"] = state.particle_qd.numpy().tolist()
        except Exception:
            pass

    if model.body_count > 0:
        try:
            data["body_transforms"] = state.body_q.numpy().tolist()
            data["body_velocities"] = state.body_qd.numpy().tolist()
        except Exception:
            pass

    if model.joint_count > 0:
        try:
            data["joint_positions"] = state.joint_q.numpy().tolist()
            data["joint_velocities"] = state.joint_qd.numpy().tolist()
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2 if pretty else None)

    file_size = os.path.getsize(output_path)
    return {
        "output": output_path,
        "file_size": file_size,
        "format": "json",
        "body_count": model.body_count,
        "particle_count": model.particle_count,
    }


def list_export_formats() -> list[dict]:
    """List available export formats.

    Returns:
        List of format info dicts.
    """
    return [
        {
            "format": "usd",
            "extension": ".usd",
            "description": "Universal Scene Description — time-sampled meshes, transforms, materials",
            "requires": "usd-core>=25.5",
        },
        {
            "format": "json",
            "extension": ".json",
            "description": "JSON state snapshot — positions, velocities, model structure",
            "requires": None,
        },
        {
            "format": "file",
            "extension": ".json/.bin",
            "description": "ViewerFile recording — full model + state history (JSON or CBOR)",
            "requires": "cbor2>=5.7.0 (for .bin)",
        },
    ]
