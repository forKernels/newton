# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton backend: verifies Newton + Warp are installed and importable.

Newton IS the backend — it's a Python library, not an external executable.
This module validates the environment and provides version/device info.
"""

from __future__ import annotations


def find_newton() -> dict:
    """Verify Newton is installed and return version info.

    Returns:
        Dict with newton version, warp version, device info.

    Raises:
        RuntimeError: If Newton or Warp are not installed.
    """
    try:
        import newton
    except ImportError:
        raise RuntimeError(
            "Newton is not installed. Install it with:\n"
            "  pip install newton-sim\n"
            "Or from source:\n"
            "  cd /path/to/newton && pip install -e .\n"
            "  # or: uv sync --extra examples"
        )

    try:
        import warp as wp
    except ImportError:
        raise RuntimeError("NVIDIA Warp is not installed. Install it with:\n  pip install warp-lang>=1.11.0")

    info = {
        "newton_version": getattr(newton, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
    }

    # Device info
    try:
        wp.init()
        devices = wp.get_devices()
        info["devices"] = [str(d) for d in devices]
        cuda_devices = [d for d in devices if "cuda" in str(d).lower()]
        info["cuda_available"] = len(cuda_devices) > 0
        info["cuda_device_count"] = len(cuda_devices)
    except Exception as e:
        info["devices"] = []
        info["cuda_available"] = False
        info["device_error"] = str(e)

    return info


def is_available() -> bool:
    """Check if Newton is available."""
    try:
        find_newton()
        return True
    except RuntimeError:
        return False


def get_examples() -> list[dict]:
    """List available Newton examples.

    Returns:
        List of dicts with example name, module, description.
    """
    import pkgutil

    examples = []
    try:
        import newton.examples

        examples_path = newton.examples.__path__

        # Discover example modules
        for importer, modname, ispkg in pkgutil.walk_packages(examples_path, prefix="newton.examples."):
            if modname.startswith("newton.examples.example_") or (ispkg and not modname.endswith("__pycache__")):
                # Extract example name from module
                parts = modname.split(".")
                name = parts[-1]
                if name.startswith("example_"):
                    name = name[len("example_") :]
                examples.append(
                    {
                        "name": name,
                        "module": modname,
                    }
                )
    except ImportError:
        pass

    return examples
