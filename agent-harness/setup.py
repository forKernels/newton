# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_namespace_packages, setup

setup(
    name="cli-anything-newton",
    version="1.0.0",
    description="CLI harness for the Newton GPU-accelerated physics simulation engine",
    long_description=open("cli_anything/newton/README.md").read(),
    long_description_content_type="text/markdown",
    author="The Newton Developers",
    license="Apache-2.0",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    package_data={
        "cli_anything.newton": ["skills/*.md"],
    },
    install_requires=[
        "click>=8.0.0",
        "prompt-toolkit>=3.0.0",
    ],
    extras_require={
        "mesh": ["trimesh>=4.6.8", "scipy"],
    },
    entry_points={
        "console_scripts": [
            "cli-anything-newton=cli_anything.newton.newton_cli:main",
        ],
    },
    python_requires=">=3.10",
)
