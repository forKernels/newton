"""Inspect tri_poses format and shape."""

import sys

sys.path.insert(0, "scripts/disco")
import warp as wp

import newton

wp.init()
b = newton.ModelBuilder()
b.add_cloth_grid(
    pos=(0, 0, 0), rot=wp.quat_identity(), vel=(0, 0, 0), dim_x=3, dim_y=3, cell_x=0.1, cell_y=0.1, mass=0.1
)
b.color()
m = b.finalize()

tp = m.tri_poses.numpy()
print(f"tri_poses shape: {tp.shape}, dtype: {tp.dtype}")
print(f"First 3 tri_poses:\n{tp[:3]}")
print(f"tri_areas shape: {m.tri_areas.numpy().shape}")
print(f"tri_areas[:3]: {m.tri_areas.numpy()[:3]}")
print(f"tri_indices shape: {m.tri_indices.numpy().shape}")
print(f"tri_materials shape: {m.tri_materials.numpy().shape}")
print(f"tri_materials[:3]:\n{m.tri_materials.numpy()[:3]}")
print(f"edge_rest_angle shape: {m.edge_rest_angle.numpy().shape}")
print(f"edge_rest_length shape: {m.edge_rest_length.numpy().shape}")

# Check what evaluate_stvk_force_hessian uses
print("\nModel attributes with 'rest' or 'pose':")
for a in sorted(dir(m)):
    if "rest" in a.lower() or "pose" in a.lower():
        attr = getattr(m, a)
        if hasattr(attr, "shape"):
            print(f"  {a}: shape={attr.shape}, dtype={attr.dtype}")
        elif hasattr(attr, "numpy"):
            arr = attr.numpy()
            print(f"  {a}: shape={arr.shape}")
        else:
            print(f"  {a}: {type(attr)}")
