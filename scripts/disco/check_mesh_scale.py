"""Check garment mesh scale at each stage of the pipeline."""
import sys
sys.path.insert(0, "scripts/disco")

import numpy as np
from newton_sim_utils import load_mesh_from_file, load_garment_mesh

path = r"C:\_SimObj\ClothDataset\dress_sleeveless_2550\dress_sleeveless_000YCTJ9HS\dress_sleeveless_000YCTJ9HS_sim.obj"

# 1. Raw OBJ
raw_v, raw_f = load_mesh_from_file(path)
ext = raw_v.max(axis=0) - raw_v.min(axis=0)
print("=== RAW OBJ (as loaded from file) ===")
print(f"  Extents: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}")
print(f"  Range: x=[{raw_v[:,0].min():.2f}, {raw_v[:,0].max():.2f}]")
print(f"         y=[{raw_v[:,1].min():.2f}, {raw_v[:,1].max():.2f}]")
print(f"         z=[{raw_v[:,2].min():.2f}, {raw_v[:,2].max():.2f}]")
units = "cm" if ext.max() > 10 else "m"
print(f"  Likely units: {units}")

# 2. After load_garment_mesh (cm->m conversion, flip, center)
g = {"path": path, "is_clean": False}
proc_v, proc_f = load_garment_mesh(g)
ext2 = proc_v.max(axis=0) - proc_v.min(axis=0)
print()
print("=== After load_garment_mesh (cm->m, flip, center, floor) ===")
print(f"  Extents: {ext2[0]:.4f} x {ext2[1]:.4f} x {ext2[2]:.4f} m")
print(f"  Range: x=[{proc_v[:,0].min():.4f}, {proc_v[:,0].max():.4f}]")
print(f"         y=[{proc_v[:,1].min():.4f}, {proc_v[:,1].max():.4f}]")
print(f"         z=[{proc_v[:,2].min():.4f}, {proc_v[:,2].max():.4f}]")

# 3. Expected real-world dimensions
print()
print("=== Scale check ===")
print(f"  Actual width:  {ext2[0]:.3f}m")
print(f"  Actual height: {ext2[1]:.3f}m")
print(f"  Actual depth:  {ext2[2]:.3f}m")
print(f"  Expected dress: ~0.4-0.6m wide, ~0.7-1.0m tall")
if ext2.max() < 0.1:
    print("  *** WARNING: Mesh is WAY too small! Likely double-converted cm->m ***")
    print(f"  *** If raw is in cm ({ext[0]:.1f}x{ext[1]:.1f}), do NOT multiply by 0.01 ***")
elif ext2.max() > 5:
    print("  *** WARNING: Mesh is too large! Likely still in cm ***")
elif ext2.max() > 0.3 and ext2.max() < 2.0:
    print("  Scale looks reasonable for a garment")
else:
    print("  Scale might be off")

# 4. Triangle metrics at current scale
areas = []
edge_lengths = []
for f in proc_f:
    v0, v1, v2 = proc_v[f[0]], proc_v[f[1]], proc_v[f[2]]
    areas.append(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
    edge_lengths.extend([
        np.linalg.norm(v1 - v0),
        np.linalg.norm(v2 - v1),
        np.linalg.norm(v0 - v2),
    ])
areas = np.array(areas)
edge_lengths = np.array(edge_lengths)
print()
print("=== Triangle metrics ===")
print(f"  Avg area:    {areas.mean():.2e} m^2")
print(f"  Min area:    {areas.min():.2e} m^2")
print(f"  Avg edge:    {edge_lengths.mean():.4f} m ({edge_lengths.mean()*100:.2f} cm)")
print(f"  Min edge:    {edge_lengths.min():.6f} m ({edge_lengths.min()*100:.4f} cm)")
print(f"  Max edge:    {edge_lengths.max():.4f} m ({edge_lengths.max()*100:.2f} cm)")

# 5. Also check if 'clean' garments are available and compare
clean_path = path.replace("_sim.obj", "_clean.obj")
import os
if os.path.exists(clean_path):
    clean_v, _ = load_mesh_from_file(clean_path)
    ext_c = clean_v.max(axis=0) - clean_v.min(axis=0)
    print()
    print("=== Clean mesh (for comparison) ===")
    print(f"  Extents: {ext_c[0]:.4f} x {ext_c[1]:.4f} x {ext_c[2]:.4f}")
    clean_units = "cm" if ext_c.max() > 10 else "m"
    print(f"  Likely units: {clean_units}")

# 6. Check the MariaDataset garments too (the ones used by default config)
maria_path = r"C:/Users/davidclabaugh/Documents/Blender/SimObj/_ClothObj_Dataset/MariaDataset"
if os.path.exists(maria_path):
    import glob
    objs = glob.glob(os.path.join(maria_path, "*/*/*.obj"))
    if objs:
        mv, _ = load_mesh_from_file(objs[0])
        ext_m = mv.max(axis=0) - mv.min(axis=0)
        print()
        print(f"=== MariaDataset garment (for comparison) ===")
        print(f"  File: {os.path.basename(objs[0])}")
        print(f"  Extents: {ext_m[0]:.2f} x {ext_m[1]:.2f} x {ext_m[2]:.2f}")
        maria_units = "cm" if ext_m.max() > 10 else "m"
        print(f"  Likely units: {maria_units}")
