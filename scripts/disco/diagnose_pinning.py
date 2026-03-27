"""Quick diagnostic: inspect edge_rest_angle and bending properties."""
import sys
sys.path.insert(0, "scripts/disco")

import warp as wp
import newton
import numpy as np
from newton_sim_utils import *
from test_cloth_self_collision import discover_original_garments, ORIGINAL_GARMENT_DIR

wp.init()

with wp.ScopedDevice(wp.get_preferred_device()):
    config = load_config()
    builder = newton.ModelBuilder()
    add_ground_plane(builder)

    garments = discover_original_garments(ORIGINAL_GARMENT_DIR, "dress")
    if not garments:
        garments = discover_garments(config.get("garment_dir", ""))
    g = garments[0]
    print(f"Garment: {g['name']}")

    add_garment_as_cloth(builder, g, position=(0, 0, 1.0))
    builder.color()
    model = builder.finalize()

    # Inspect BEFORE zeroing
    era = model.edge_rest_angle.numpy()
    print(f"\n--- BEFORE edge_rest_angle.zero_() ---")
    print(f"edge_rest_angle shape: {era.shape}")
    print(f"edge_rest_angle range: [{era.min():.6f}, {era.max():.6f}]")
    print(f"edge_rest_angle nonzero: {np.count_nonzero(era)} / {len(era)}")
    print(f"edge_rest_angle abs mean: {np.abs(era).mean():.6f}")
    print(f"edge_rest_angle abs > 0.1: {np.sum(np.abs(era) > 0.1)}")
    print(f"edge_rest_angle abs > 0.5: {np.sum(np.abs(era) > 0.5)}")
    print(f"edge_rest_angle abs > 1.0: {np.sum(np.abs(era) > 1.0)}")

    # Zero it
    model.edge_rest_angle.zero_()
    era2 = model.edge_rest_angle.numpy()
    print(f"\n--- AFTER edge_rest_angle.zero_() ---")
    print(f"edge_rest_angle range: [{era2.min():.6f}, {era2.max():.6f}]")
    print(f"nonzero: {np.count_nonzero(era2)}")

    # Check edge_bending_properties
    if hasattr(model, "edge_bending_properties"):
        ebp = model.edge_bending_properties.numpy()
        print(f"\n--- edge_bending_properties ---")
        print(f"shape: {ebp.shape}")
        print(f"col 0 (ke) range: [{ebp[:, 0].min():.6e}, {ebp[:, 0].max():.6e}]")
        if ebp.shape[1] > 1:
            print(f"col 1 (kd) range: [{ebp[:, 1].min():.6e}, {ebp[:, 1].max():.6e}]")
        print(f"col 0 nonzero: {np.count_nonzero(ebp[:, 0])}")

    # Mass / inv_mass
    m = model.particle_mass.numpy()
    im = model.particle_inv_mass.numpy()
    print(f"\n--- Particle mass ---")
    print(f"mass range: [{m.min():.2e}, {m.max():.2e}]")
    print(f"inv_mass range: [{im.min():.2e}, {im.max():.2e}]")
    print(f"zero mass: {np.sum(m == 0)}, zero inv_mass: {np.sum(im == 0)}")

    # Flags
    f = model.particle_flags.numpy()
    print(f"\n--- Particle flags ---")
    print(f"all ACTIVE: {np.all(f & 1)}")
    print(f"inactive count: {np.sum((f & 1) == 0)}")
