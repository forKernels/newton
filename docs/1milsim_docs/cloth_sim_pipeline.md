# 1 Million Animated USDA — Cloth Simulation Pipeline

## Goal
Generate **1,000,000+ animated USDA files** from Blender cloth/RBD simulations
for training cloth manipulation policies in Newton / Isaac Lab.

---

## Garment Dataset (Prepped)

All garment categories have been cleaned and prepped with `clean_cloth_dataset.py`:
scale cm→m, origin centered, floor at Z=0, shade smooth, solidify (0.005m thickness).

| Category | Garments | Variants | Fabric Types |
|---|---|---|---|
| dress_sleeveless_2550 | 2,550 | pre-existing | default |
| jacket_2200 | 2,200 | pre-existing | default |
| jacket_hood_2700 | 2,700 | 2,700 | default |
| pants_straight_sides_1000 | 1,000 | 1,000 | default |
| skirt_2_panels_1200 | 1,200 | 1,200 | default |
| skirt_4_panels_1600 | 1,600 | 1,600 | default |
| skirt_8_panels_1000 | 1,000 | 1,000 | default |
| tee_2300 | 2,300 | 2,300 | default |
| tee_sleeveless_1800 | 1,800 | 1,800 | default |
| wb_pants_straight_1500 | 1,500 | 7,500 | cotton, polyester, jersey, wool, denim |
| wb_dress_sleeveless_2600 | 2,600 | 13,000 | cotton, polyester, jersey, wool, denim |
| **Total** | **~21,450** | **~35,300** | |

---

## Simulation Scripts

All scripts live in `scripts/disco/` and run headless via Blender:
```bash
blender --background --python scripts/disco/<script>.py -- [args]
```

### Shared Infrastructure
| File | Purpose |
|---|---|
| `cloth_sim_utils.py` | Scene management, discovery, collision/cloth setup, baking, USDA export, config loading |
| `sim_config.json` | Portable path configuration (edit for new machine) |
| `clean_cloth_dataset.py` | Mesh prep pipeline (scale, origin, solidify, fabric variants) |
| `zip_categories.py` | Zip each category for upload (standalone Python, no Blender) |

### Cloth Simulation Scripts

| Script | Mode | Description | Status |
|---|---|---|---|
| `cloth_drop_sim.py` | drop | Garment released from rest above furniture (gravity only) | Done |
| `cloth_drop_sim.py` | throw | Garment launched with velocity toward furniture | Done |
| `cloth_stack_sim.py` | stack | 2-5 garments dropped at staggered heights, pile up | Done |
| `cloth_drag_sim.py` | drag | Pinned edge vertices keyframed across surface, rest drags | Done |
| `cloth_prop_sim.py` | prop | Cloth dropped onto furniture with props (computer, lamp, bottle) | Planned |
| `cloth_wind_sim.py` | wind | Force field blows cloth (flags, curtains, laundry) | Planned |
| `cloth_drape_sim.py` | drape | Cloth placed flat then draped over furniture edges | Planned |
| `cloth_curtain_sim.py` | curtain | Cloth pinned along top edge, slides open/close on rail | Planned |

### RBD / Other Simulation Scripts (Planned)

| Script | Description |
|---|---|
| `rbd_lego_drop_sim.py` | Separated Lego bricks with rigid body dynamics, dropped from height |
| `rbd_pool_sim.py` | Pool table break shots, ball collisions |
| `rope_sim.py` | Rope/chain dynamics (3-section staff with chain + silk ribbon) |

---

## Output Structure

```
_Sims/
  {furniture_name}/
    drop/{preset}/{garment_id}_{idx}.usda + .json
    throw/{preset}/{garment_id}_{idx}.usda + .json
    stack/{preset}/stack{N}_{hash}_{idx}.usda + .json
    drag/{preset}/{garment_id}_{idx}.usda + .json
    prop/{preset}/{garment_id}_{idx}.usda + .json
    ...
```

Each `.usda` has an animated mesh (cloth simulation baked).
Each `.json` sidecar has metadata (furniture, garment, preset, positions, seed, bake time).

---

## Cloth Presets (Blender Built-in)

| Preset | Behavior |
|---|---|
| Silk | Light, flowing, low friction |
| Cotton | Medium weight, natural drape |
| Denim | Heavy, stiff, high friction |
| Leather | Very stiff, minimal drape |
| Rubber | High friction, bouncy |

---

## Volume Estimate to 1M USDA

### Furniture
- **Local**: ~24 droppable pieces (chairs, tables, sofas, beds, desks)
- **Available**: ~100 total (user has more .blend assets)

### Combinations (with 100 furniture)

| Mode | Formula | Estimate |
|---|---|---|
| Drop | 100 furn x 21K garments x 5 presets x 1 sample | 10,500,000 |
| Throw | 100 furn x 21K garments x 5 presets x 1 sample | 10,500,000 |
| Stack | 100 furn x 5 presets x 500 samples | 250,000 |
| Drag | 100 furn x 21K garments x 5 presets x 1 sample | 10,500,000 |
| Props | 100 furn x 21K garments x 5 presets x 1 sample | 10,500,000 |

**Even just drop mode with 2 samples per combo exceeds 1M easily.**

### Practical batching strategy
Pick a manageable subset per run:
```bash
# 10 furniture x all garments x 1 preset x 2 samples = ~430K USDA
blender --background --python cloth_drop_sim.py -- \
    --config sim_config.json \
    --cloth-preset Cotton --num-samples 2
```

---

## Portable Config

Edit `scripts/disco/sim_config.json` when moving to a new machine:
```json
{
  "furniture_dir": "D:/_blender/_myBlender/SimulationWork/seedAssets/scenes",
  "garment_dir": "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Maria_Set",
  "output_dir": "D:/_blender/_myBlender/SimulationWork/ClothDataset/_Sims",
  "blender_preset_dir": "C:/Program Files/Blender Foundation/Blender 4.5/4.5/scripts/presets/cloth"
}
```

All scripts accept `--config sim_config.json` to load these paths as defaults.
CLI arguments always override config values.

---

## Pipeline Flow

```
1. PREP     clean_cloth_dataset.py    →  *_sim_prep.obj (scale, origin, solidify)
2. SIM      cloth_{drop,stack,drag}_sim.py  →  animated .usda + .json metadata
3. ZIP      zip_categories.py         →  per-category .zip for upload
4. TRAIN    load_blender_demos.py     →  HDF5 dataset for Isaac Lab
```

---

## Simulation Physics Setup

- **Clothing**: Blender Cloth modifier (Silk/Cotton/Denim/Leather/Rubber presets)
- **Furniture**: Blender Collision modifier (passive rigid body collider)
  - `thickness_outer = 0.002`
  - `cloth_friction = 5.0`
- **Props** (planned): Collision modifier for static props, optional RBD for reactive props
- **Legos** (planned): Blender Rigid Body dynamics (active, separated bricks)

---

## Quick Start

```bash
# 1. Edit config for your machine
vim scripts/disco/sim_config.json

# 2. Run a test drop sim (1 furniture, 1 preset, 1 sample)
blender --background --python scripts/disco/cloth_drop_sim.py -- \
    --config scripts/disco/sim_config.json \
    --furniture "Chair_01.blend" \
    --cloth-preset Cotton --num-samples 1

# 3. Run stack sim
blender --background --python scripts/disco/cloth_stack_sim.py -- \
    --config scripts/disco/sim_config.json \
    --cloth-preset Cotton --num-samples 5

# 4. Run drag sim
blender --background --python scripts/disco/cloth_drag_sim.py -- \
    --config scripts/disco/sim_config.json \
    --cloth-preset Silk --num-samples 3

# 5. Batch: all furniture, all presets
blender --background --python scripts/disco/cloth_drop_sim.py -- \
    --config scripts/disco/sim_config.json \
    --cloth-preset all --num-samples 2

# 6. Zip for upload
python scripts/disco/zip_categories.py --config scripts/disco/sim_config.json
```
