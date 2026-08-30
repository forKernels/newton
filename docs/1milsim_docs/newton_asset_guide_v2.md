# Newton Physics — Simulation Asset Guide
## Granular · RBD · Cloth · Paper · Soft Body · Cards · Pool Table

**No fluids. No liquids. Period.**

---

## Solver → Object Type Quick Map

| Solver | Simulates | Blender Export | Newton API |
|--------|-----------|---------------|------------|
| **MuJoCo Warp** | Rigid bodies, pool balls, cards, articulated chains, robots | URDF / MJCF XML | `add_mjcf()` / `add_urdf()` |
| **VBD** | Cloth, garments, thin deformables, soft bodies | Triangulated OBJ → verts + faces | `add_cloth_mesh()` / `add_soft_mesh()` |
| **MPM** | Granular only (sand, beans, rice, gravel) | Particle positions (NOT mesh) | `add_particle_grid()` |
| **Featherstone** | Articulated rigid bodies (chains, staffs) | URDF tree | `add_urdf()` |
| **XPBD** | Ropes, springs, soft constraints | Particle chain + distance constraints | `add_particle()` |
| **Style3D** | Garments specifically | Cloth mesh | Specialized garment solver |

---

## 1. RIGID BODY DYNAMICS — Pool Table, Balls, Dominoes, Dice
**Solver:** MuJoCo Warp  
**Newton Examples:** `poker_cards_stacking`, `selection_materials`, `ik_cube_stacking`

### Pool Table Physics Constants

| Property | Value | In Meters |
|----------|-------|-----------|
| Table playing surface (9-ft) | 50" × 100" | **1.27m × 2.54m** |
| Ball diameter | 2.25" | **0.05715m** (radius: 0.028575) |
| Ball mass | 6 oz | **0.170 kg** |
| Cue ball mass | 6 oz | **0.170 kg** |
| Cushion height (from bed) | ~63.5% of ball diameter | **0.0363m** |
| Rail width | ~2" | **0.05m** |
| Pocket opening | ~4.5–5.0" | **0.114–0.127m** |
| Ball-ball restitution | 0.92–0.98 | — |
| Ball-cloth friction (sliding) | 0.15–0.25 | — |
| Ball-ball friction | 0.03–0.06 | — |
| Ball-cushion restitution | 0.80–0.90 | — |
| Cloth rolling resistance | 0.005–0.015 | — |

### Pool Table Meshes to Model in Blender

| Component | How to Model | Export As |
|-----------|-------------|-----------|
| **Table bed (slate)** | Flat box, 1.27×2.54×0.025m | Static rigid body (body=-1) |
| **Rails (4 long + 2 short)** | 6 box colliders along edges | Static rigid bodies |
| **Cushions** | Angled geometry, triangular cross-section | Static — use box/capsule approximation |
| **Pockets (6)** | Holes — model as sensor zones, NOT geometry | Detect ball entry with position check |
| **Balls (16)** | Don't model — use `add_shape_sphere()` primitive | Sphere primitives (WAY faster than mesh) |
| **Cue stick** | Tapered cylinder | Kinematic body (you control it) |
| **Visual table mesh** | Full detailed mesh with legs, felt texture | OBJ for rendering only, NOT for collision |

### Blender Modeling Tips — Pool Table
```
COLLISION geometry ≠ VISUAL geometry

Visual: Model the whole beautiful table in Blender (legs, pockets, felt, wood)
Collision: Use PRIMITIVES only:
  - Table bed:  1 box  (1.27 × 0.025 × 2.54)
  - Rail top:   4+2 boxes along edges
  - Cushions:   4+2 angled boxes (tilted ~12° inward)
  - Balls:      16 spheres (radius 0.028575)
  - Pockets:    GAPS in the rail geometry (no geometry = ball falls through)

Export visual mesh as OBJ for rendering
Build collision in MJCF/URDF using primitives
```

### MJCF Approach for Pockets
Pockets aren't modeled as geometry. Instead, leave gaps in the rails and let
gravity pull balls through. Add invisible sensors at pocket positions to detect scoring.

```xml
<!-- Rail with gap for corner pocket -->
<!-- Instead of one continuous rail, use two segments with a gap -->
<geom name="rail_long_left_1" type="box" 
      pos="-0.635 0.0182 0.60" size="0.53 0.0182 0.025"
      rgba="0.2 0.5 0.2 1"/>
<!-- gap here = pocket -->
<geom name="rail_long_left_2" type="box" 
      pos="-0.635 0.0182 1.88" size="0.53 0.0182 0.025"
      rgba="0.2 0.5 0.2 1"/>
```

### Other RBD Objects to Collect / Model

| Object | Dimensions | Density (kg/m³) | Friction | Restitution |
|--------|-----------|-----------------|----------|-------------|
| Billiard ball | r=0.02858m | 1750 (phenolic resin) | 0.05 b-b, 0.2 b-cloth | 0.95 |
| Bowling ball | r=0.1085m | 1200–1400 | 0.3 | 0.75 |
| Bowling pin | capsule h=0.38m, r=0.06m | 800 | 0.3 | 0.6 |
| Domino | box 44×22×8mm | 1400 (bone/plastic) | 0.4 | 0.3 |
| Dice (d6) | cube 16mm | 1200 | 0.3 | 0.4 |
| Marble | r=0.008m | 2500 (glass) | 0.1 | 0.9 |
| Coin | cylinder r=0.012m h=0.002m | 8900 (copper) | 0.3 | 0.5 |
| Wooden block | box various | 600–800 (varies by wood) | 0.4 | 0.3 |
| Steel ball bearing | r=0.005–0.01m | 7800 | 0.1 | 0.92 |
| Brick | box 200×100×65mm | 1900 | 0.6 | 0.2 |
| Jenga block | box 75×25×15mm | 500 (pine) | 0.4 | 0.2 |
| Bocce ball | r=0.054m | 2800 (resin) | 0.3 | 0.5 |

### RBD Simulation Ideas

| Sim | What You Learn | Extend From |
|-----|---------------|-------------|
| Pool break shot | Ball-ball elastic collision, spin, friction | Custom MJCF |
| Domino toppling | Chain reaction, spacing sensitivity | `poker_cards_stacking` |
| Jenga pull | Friction, stacking stability | `ik_cube_stacking` |
| Newton's cradle | Momentum transfer, conservation of energy | `basic_pendulum` |
| Marble run | Ramps, curves, energy conservation | `selection_materials` |
| Bowling strike | Multi-body collision cascade | Custom MJCF |
| Dice roll | Random rotation, bouncing | `selection_materials` |
| Coin spin | Gyroscopic precession, settling | `selection_materials` |

---

## 2. PLAYING CARDS — Thin Rigid Bodies
**Solver:** MuJoCo Warp  
**Newton Example:** `poker_cards_stacking`

### Card Physics

| Property | Value |
|----------|-------|
| Standard card size | 63.5mm × 88.9mm (2.5" × 3.5") |
| Thickness | 0.3mm (0.0003m) |
| Mass | ~1.8g (0.0018 kg) |
| Material density | ~1200 kg/m³ (card stock) |
| Card-card friction | 0.3–0.5 |
| Card-felt friction | 0.5–0.7 |

### How to Model Cards in Blender
```
DON'T model cards as meshes for collision.
Use box primitives: hx=0.0318, hy=0.00015, hz=0.0445

Visual mesh: Plane with card texture (for rendering)
Collision: Thin box primitive (fast, stable)

CRITICAL: Minimum collision thickness
  Real card = 0.3mm, but use ≥1mm (0.001m) for simulation stability
  Otherwise MuJoCo contact solver can miss the thin geometry
```

### Newton Code Pattern
```python
import newton
import numpy as np

builder = newton.ModelBuilder()

# Table surface
builder.add_shape_box(body=-1, pos=(0, 0, 0.4), hx=0.5, hy=0.02, hz=0.5)

# 52 cards in a deck
for i in range(52):
    body = builder.add_body(
        origin=wp.transform(
            (0.0, 0.42 + 0.0012 * i, 0.0),  # stack with slight gaps
            wp.quat_identity()
        ),
    )
    builder.add_shape_box(
        body=body,
        hx=0.0318,    # half-width
        hy=0.0005,     # half-thickness (1mm total for stability)
        hz=0.0445,     # half-height
        density=1200.0,
        friction=0.4,
    )

model = builder.finalize()
solver = newton.SolverMuJoCo()
```

### Card Simulation Ideas

| Sim | Description |
|-----|-------------|
| Card tower | Build pyramids, test structural stability |
| Deck shuffle | Riffle shuffle physics |
| Card toss | Frisbee-like flight with air drag (add custom force) |
| 52 pickup | Drop deck, cards scatter on table |
| Dealing | Kinematic hand slides cards across felt |
| House of cards collapse | Build then disturb |

---

## 3. CLOTH & GARMENTS (VBD Solver)
**Newton Examples:** `cloth_franka`, `diffsim_cloth`

### Meshes to Model / Collect

| Object | Scale (meters) | Stiffness | Notes |
|--------|---------------|-----------|-------|
| T-shirt | 0.6m tall | 500–1000 | Use your 16K garment dataset |
| Towel | 0.3–0.7m | 300–800 | Subdivided plane |
| Bed sheet | 2.0 × 2.0m | 200–500 | High quad count (200×200) |
| Napkin | 0.3 × 0.3m | 200–400 | Small plane |
| Flag | 0.5–1.0m | 300–600 | Pin top edge vertices |
| Curtain | 1.5 × 2.0m | 400–800 | Pin top row to rail |
| Tablecloth | 1.5 × 1.5m | 300–600 | Drape over table rigid body |
| Tarp | 2–3m | 2000–5000 | High stiffness, thick |
| Bandage / ribbon | 0.05 × 1.0m | 500–1500 | Narrow strip |
| Pool table felt | 1.27 × 2.54m | 3000+ | Very high stiffness (backed by slate) |

### Blender Export Checklist — Cloth
```
☐ Apply all transforms (Ctrl+A → All Transforms)
☐ Scale: 1 Blender unit = 1 meter
☐ TRIANGULATE (Ctrl+T) — VBD requires triangles
☐ Check normals (all blue in Face Orientation overlay)
☐ No loose vertices or edges
☐ Export as OBJ with normals
☐ Forward: -Z, Up: Y (matches Newton Y-up)
```

### Newton Code Pattern
```python
builder = newton.ModelBuilder()

vertices, faces = load_obj("towel.obj")

# Pin specific vertices (e.g., top edge of flag)
pinned = np.where(vertices[:, 1] > vertices[:, 1].max() - 0.01)[0]

builder.add_cloth_mesh(
    pos=vertices,
    tri=faces,
    vel=np.zeros_like(vertices),
    mass=0.001 * np.ones(len(vertices)),
    stiffness=500.0,   # cotton ~500, silk ~200, denim ~5000, felt ~3000
    damping=10.0,
)

for idx in pinned:
    builder.add_particle_constraint(idx, enabled=True)

model = builder.finalize()
solver = newton.SolverVBD()
```

### Cloth Simulations

| Sim | Description | Extend From |
|-----|-------------|-------------|
| Towel folding | Franka folds towel on table | `cloth_franka` |
| Tablecloth pull | Classic magic trick — yank cloth, objects stay | `cloth_franka` + RBD |
| Flag in wind | Pinned cloth + external force | `diffsim_cloth` |
| Curtain open/close | Cloth on rail, kinematic rod | `cloth_franka` |
| Pool table re-felt | Drape cloth over flat surface | `diffsim_cloth` |
| Bandage wrapping | Narrow strip around cylindrical body | `cloth_franka` |

---

## 4. PAPER & THIN SHELLS (Rigid or Stiff Cloth)
**Newton Examples:** `poker_cards_stacking`, `falling_gift`

Paper is tricky — it sits between rigid and cloth depending on stiffness:
- **Stiff paper** (cardboard, card stock) → MuJoCo rigid body (thin box)
- **Flexible paper** (printer paper, tissue) → VBD cloth with HIGH stiffness
- **Pre-folded origami** → MuJoCo rigid body (single static shape)

### Paper Objects

| Object | Solver | Modeling Approach |
|--------|--------|-------------------|
| Playing card | MuJoCo | Thin box primitive (see Cards section) |
| Cardboard box | MuJoCo | 6 thin box panels, fixed together |
| Manila folder | MuJoCo | 2 thin boxes + revolute hinge joint |
| Open book | MuJoCo | 2 boxes + revolute joint |
| Closed book | MuJoCo | Single thick box |
| Printer paper (A4) | VBD cloth | Plane mesh, stiffness ~5000–10000 |
| Tissue paper | VBD cloth | Plane mesh, stiffness ~100–300 |
| Post-it note | VBD cloth | Small plane, mild adhesion |
| Origami crane | MuJoCo | Pre-folded rigid mesh |
| Paper airplane | MuJoCo | Rigid body + custom aero drag force |
| Gift wrapping | VBD cloth | Deformable sheet around rigid box |
| Cardboard tube | MuJoCo | Cylinder primitive |

### Gift Wrapping Pattern (from `falling_gift`)
```python
# VBD cloth (wrapping paper) + MuJoCo rigid (box inside)
builder = newton.ModelBuilder()

# Rigid gift box
box_body = builder.add_body(origin=wp.transform((0, 0.5, 0), wp.quat_identity()))
builder.add_shape_box(body=box_body, hx=0.1, hy=0.08, hz=0.15, density=500)

# Wrapping paper as cloth draped over box
paper_verts, paper_faces = load_obj("wrapping_paper.obj")
builder.add_cloth_mesh(
    pos=paper_verts + np.array([0, 0.6, 0]),  # start above box
    tri=paper_faces,
    stiffness=8000.0,  # paper is stiff
    damping=50.0,
)
```

---

## 5. SOFT BODIES (VBD Solver)
**Newton Examples:** `softbody_hanging`, `softbody_dropping_to_cloth`, `diffsim_soft_body`, `diffsim_bear`

### Meshes to Model / Collect

| Object | Stiffness | Volume Preservation | Notes |
|--------|-----------|--------------------|----|
| Rubber ball | 5000 | High | UV Sphere, 32 segments |
| Stress ball | 100 | Medium | Very squishy |
| Teddy bear | 500 | Medium | MUST be watertight |
| Pillow / cushion | 200 | Low | Rounded cube |
| Sponge | 300 | Low | Cube, subdivided |
| Rubber duck | 1000 | High | Sculpted, watertight |
| Yoga mat (rolled) | 3000 | High | Spiral cylinder |
| Foam block | 50–200 | Low | Simple cube |
| Jello / gelatin | 20–80 | Very high | Cube or mold shape |
| Eraser | 2000 | High | Small box, rounded edges |
| Stress toy (various shapes) | 100–500 | Medium | Any watertight shape |
| Rubber band ball | 800 | High | Sphere |
| Memory foam | 50–150 | Very low | Slow spring-back (high damping) |
| Marshmallow | 30–60 | Medium | Cylinder/sphere |

### CRITICAL: Soft Body Mesh Requirements
```
☐ MUST be watertight (closed manifold, zero holes)
☐ MUST be volumetric — Newton tetrahedralizes internally
☐ No non-manifold edges
☐ No flipped normals
☐ Blender: Mesh → Clean Up → Make Manifold
☐ Blender: 3D Print Toolbox add-on → check for issues
☐ Export as OBJ with normals
```

### Newton Code Pattern
```python
builder = newton.ModelBuilder()

vertices, faces = load_obj("teddy_bear.obj")

builder.add_soft_mesh(
    pos=vertices,
    tri=faces,
    vel=np.zeros_like(vertices),
    mass=0.5,
    stiffness=500.0,
    damping=10.0,
    volume_stiffness=1000.0,
)

model = builder.finalize()
solver = newton.SolverVBD()
```

### Soft Body Simulations

| Sim | Description | Extend From |
|-----|-------------|-------------|
| Stress ball squeeze | Franka gripper compresses soft sphere | `softbody_hanging` + `cloth_franka` |
| Bear drop on cloth | Soft-on-soft collision | `softbody_dropping_to_cloth` |
| Jello wobble | Shake the table, watch it jiggle | `diffsim_soft_body` |
| Pillow stacking | Stack soft objects, observe deformation | `softbody_hanging` |
| Sponge absorption | Compress sponge, watch it spring back | `diffsim_soft_body` |
| Marshmallow crush | Very soft body between rigid plates | `softbody_hanging` |
| Optimize bear shape | Differentiable sim for shape optimization | `diffsim_bear` |

---

## 6. GRANULAR MATERIALS (MPM Solver) — NO FLUIDS
**Newton Examples:** `mpm_granular`, `mpm_twoway_coupling`

**MPM for DRY granular ONLY.** Sand, rice, beans, gravel, powder, beads.

### What to Simulate

| Material | Particle Count | Radius (m) | Density (kg/m³) | Friction | Angle of Repose |
|----------|---------------|------------|-----------------|----------|-----------------|
| Sand (fine) | 10K–100K | 0.001 | 1600 | 0.6 | 30–35° |
| Sand (coarse) | 5K–50K | 0.003 | 1700 | 0.65 | 33–37° |
| Rice grains | 5K–50K | 0.002 | 750 | 0.4 | 25–30° |
| Kidney beans | 500–5K | 0.008 | 1300 | 0.3 | 20–25° |
| Coffee beans | 500–5K | 0.006 | 560 | 0.35 | 22–28° |
| Gravel / pebbles | 1K–10K | 0.01–0.02 | 2500 | 0.7 | 35–45° |
| Flour / powder | 100K+ | 0.0005 | 600 | 0.5 | 45–55° |
| Ball pit balls | 100–1K | 0.035 | 50 (hollow) | 0.3 | ~25° |
| Peppercorns | 1K–10K | 0.003 | 500 | 0.4 | 25–30° |
| Steel shot (BBs) | 1K–10K | 0.002 | 7800 | 0.1 | 20–25° |
| Corn kernels | 500–5K | 0.005 | 720 | 0.35 | 25–30° |

### Legos Are NOT Granular — They're Rigid Bodies
Legos interlock. Use MuJoCo with individual rigid body meshes per brick.
Only use MPM for a massive pile where individual brick shape doesn't matter.

### Generating Particle Positions in Blender
```python
# Run in Blender Python console
import bpy, numpy as np

# Select your fill-volume object (cube, cylinder, etc.)
obj = bpy.data.objects["FillVolume"]
bbox = obj.bound_box

n_particles = 10000
min_c = np.array([bbox[0][0], bbox[0][1], bbox[0][2]])
max_c = np.array([bbox[6][0], bbox[6][1], bbox[6][2]])

particles = np.random.uniform(min_c, max_c, size=(n_particles, 3))
particles *= 0.01  # cm → meters if needed

np.save("//particle_positions.npy", particles)
```

### Newton Code Pattern
```python
builder = newton.ModelBuilder()

# Container (rigid body — bowl, box, hopper)
container_verts, container_faces = load_obj("bowl.obj")
builder.add_shape_mesh(body=-1, mesh=newton.Mesh(container_verts, container_faces))

# Granular particles
positions = np.load("particle_positions.npy")
builder.add_particle_grid(
    pos=positions,
    vel=np.zeros_like(positions),
    mass=0.001,
    radius=0.002,    # per material
    jitter=0.0005,
)

model = builder.finalize()
solver = newton.SolverMPM()
```

### Granular Simulations

| Sim | Description | Extend From |
|-----|-------------|-------------|
| Sand pouring | Pour into bowl, observe angle of repose | `mpm_granular` |
| Hourglass | Sand through narrow opening | `mpm_granular` |
| Rice scoop | Franka scoops rice with spoon | `mpm_granular` + `cloth_franka` |
| Bean bag toss | Granular-filled soft body | `mpm_twoway_coupling` |
| Sandbox dig | Robot digs through sand | `mpm_anymal` (without the quadruped) |
| Funnel fill | Particles through funnel into container | `mpm_granular` |
| Tilt table | Pile on tilting surface, observe avalanche | `mpm_granular` |
| Vibrating sieve | Particles on vibrating mesh, size separation | `mpm_twoway_coupling` |

---

## 7. CHAINS, ROPES & ARTICULATED BODIES
**Solver:** MuJoCo Warp  
**Newton Examples:** `basic_pendulum`, `selection_articulations`

### Key Principle
Chains = articulated rigid body chain in URDF/MJCF.
**NOT** a single mesh. **NOT** VBD cloth. Each link = separate rigid body + joint.

See `three_section_staff.xml` (MJCF) and `three_section_staff.urdf` for working examples.

### Chain Objects

| Object | Links | Joint Type | Link Shape |
|--------|-------|-----------|------------|
| 3-section staff | 3 rods + 8 chain links | Ball joints | Capsule |
| Nunchaku | 2 rods + 3–5 chain links | Ball joints | Capsule |
| Metal chain | 20–50 links | Alternating revolute axes | Torus or capsule |
| Thick rope | 20–40 segments | Ball joints (3-DOF) | Capsule |
| Jump rope | 30+ segments, 2 kinematic ends | Ball joints | Capsule |
| Pendulum | 1–5 links | Revolute | Capsule or box |
| Newton's cradle | 5 pendulums side by side | Revolute (each) | Sphere + thin rod |
| Bead necklace | 20–50 spheres | Ball joints | Sphere |
| Dog leash | 15–25 segments | Ball joints | Capsule |
| Whip | 20–30 tapered segments | Ball joints | Capsule (decreasing radius) |

### Chain Stability Checklist
```
☐ Chain link mass ≥ 1/50th of connected rigid body mass
☐ Joint damping: 0.1–0.5 (prevents jitter)
☐ Timestep: ≤ 1/240 for chains (1/500 ideal)
☐ Use MJCF ball joints (3-DOF) not URDF revolute (1-DOF)
☐ Enable collision filtering between parent-child links
☐ Add armature (0.001) to prevent singularities
```

---

## MASTER SIMULATION MATRIX

| Newton Example | What to Replace | With Your Mesh | Solver |
|----------------|----------------|----------------|--------|
| `cloth_franka` | Dummy cloth plane | Your garment OBJ | VBD + MuJoCo |
| `softbody_dropping_to_cloth` | Generic soft body | Teddy bear, pillow, etc. | VBD |
| `softbody_hanging` | Generic deformable | Any watertight soft mesh | VBD |
| `diffsim_soft_body` | Parameter target | Your soft object | VBD (differentiable) |
| `diffsim_bear` | Bear mesh | Any soft shape | VBD (differentiable) |
| `diffsim_cloth` | Cloth params | Your cloth with real stiffness | VBD (differentiable) |
| `poker_cards_stacking` | Cards | Your card/domino/coin meshes | MuJoCo |
| `falling_gift` | Gift wrap | Your paper mesh + rigid box | VBD + MuJoCo |
| `mpm_granular` | Generic particles | Your grain type (rice, sand, beans) | MPM |
| `mpm_twoway_coupling` | Generic coupling | Granular + rigid container | MPM + MuJoCo |
| `basic_pendulum` | Single pendulum | Chain, rope, staff | MuJoCo |
| `selection_articulations` | Generic articulated | Your URDF chain/staff | MuJoCo |
| `ik_cube_stacking` | Cubes | Dice, blocks, Jenga pieces | MuJoCo |
| `selection_materials` | Material comparison | Ball types (pool, marble, steel) | MuJoCo |
| **NEW: Pool table** | N/A | Custom MJCF (provided) | MuJoCo |

---

## POOL TABLE — Complete Setup

See `pool_table.xml` for the full MJCF definition.

### Quick Start
```python
import newton

builder = newton.ModelBuilder()
builder.add_mjcf("pool_table.xml")
model = builder.finalize()
solver = newton.SolverMuJoCo()
state = model.state()

# Apply cue strike: set initial velocity on cue ball
# Ball index 0 = cue ball
cue_ball_velocity = np.array([0, 0, 3.0])  # 3 m/s forward
state.body_qd[0, :3] = cue_ball_velocity

for frame in range(1000):
    solver.step(model, state, dt=1.0/240.0)
```

### What Makes Pool Sim Tricky
1. **Rolling friction** — balls roll, not slide. Need rolling resistance model.
2. **Spin transfer** — English (sidespin) affects collision angles.
3. **Cushion physics** — rubber deforms, angle isn't perfect mirror reflection.
4. **Pocket geometry** — balls can rattle in/out of pockets.
5. **Cue ball control** — draw, follow, masse all depend on spin state.

MuJoCo handles (1) and (2) natively through its contact model.
(3) can be approximated with cushion restitution + angled geometry.
(4) and (5) require careful geometry setup (provided in pool_table.xml).
