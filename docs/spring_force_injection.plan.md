## Context

Newton's VBD solver has a mesh cloth pinning bug where ~10% of particles on imported garments produce zero displacement. We identified that injecting mass-spring forces as external forces into `state.particle_f` (between `clear_forces()` and `solver.step()`) can provide a "gravity assist" for stuck particles without replacing the VBD solver.

The DexCloth-cleanup project has proven mass-spring force computation code that we'll port to Warp GPU kernels.

---

## Phase 1: Create `SpringForceAssist` class

**File:** `c:\Users\davidclabaugh\Documents\GitHub\newton-Zhaocorp-cloth-dataset-pickup\scripts\disco\spring_force_assist.py` (new)

Create a self-contained class that:

1. **Builds a spring network from mesh topology** (constructor)
   - Extract 1-ring edges from `model.tri_indices` → stretching springs (ks=500)
   - Compute 2-ring neighbors via BFS → bending springs (kb=25)
   - Store as flat arrays: `spring_indices` (wp.vec2i), `rest_lengths` (float), `stiffness` (float), `damping` (float)
   - All arrays on GPU as Warp arrays

2. **Warp kernel: `compute_spring_forces`** — runs per-spring on GPU
   - For each spring: `diff = pos[i] - pos[j]`, `dist = length(diff)`, `F = k * (1 - rest_len/dist) * diff`
   - `wp.atomic_add(forces, i, -F)` and `wp.atomic_add(forces, j, F)` (Newton's 3rd law)
   - Optional damping: `F_damp = -kd * dot(v_rel, dir) * dir`

3. **Stuck particle detection** — track displacement per frame, maintain `was_stuck` mask

4. **`apply()` method** — called between `clear_forces()` and `solver.step()`
   - Computes spring forces
   - Writes to `state.particle_f` for stuck particles only (or all particles with low weight)
   - Applies a configurable `boost_factor`

Key references:
- DexCloth force: `F = k * (rest_len/dist - 1) * diff`, accumulated via `wp.atomic_add/sub`
- Newton injection: write to `state.particle_f` (wp.array dtype=wp.vec3, units=Newtons)
- `forward_step` kernel uses: `accel = external_force * inv_mass`

## Phase 2: Integrate into simulation loop

**File:** `c:\Users\davidclabaugh\Documents\GitHub\newton-Zhaocorp-cloth-dataset-pickup\scripts\disco\newton_sim_utils.py` (modify)

- Add a `create_spring_assist()` helper that builds `SpringForceAssist` from a finalized model
- No changes to `build_solver` — the spring assist is a separate wrapper

## Phase 3: Update test script

**File:** `c:\Users\davidclabaugh\Documents\GitHub\newton-Zhaocorp-cloth-dataset-pickup\scripts\disco\test_cloth_self_collision.py` (modify)

- Add `--spring-assist` flag (default True for mesh cloth)
- Insert `spring_assist.apply(state_0)` between `clear_forces()` and `solver.step()` in the sim loop
- Log stuck particle count with and without spring assist

## Verification

1. Run `test_cloth_self_collision.py --garment dress --viewer gl` — the dress should drape fully onto the cone with zero stuck particles
2. Compare stuck count: before (891) vs after (target: <50)
3. Run grid cloth to verify spring assist doesn't break the already-working case
4. Performance check: spring force compute should add <20% overhead