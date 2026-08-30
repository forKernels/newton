"""
Unified Fabric Library
========================
Single source of truth for cloth physics properties across all engines:
  - Blender (OpenCue)
  - Newton / Warp (GPU)
  - Marvelous Designer (future)

No engine-specific imports (no bpy, no warp).
Both ``clean_cloth_dataset.py`` (Blender prep) and ``newton_sim_utils.py``
(Newton runtime) import from here.

Adding a new engine
--------------------
1. Add a sub-dict keyed by the engine name to each fabric entry.
2. The prep script (``clean_cloth_dataset.py``) will write it into
   ``cloth_properties.json`` automatically.
3. Your sim script reads the relevant section at runtime.
"""

from __future__ import annotations

# =============================================================================
# FABRIC LIBRARY -- Physical properties for simulation
# =============================================================================
# Sources: textile engineering references, typical values for cloth simulation.
#
# Each fabric entry contains:
#
#   SHARED (engine-agnostic)
#   ─────────────────────────
#   description      human-readable label
#   weight_class     ultralight | light | medium | heavy
#   density          areal density  [kg/m²]
#   friction         surface friction coefficient
#
#   BLENDER (cloth modifier)
#   ────────────────────────
#   tri_ke           stretch stiffness
#   tri_ka           area preservation stiffness
#   tri_kd           damping coefficient
#
#   NEWTON (Warp cloth solver)
#   ──────────────────────────
#   newton.tri_ke / tri_ka / tri_kd   (can differ from Blender values)
#   newton.edge_ke                    bending stiffness
#   newton.edge_kd                    bending damping
#   newton.particle_radius            collision radius per particle [m]
#   newton.self_contact_radius        self-collision radius [m]
#   newton.contact_mu                 contact friction override
#
#   MARVELOUS DESIGNER  (placeholder — values TBD from MD export)
#   ──────────────────────────────────────────────────────────────
#   md.preset_name        name of the built-in MD preset (if any)
#   md.stretch_weft       weft stretch resistance
#   md.stretch_warp       warp stretch resistance
#   md.shear              shear resistance
#   md.bending_weft       weft bending resistance
#   md.bending_warp       warp bending resistance
#   md.buckling_ratio     buckling ratio
#   md.density_gsm        g/m² (MD uses grams, we store as-is)
#
# To add properties from a new engine, add a new sub-dict keyed by the
# engine name.  The prep script writes all of them into
# cloth_properties.json so any consumer can pick the keys it needs.
# =============================================================================

FABRICS: dict[str, dict] = {
    # --- Ultralight -----------------------------------------------------------
    "silk": {
        "description": "Silk -- very light, smooth, fluid drape",
        "density": 0.08,
        "friction": 0.20,
        "weight_class": "ultralight",
        "tri_ke": 30.0,
        "tri_ka": 25.0,
        "tri_kd": 0.5e-6,
        "newton": {
            "tri_ke": 5e1,
            "tri_ka": 5e1,
            "tri_kd": 1e-6,
            "edge_ke": 5e-5,
            "edge_kd": 5e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.4,
        },
        "marvelous_designer": {
            "preset_name": "Silk (Charmeuse)",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 80,
        },
    },
    "chiffon": {
        "description": "Chiffon -- sheer, floaty, delicate",
        "density": 0.06,
        "friction": 0.25,
        "weight_class": "ultralight",
        "tri_ke": 20.0,
        "tri_ka": 15.0,
        "tri_kd": 0.3e-6,
        "newton": {
            "tri_ke": 3e1,
            "tri_ka": 3e1,
            "tri_kd": 5e-7,
            "edge_ke": 3e-5,
            "edge_kd": 3e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.4,
        },
        "marvelous_designer": {
            "preset_name": "Chiffon",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 60,
        },
    },
    "organza": {
        "description": "Organza -- crisp, sheer, holds shape",
        "density": 0.07,
        "friction": 0.22,
        "weight_class": "ultralight",
        "tri_ke": 45.0,
        "tri_ka": 40.0,
        "tri_kd": 0.4e-6,
        "newton": {
            "tri_ke": 6e1,
            "tri_ka": 6e1,
            "tri_kd": 8e-7,
            "edge_ke": 1e-4,
            "edge_kd": 5e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.4,
        },
        "marvelous_designer": {
            "preset_name": "Organza",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 70,
        },
    },
    "satin": {
        "description": "Satin -- smooth, glossy, medium drape",
        "density": 0.12,
        "friction": 0.15,
        "weight_class": "light",
        "tri_ke": 40.0,
        "tri_ka": 35.0,
        "tri_kd": 0.6e-6,
        "newton": {
            "tri_ke": 6e1,
            "tri_ka": 6e1,
            "tri_kd": 1e-6,
            "edge_ke": 8e-5,
            "edge_kd": 8e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.3,
        },
        "marvelous_designer": {
            "preset_name": "Satin",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 120,
        },
    },
    # --- Light ----------------------------------------------------------------
    "cotton": {
        "description": "Cotton broadcloth -- everyday woven, medium body",
        "density": 0.15,
        "friction": 0.45,
        "weight_class": "light",
        "tri_ke": 80.0,
        "tri_ka": 70.0,
        "tri_kd": 1.2e-6,
        "newton": {
            "tri_ke": 1e2,
            "tri_ka": 1e2,
            "tri_kd": 1.5e-6,
            "edge_ke": 1e-4,
            "edge_kd": 1e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": "Cotton",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 150,
        },
    },
    "linen": {
        "description": "Linen -- natural fiber, crisp, wrinkles easily",
        "density": 0.17,
        "friction": 0.40,
        "weight_class": "light",
        "tri_ke": 100.0,
        "tri_ka": 90.0,
        "tri_kd": 1.5e-6,
        "newton": {
            "tri_ke": 1.2e2,
            "tri_ka": 1.2e2,
            "tri_kd": 2e-6,
            "edge_ke": 2e-4,
            "edge_kd": 1e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.5,
        },
        "marvelous_designer": {
            "preset_name": "Linen",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 170,
        },
    },
    "jersey": {
        "description": "Jersey knit -- stretchy, soft, t-shirt fabric",
        "density": 0.18,
        "friction": 0.50,
        "weight_class": "light",
        "tri_ke": 35.0,
        "tri_ka": 30.0,
        "tri_kd": 1.0e-6,
        "newton": {
            "tri_ke": 8e1,
            "tri_ka": 8e1,
            "tri_kd": 1e-6,
            "edge_ke": 8e-5,
            "edge_kd": 8e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": "Cotton Jersey",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 180,
        },
    },
    "polyester": {
        "description": "Polyester -- wrinkle-resistant, slightly slippery",
        "density": 0.14,
        "friction": 0.30,
        "weight_class": "light",
        "tri_ke": 60.0,
        "tri_ka": 55.0,
        "tri_kd": 0.8e-6,
        "newton": {
            "tri_ke": 9e1,
            "tri_ka": 9e1,
            "tri_kd": 1.2e-6,
            "edge_ke": 1e-4,
            "edge_kd": 8e-4,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.5,
        },
        "marvelous_designer": {
            "preset_name": "Polyester",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 140,
        },
    },
    # --- Medium ---------------------------------------------------------------
    "cotton_twill": {
        "description": "Cotton twill -- chinos, structured but not stiff",
        "density": 0.25,
        "friction": 0.40,
        "weight_class": "medium",
        "tri_ke": 120.0,
        "tri_ka": 110.0,
        "tri_kd": 2.0e-6,
        "newton": {
            "tri_ke": 1.5e2,
            "tri_ka": 1.5e2,
            "tri_kd": 2.5e-6,
            "edge_ke": 3e-4,
            "edge_kd": 2e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": "Cotton (Heavy)",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 250,
        },
    },
    "wool": {
        "description": "Wool -- warm, medium drape, textured surface",
        "density": 0.28,
        "friction": 0.55,
        "weight_class": "medium",
        "tri_ke": 100.0,
        "tri_ka": 90.0,
        "tri_kd": 2.5e-6,
        "newton": {
            "tri_ke": 1.5e2,
            "tri_ka": 1.5e2,
            "tri_kd": 3e-6,
            "edge_ke": 3e-4,
            "edge_kd": 2e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.7,
        },
        "marvelous_designer": {
            "preset_name": "Wool",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 280,
        },
    },
    "flannel": {
        "description": "Flannel -- soft, brushed, warm",
        "density": 0.24,
        "friction": 0.60,
        "weight_class": "medium",
        "tri_ke": 90.0,
        "tri_ka": 80.0,
        "tri_kd": 2.0e-6,
        "newton": {
            "tri_ke": 1.2e2,
            "tri_ka": 1.2e2,
            "tri_kd": 2.5e-6,
            "edge_ke": 2e-4,
            "edge_kd": 1.5e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.7,
        },
        "marvelous_designer": {
            "preset_name": "Flannel",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 240,
        },
    },
    "velvet": {
        "description": "Velvet -- plush, heavy drape, grippy surface",
        "density": 0.30,
        "friction": 0.65,
        "weight_class": "medium",
        "tri_ke": 85.0,
        "tri_ka": 75.0,
        "tri_kd": 2.5e-6,
        "newton": {
            "tri_ke": 1.2e2,
            "tri_ka": 1.2e2,
            "tri_kd": 3e-6,
            "edge_ke": 2e-4,
            "edge_kd": 1.5e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.8,
        },
        "marvelous_designer": {
            "preset_name": "Velvet",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 300,
        },
    },
    "corduroy": {
        "description": "Corduroy -- ribbed texture, medium-heavy",
        "density": 0.32,
        "friction": 0.55,
        "weight_class": "medium",
        "tri_ke": 130.0,
        "tri_ka": 120.0,
        "tri_kd": 2.5e-6,
        "newton": {
            "tri_ke": 1.8e2,
            "tri_ka": 1.8e2,
            "tri_kd": 3e-6,
            "edge_ke": 4e-4,
            "edge_kd": 2e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.7,
        },
        "marvelous_designer": {
            "preset_name": "Corduroy",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 320,
        },
    },
    # --- Heavy ----------------------------------------------------------------
    "denim": {
        "description": "Denim -- heavy, stiff, jeans fabric",
        "density": 0.40,
        "friction": 0.45,
        "weight_class": "heavy",
        "tri_ke": 250.0,
        "tri_ka": 230.0,
        "tri_kd": 4.0e-6,
        "newton": {
            "tri_ke": 2e2,
            "tri_ka": 2e2,
            "tri_kd": 4e-6,
            "edge_ke": 5e-4,
            "edge_kd": 3e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": "Denim",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 400,
        },
    },
    "canvas": {
        "description": "Canvas -- heavy-duty, very stiff, work wear",
        "density": 0.45,
        "friction": 0.50,
        "weight_class": "heavy",
        "tri_ke": 300.0,
        "tri_ka": 280.0,
        "tri_kd": 5.0e-6,
        "newton": {
            "tri_ke": 3e2,
            "tri_ka": 3e2,
            "tri_kd": 5e-6,
            "edge_ke": 8e-4,
            "edge_kd": 5e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": None,
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 450,
        },
    },
    "leather": {
        "description": "Leather -- heavy, stiff, minimal stretch",
        "density": 0.60,
        "friction": 0.35,
        "weight_class": "heavy",
        "tri_ke": 400.0,
        "tri_ka": 380.0,
        "tri_kd": 6.0e-6,
        "newton": {
            "tri_ke": 4e2,
            "tri_ka": 4e2,
            "tri_kd": 6e-6,
            "edge_ke": 1e-3,
            "edge_kd": 5e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.5,
        },
        "marvelous_designer": {
            "preset_name": "Leather",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 600,
        },
    },
    "wool_coat": {
        "description": "Wool coating -- thick, structured, overcoats",
        "density": 0.50,
        "friction": 0.50,
        "weight_class": "heavy",
        "tri_ke": 280.0,
        "tri_ka": 260.0,
        "tri_kd": 5.0e-6,
        "newton": {
            "tri_ke": 3e2,
            "tri_ka": 3e2,
            "tri_kd": 5e-6,
            "edge_ke": 8e-4,
            "edge_kd": 5e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.6,
        },
        "marvelous_designer": {
            "preset_name": "Wool (Heavy)",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 500,
        },
    },
    "fleece": {
        "description": "Fleece -- soft, thick, stretchy, high friction",
        "density": 0.35,
        "friction": 0.70,
        "weight_class": "heavy",
        "tri_ke": 60.0,
        "tri_ka": 50.0,
        "tri_kd": 3.0e-6,
        "newton": {
            "tri_ke": 1.5e2,
            "tri_ka": 1.5e2,
            "tri_kd": 3e-6,
            "edge_ke": 2e-4,
            "edge_kd": 2e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.8,
        },
        "marvelous_designer": {
            "preset_name": "Fleece",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 350,
        },
    },
    "neoprene": {
        "description": "Neoprene -- thick, rubbery, wetsuit material",
        "density": 0.55,
        "friction": 0.60,
        "weight_class": "heavy",
        "tri_ke": 150.0,
        "tri_ka": 140.0,
        "tri_kd": 5.0e-6,
        "newton": {
            "tri_ke": 3e2,
            "tri_ka": 3e2,
            "tri_kd": 5e-6,
            "edge_ke": 5e-4,
            "edge_kd": 3e-3,
            "particle_radius": 0.008,
            "self_contact_radius": 0.002,
            "contact_mu": 0.7,
        },
        "marvelous_designer": {
            "preset_name": "Neoprene",
            "stretch_weft": None,
            "stretch_warp": None,
            "shear": None,
            "bending_weft": None,
            "bending_warp": None,
            "buckling_ratio": None,
            "density_gsm": 550,
        },
    },
}


# =============================================================================
# GARMENT -> COMPATIBLE FABRICS mapping
# =============================================================================

CATEGORY_FABRICS: dict[str, list[str]] = {
    "dress_sleeveless": [
        "silk",
        "chiffon",
        "satin",
        "cotton",
        "linen",
        "jersey",
        "polyester",
    ],
    "dress": [
        "silk",
        "chiffon",
        "satin",
        "cotton",
        "linen",
        "jersey",
        "polyester",
        "wool",
        "velvet",
    ],
    "tee_sleeveless": ["jersey", "cotton", "polyester", "silk"],
    "tee": ["jersey", "cotton", "polyester", "flannel"],
    "jacket": [
        "denim",
        "canvas",
        "leather",
        "wool_coat",
        "cotton_twill",
        "corduroy",
        "neoprene",
    ],
    "jacket_hood": ["fleece", "denim", "canvas", "neoprene", "wool_coat"],
    "pants_straight_sides": [
        "denim",
        "cotton_twill",
        "wool",
        "corduroy",
        "linen",
        "polyester",
        "leather",
        "flannel",
    ],
    "pants": [
        "denim",
        "cotton_twill",
        "wool",
        "corduroy",
        "linen",
        "polyester",
        "leather",
    ],
    "skirt_2_panels": [
        "silk",
        "cotton",
        "linen",
        "polyester",
        "wool",
        "satin",
        "velvet",
    ],
    "skirt_4_panels": [
        "silk",
        "cotton",
        "linen",
        "polyester",
        "wool",
        "satin",
        "velvet",
        "chiffon",
    ],
    "skirt_8_panels": [
        "silk",
        "cotton",
        "linen",
        "polyester",
        "wool",
        "chiffon",
        "organza",
    ],
    "skirt": ["silk", "cotton", "linen", "polyester", "wool", "satin"],
    "jumpsuit_sleeveless": ["jersey", "cotton", "linen", "polyester", "denim"],
    "wb_dress_sleeveless": [
        "silk",
        "chiffon",
        "satin",
        "cotton",
        "linen",
        "jersey",
        "polyester",
    ],
    "wb_pants_straight": [
        "denim",
        "cotton_twill",
        "wool",
        "corduroy",
        "linen",
        "polyester",
        "leather",
        "flannel",
    ],
}

DEFAULT_FABRICS: list[str] = ["cotton", "polyester", "jersey", "wool", "denim"]

SOLIDIFY_THICKNESS: float = 0.005
CLOTH_SCALE: float = 0.01  # cm -> m


# =============================================================================
# Helper functions
# =============================================================================


def get_fabrics_for_category(category_name: str) -> list[str]:
    """Get compatible fabric names for a garment category.

    Strips trailing ``_NNNN`` vertex counts, then tries longest-prefix matching.
    """
    parts = category_name.rsplit("_", 1)
    base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else category_name

    if base_name in CATEGORY_FABRICS:
        return CATEGORY_FABRICS[base_name]

    for key in sorted(CATEGORY_FABRICS.keys(), key=len, reverse=True):
        if base_name.startswith(key):
            return CATEGORY_FABRICS[key]

    return DEFAULT_FABRICS


def get_fabric_properties(fabric_name: str) -> dict:
    """Get full simulation properties for a named fabric."""
    if fabric_name not in FABRICS:
        raise ValueError(f"Unknown fabric: {fabric_name}. Available: {list(FABRICS.keys())}")
    return FABRICS[fabric_name].copy()


def get_newton_preset(fabric_name: str) -> dict:
    """Extract Newton-specific cloth params for a fabric.

    Returns a flat dict with keys: density, tri_ke, tri_ka, tri_kd,
    edge_ke, edge_kd, particle_radius, self_contact_radius, contact_mu.
    """
    props = get_fabric_properties(fabric_name)
    n = props.get("newton", {})
    return {
        "density": n.get("density", props.get("density", 0.15)),
        "tri_ke": n.get("tri_ke", props.get("tri_ke", 1e3)),
        "tri_ka": n.get("tri_ka", props.get("tri_ka", 1e3)),
        "tri_kd": n.get("tri_kd", props.get("tri_kd", 1e-1)),
        "edge_ke": n.get("edge_ke", 0.01),
        "edge_kd": n.get("edge_kd", 0.01),
        "particle_radius": n.get("particle_radius", 0.005),
        "self_contact_radius": n.get("self_contact_radius", 0.01),
        "contact_mu": n.get("contact_mu", 0.6),
    }


def get_blender_preset(fabric_name: str) -> dict:
    """Extract Blender-specific cloth params for a fabric.

    Returns a flat dict with keys: density, tri_ke, tri_ka, tri_kd, friction.
    """
    props = get_fabric_properties(fabric_name)
    return {
        "density": props["density"],
        "tri_ke": props["tri_ke"],
        "tri_ka": props["tri_ka"],
        "tri_kd": props["tri_kd"],
        "friction": props["friction"],
    }


def build_cloth_properties_json(
    fabric_name: str,
    category: str,
    garment_id: str,
    original_units: float = 0.01,
    mesh_stats: dict | None = None,
) -> dict:
    """Build the full unified cloth_properties.json content.

    Used by the prep script to write per-garment metadata.
    """
    fabric_props = get_fabric_properties(fabric_name)
    newton_props = fabric_props.get("newton", {})
    md_props = fabric_props.get("marvelous_designer", {})

    result = {
        "category": category,
        "garment_id": garment_id,
        "fabric": fabric_name,
        "fabric_description": fabric_props["description"],
        "weight_class": fabric_props["weight_class"],
        "original_units_in_meter": original_units,
        "units": "meters",
        "solidify_thickness": SOLIDIFY_THICKNESS,
        # Shared
        "density": fabric_props["density"],
        "friction": fabric_props["friction"],
        # Blender cloth modifier
        "blender": {
            "tri_ke": fabric_props["tri_ke"],
            "tri_ka": fabric_props["tri_ka"],
            "tri_kd": fabric_props["tri_kd"],
            "density": fabric_props["density"],
            "friction": fabric_props["friction"],
        },
        # Newton (Warp) solver
        "newton": {
            "density": newton_props.get("density", fabric_props["density"]),
            "tri_ke": newton_props.get("tri_ke", fabric_props["tri_ke"]),
            "tri_ka": newton_props.get("tri_ka", fabric_props["tri_ka"]),
            "tri_kd": newton_props.get("tri_kd", fabric_props["tri_kd"]),
            "edge_ke": newton_props.get("edge_ke", 0.01),
            "edge_kd": newton_props.get("edge_kd", 0.01),
            "particle_radius": newton_props.get("particle_radius", 0.005),
            "self_contact_radius": newton_props.get("self_contact_radius", 0.01),
            "contact_mu": newton_props.get("contact_mu", 0.6),
        },
        # Marvelous Designer (placeholder — fill from MD export)
        "marvelous_designer": {
            "preset_name": md_props.get("preset_name"),
            "density_gsm": md_props.get("density_gsm"),
            "stretch_weft": md_props.get("stretch_weft"),
            "stretch_warp": md_props.get("stretch_warp"),
            "shear": md_props.get("shear"),
            "bending_weft": md_props.get("bending_weft"),
            "bending_warp": md_props.get("bending_warp"),
            "buckling_ratio": md_props.get("buckling_ratio"),
        },
    }

    if mesh_stats:
        result["extent_m"] = mesh_stats.get("extent_m")

    return result
