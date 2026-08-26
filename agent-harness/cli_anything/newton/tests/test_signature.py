"""Keyword introspection and contact-buffer sizing. No newton needed.

Both were ported from the newton-lab harness in the cli-anything repo, and
both exist because of a specific bug that cost real time:

  * `density=` moved onto ShapeConfig and was DROPPED from every add_shape_*
    call for a whole release. Nothing raised. Every rigid body silently used
    newton's default density.
  * A collision pipeline built without an explicit rigid_contact_max gets
    newton's per-model estimate, which is small. Overflow WARNS and drops
    contacts rather than raising, so the symptom is a body passing through
    something it should have hit.

newton is stubbed here so these run anywhere; whether the answers match the
real engine is what `cli-anything-newton signature` is for.
"""

import inspect
import types

import pytest

from cli_anything.newton.core import signature as sig
from cli_anything.newton.core.simulation import (
    CONTACT_MAX_FLOOR, contact_capacity, make_pipeline)


class FakeShapeConfig:
    def __init__(self, density=1000.0, mu=0.5, restitution=0.0):
        pass


def add_shape_box(body, xform=None, hx=0.5, hy=0.5, hz=0.5, cfg=None):
    """A shape call that does NOT take density - the whole point."""


def takes_anything(model, **kwargs):
    pass


def fake_newton():
    n = types.ModuleType("newton")
    n.__version__ = "1.6.0.dev0"
    builder = types.SimpleNamespace(
        add_shape_box=add_shape_box, ShapeConfig=FakeShapeConfig)
    n.ModelBuilder = builder
    n.solvers = types.SimpleNamespace(SolverAnything=takes_anything)
    return n


def test_a_moved_keyword_is_reported_as_dropped():
    """The density bug, in one assertion. It is not on the shape call and is
    on ShapeConfig, and the answer has to distinguish them."""
    box, cfg = sig.signatures(
        fake_newton(),
        ["ModelBuilder.add_shape_box", "ModelBuilder.ShapeConfig"],
        accepts=["density"])
    assert box["would_be_dropped"] == ["density"]
    assert cfg["accepts"]["density"] is True
    assert cfg["would_be_dropped"] == []


def test_a_class_is_described_by_its_init():
    """A caller of SolverVBD(model, ...) needs __init__'s parameters, not the
    class's, and the result must SAY which it gave."""
    cfg, = sig.signatures(fake_newton(), ["ModelBuilder.ShapeConfig"])
    assert cfg["kind"] == "class (__init__)"
    assert "self" not in cfg["keywords"]
    assert "density" in cfg["keywords"]


def test_var_keyword_is_flagged_because_it_validates_nothing():
    """Anything taking **kwargs swallows every keyword, so nothing is dropped
    and nothing is checked either. A silent success there means less."""
    e, = sig.signatures(fake_newton(), ["solvers.SolverAnything"],
                        accepts=["nonsense"])
    assert e["accepts_var_keyword"] is True
    assert e["accepts"]["nonsense"] is True
    assert e["would_be_dropped"] == []


def test_required_keywords_are_separated():
    """A required one cannot be rescued by dropping it - add_cloth_mesh(vel=)
    has no default, so omitting it raises rather than degrading."""
    e, = sig.signatures(fake_newton(), ["ModelBuilder.add_shape_box"])
    assert e["required"] == ["body"]


def test_an_unknown_path_reports_what_it_found():
    e, = sig.signatures(fake_newton(), ["ModelBuilder.add_shape_nonsense"])
    assert e["resolved"] is False
    assert "has no attribute" in e["error"]


def test_a_wrong_path_suggests_near_names():
    e, = sig.signatures(fake_newton(), ["ModelBuilder.add_shape_bo"])
    assert "add_shape_box" in e["error"]


def test_the_well_known_list_is_only_a_shortcut():
    """Any dotted path resolves; the list exists so a caller need not know
    newton's module layout to ask a question."""
    assert "ModelBuilder.ShapeConfig" in sig.WELL_KNOWN
    e, = sig.signatures(fake_newton(), ["solvers.SolverAnything"])
    assert e["resolved"] is True


# --- contact buffer -------------------------------------------------------

class FakeModel:
    def __init__(self, bodies):
        self.body_count = bodies
        self.rigid_contact_max = None


def test_a_small_scene_still_gets_the_floor():
    """newton's own estimate for one box on a ground plane is 1000. The floor
    is what stops a harness measuring a buffer no consumer runs."""
    assert contact_capacity(FakeModel(1)) == CONTACT_MAX_FLOOR
    assert CONTACT_MAX_FLOOR >= 16384


def test_capacity_scales_with_bodies():
    assert contact_capacity(FakeModel(200)) == 200 * 512


def test_the_model_is_told_too_not_just_the_pipeline():
    """Both halves matter: the pipeline allocates the buffer, the model field
    is what a solver reads when sizing its own."""
    seen = {}

    def CollisionPipeline(model, rigid_contact_max=None, **kw):
        seen["max"] = rigid_contact_max
        return "pipeline"

    n = types.ModuleType("newton")
    n.CollisionPipeline = CollisionPipeline
    model = FakeModel(48)
    assert make_pipeline(n, model) == "pipeline"
    assert seen["max"] == contact_capacity(model)
    assert model.rigid_contact_max == contact_capacity(model)


def test_a_build_without_the_keyword_still_gets_the_model_field():
    """An older newton that does not take rigid_contact_max must not crash the
    harness - it degrades to setting the model field alone."""
    def CollisionPipeline(model, **kw):
        if "rigid_contact_max" in kw:
            raise TypeError("unexpected keyword")
        return "old-pipeline"

    n = types.ModuleType("newton")
    n.CollisionPipeline = CollisionPipeline
    model = FakeModel(4)
    assert make_pipeline(n, model) == "old-pipeline"
    assert model.rigid_contact_max == CONTACT_MAX_FLOOR
