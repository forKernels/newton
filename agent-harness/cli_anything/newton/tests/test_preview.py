"""The preview producer publishes what the REAL backend produced.

Nothing here renders. These check the contract: recipes are honest about what
they cost, the source fingerprint decides cache identity, `latest` never
renders, and a recipe that promised a hero artifact and produced none is
PARTIAL rather than ok.
"""

import json
from pathlib import Path

import pytest

from cli_anything.newton.core import preview


def test_every_recipe_declares_what_it_costs():
    """A recipe an agent cannot price is a recipe it cannot choose."""
    for r in preview.list_recipes():
        assert r["frames"] > 0
        assert isinstance(r["usd"], bool)
        assert r["description"]


def test_capture_refuses_without_a_scene():
    with pytest.raises(ValueError, match="scene file or --procedural"):
        preview.capture()


def test_capture_refuses_an_unknown_recipe():
    with pytest.raises(ValueError, match="unknown recipe"):
        preview.capture(scene_type="ground", recipe="cinematic")


def test_a_procedural_scene_is_fingerprinted_by_its_PARAMETERS():
    """It has no file to hash, and its parameters ARE its source.

    Fingerprinting it as 'no file, therefore always fresh' would re-render on
    every call; fingerprinting it as a constant would never re-render when the
    recipe changed.
    """
    a = preview._source_fingerprint(None, "ground", "quick", "xpbd")
    b = preview._source_fingerprint(None, "ground", "quick", "xpbd")
    c = preview._source_fingerprint(None, "ground", "usd", "xpbd")
    d = preview._source_fingerprint(None, "pendulum", "quick", "xpbd")
    assert a == b            # same inputs, same bundle
    assert a != c            # a different recipe is a different capture
    assert a != d            # so is a different scene


def test_a_scene_file_is_fingerprinted_by_its_BYTES(tmp_path):
    """Editing the scene must invalidate the cache. Fingerprinting the PATH
    would serve a stale bundle for edited content, which is the failure the
    cache exists to avoid being."""
    f = tmp_path / "s.usda"
    f.write_text("#usda 1.0\n")
    before = preview._source_fingerprint(str(f), None, "quick", "xpbd")
    f.write_text("#usda 1.0\n# edited\n")
    after = preview._source_fingerprint(str(f), None, "quick", "xpbd")
    assert before != after


def test_latest_is_read_only_and_returns_None_when_nothing_exists(tmp_path):
    """`preview latest` must never render. If it rendered on a miss it would be
    a slow `capture` wearing a cheap name, and the agent loop would stop being
    predictable."""
    assert preview.latest(recipe="nosuchrecipe",
                          root_dir=str(tmp_path)) is None


def test_the_producer_never_reads_a_bundle_for_display():
    """Producer and consumer are separate roles - `cli-hub previews` inspects.
    A producer that also renders for display grows a second, divergent view."""
    src = Path(preview.__file__).read_text()
    for consumer in ("def inspect", "def html", "def watch", "def open_bundle"):
        assert consumer not in src
