"""The session must be reachable, not just implemented.

`Session` tracked `modified`, set it on load and step, and displayed it at the
prompt from the first commit. It also had `save()` and a file-locking JSON
writer. None of it could be invoked: the REPL constructed `Session()` with no
path, offered no save command, and the CLI had no --session flag. A user could
load a scene, step a hundred frames, watch the prompt say modified, quit, and
lose all of it in silence.

These tests fail if any strand of that path is cut again.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli_anything.newton.core.session import Session
from cli_anything.newton.newton_cli import cli


def test_session_round_trips_through_a_file():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "s.json")
        s = Session(path)
        s.solver_type, s.frame, s.modified = "vbd", 42, True
        assert s.save() == path
        got = json.loads(Path(path).read_text())
        assert got["solver_type"] == "vbd" and got["frame"] == 42
        # A save that does not clear `modified` makes the quit warning lie.
        assert s.modified is False


def test_a_session_with_no_path_refuses_rather_than_writing_somewhere():
    with pytest.raises(ValueError, match="No save path"):
        Session().save()


def test_cli_exposes_session_and_dry_run():
    """The flags are the ONLY way the persistence path can be reached."""
    out = CliRunner().invoke(cli, ["--help"]).output
    assert "--session" in out
    assert "--dry-run" in out


def test_repl_advertises_save():
    """A save command nobody is told about is a save command nobody uses."""
    src = Path(__file__).resolve().parents[1] / "newton_cli.py"
    text = src.read_text()
    assert '"save":' in text, "save missing from the REPL command table"
    assert 'elif cmd == "save"' in text, "save advertised but not handled"


def test_quit_warns_when_modified_and_unsaved():
    src = (Path(__file__).resolve().parents[1] / "newton_cli.py").read_text()
    quit_block = src.split('if cmd in ("quit", "exit", "q"):')[1][:600]
    assert "session.modified" in quit_block, "quit discards changes silently"
