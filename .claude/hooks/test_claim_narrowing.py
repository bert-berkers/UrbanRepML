#!/usr/bin/env python3
"""Unit tests for check_claim_narrowing() in subagent-stop.py.

Run: python -m pytest .claude/hooks/test_claim_narrowing.py -v
  or: python .claude/hooks/test_claim_narrowing.py
"""
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Insert hooks dir so we can import subagent-stop functions directly.
# Because the module name has a hyphen, import via importlib.
import importlib.util

_HOOK_PATH = Path(__file__).resolve().parent / "subagent-stop.py"
_spec = importlib.util.spec_from_file_location("subagent_stop", _HOOK_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
check_claim_narrowing = _mod.check_claim_narrowing
THRESHOLD = _mod.CLAIM_NARROWING_THRESHOLD_MINUTES


def _write_claim(coordinators_dir: Path, session_id: str, claimed_paths, started_at: datetime, status: str = "active") -> None:
    """Helper: write a minimal YAML claim file."""
    import yaml
    data = {
        "session_id": session_id,
        "claimed_paths": claimed_paths,
        "started_at": started_at.isoformat(timespec="seconds"),
        "status": status,
        "heartbeat_at": started_at.isoformat(timespec="seconds"),
    }
    path = coordinators_dir / f"session-{session_id}.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


class TestClaimNarrowing:
    """Tests for check_claim_narrowing (warning-only, fail-open)."""

    def test_squatter_past_threshold_emits_warning(self, capsys):
        """Session with ['*'] older than threshold -> warning to stderr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            old_time = datetime.now() - timedelta(minutes=THRESHOLD + 5)
            _write_claim(coord_dir, "stale-squatter", ["*"], old_time)

            check_claim_narrowing(coord_dir)

            captured = capsys.readouterr()
            assert "claim-narrowing" in captured.err
            assert "stale-squatter" in captured.err

    def test_squatter_before_threshold_is_silent(self, capsys):
        """Session with ['*'] younger than threshold -> no warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            new_time = datetime.now() - timedelta(minutes=THRESHOLD - 2)
            _write_claim(coord_dir, "fresh-squatter", ["*"], new_time)

            check_claim_narrowing(coord_dir)

            captured = capsys.readouterr()
            assert "claim-narrowing" not in captured.err

    def test_narrowed_claim_is_silent(self, capsys):
        """Session with specific claimed_paths -> no warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            old_time = datetime.now() - timedelta(minutes=THRESHOLD + 30)
            _write_claim(coord_dir, "narrowed-session", [".claude/hooks/**", ".claude/rules/**"], old_time)

            check_claim_narrowing(coord_dir)

            captured = capsys.readouterr()
            assert "claim-narrowing" not in captured.err

    def test_ended_session_is_silent(self, capsys):
        """Ended session with ['*'] -> no warning (read_all_claims excludes ended)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            old_time = datetime.now() - timedelta(minutes=THRESHOLD + 30)
            _write_claim(coord_dir, "ended-squatter", ["*"], old_time, status="ended")

            check_claim_narrowing(coord_dir)

            captured = capsys.readouterr()
            assert "claim-narrowing" not in captured.err

    def test_missing_directory_is_silent(self, capsys):
        """Nonexistent coordinators_dir -> no crash, no output."""
        coord_dir = Path("/nonexistent/path/that/does/not/exist")

        # Must not raise
        check_claim_narrowing(coord_dir)

        captured = capsys.readouterr()
        assert "claim-narrowing" not in captured.err

    def test_malformed_yaml_is_silent(self, capsys):
        """Malformed YAML claim file -> no crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            bad_file = coord_dir / "session-bad.yaml"
            bad_file.write_text("{ this is: [not valid yaml", encoding="utf-8")

            # Must not raise
            check_claim_narrowing(coord_dir)

            # claim-narrowing warning should NOT appear for a file we couldn't parse
            captured = capsys.readouterr()
            assert "claim-narrowing" not in captured.err

    def test_missing_started_at_is_silent(self, capsys):
        """Claim with ['*'] but no started_at -> no crash, no warning."""
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            coord_dir = Path(tmpdir)
            data = {"session_id": "no-ts", "claimed_paths": ["*"], "status": "active"}
            (coord_dir / "session-no-ts.yaml").write_text(
                yaml.dump(data, default_flow_style=False), encoding="utf-8"
            )

            check_claim_narrowing(coord_dir)

            captured = capsys.readouterr()
            assert "claim-narrowing" not in captured.err


if __name__ == "__main__":
    # Allow running without pytest: python test_claim_narrowing.py
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
