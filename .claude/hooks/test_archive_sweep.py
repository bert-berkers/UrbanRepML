#!/usr/bin/env python3
"""Unit tests for archive_sweep.py.

Run: python -m pytest .claude/hooks/test_archive_sweep.py -v
  or: python .claude/hooks/test_archive_sweep.py

Tests use tmp_path fixtures — never touch the real .claude/ dirs.
"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

# Load archive_sweep from file path (module name has underscores, safe to import directly)
_SWEEP_PATH = Path(__file__).resolve().parent / "archive_sweep.py"
_spec = importlib.util.spec_from_file_location("archive_sweep", _SWEEP_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_dirs(tmp_path: Path) -> dict:
    """Create a minimal .claude-like directory tree under tmp_path."""
    claude = tmp_path / ".claude"
    supra_sessions = claude / "supra" / "sessions"
    supra_sessions.mkdir(parents=True)
    coordinators = claude / "coordinators"
    messages = coordinators / "messages"
    messages.mkdir(parents=True)
    terminals = coordinators / "terminals"
    terminals.mkdir(parents=True)
    return {
        "claude": claude,
        "supra_sessions": supra_sessions,
        "coordinators": coordinators,
        "messages": messages,
        "terminals": terminals,
        "gate": coordinators / ".last_archive_sweep",
    }


def _write_supra_yaml(supra_sessions: Path, session_id: str, last_attuned) -> Path:
    """Write a minimal supra session YAML."""
    data = {
        "supra_session_id": session_id,
        "last_attuned": last_attuned,
    }
    p = supra_sessions / f"{session_id}.yaml"
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return p


def _write_terminal_yaml(terminals: Path, pid: str, supra_session_id: str) -> Path:
    """Write a minimal terminal YAML referencing a supra session."""
    data = {
        "pid": pid,
        "session_id": f"coord-{pid}",
        "supra_session_id": supra_session_id,
    }
    p = terminals / f"{pid}.yaml"
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return p


def _make_sweep(dirs: dict):
    """Return a SweepRunner bound to tmp dirs, not real .claude/."""

    class SweepRunner:
        def __init__(self, d):
            self.dirs = d

        def maybe_run(self):
            # Monkeypatch module-level paths and call internal functions
            _mod.GATE_FILE = self.dirs["gate"]
            _mod.SUPRA_SESSIONS_DIR = self.dirs["supra_sessions"]
            _mod.COORDINATORS_DIR = self.dirs["coordinators"]
            _mod.maybe_run_sweep()

        def sweep_supra(self):
            _mod.GATE_FILE = self.dirs["gate"]
            _mod.SUPRA_SESSIONS_DIR = self.dirs["supra_sessions"]
            _mod.COORDINATORS_DIR = self.dirs["coordinators"]
            return _mod._sweep_supra_sessions()

        def sweep_messages(self):
            _mod.GATE_FILE = self.dirs["gate"]
            _mod.SUPRA_SESSIONS_DIR = self.dirs["supra_sessions"]
            _mod.COORDINATORS_DIR = self.dirs["coordinators"]
            return _mod._sweep_message_dirs()

        def read_gate(self):
            _mod.GATE_FILE = self.dirs["gate"]
            return _mod._read_gate()

    return SweepRunner(dirs)


# ── Gate file tests ─────────────────────────────────────────────────────────────

class TestGateLogic:
    def test_missing_gate_triggers_sweep(self, tmp_path: Path, capsys):
        """No .last_archive_sweep → sweep runs (creates/rewrites gate)."""
        dirs = _setup_dirs(tmp_path)
        runner = _make_sweep(dirs)
        assert not dirs["gate"].exists()

        runner.maybe_run()

        # Gate should be written after sweep
        assert dirs["gate"].exists()
        captured = capsys.readouterr()
        assert "sweep complete" in captured.err

    def test_recent_gate_skips_sweep(self, tmp_path: Path, capsys):
        """Gate < 24h old → sweep silently skipped."""
        dirs = _setup_dirs(tmp_path)
        recent = datetime.now(tz=timezone.utc) - timedelta(hours=12)
        dirs["gate"].write_text(recent.isoformat() + "\n", encoding="utf-8")

        runner = _make_sweep(dirs)
        runner.maybe_run()

        captured = capsys.readouterr()
        assert "sweep complete" not in captured.err

    def test_old_gate_triggers_sweep(self, tmp_path: Path, capsys):
        """Gate > 24h old → sweep runs."""
        dirs = _setup_dirs(tmp_path)
        old = datetime.now(tz=timezone.utc) - timedelta(hours=25)
        dirs["gate"].write_text(old.isoformat() + "\n", encoding="utf-8")

        runner = _make_sweep(dirs)
        runner.maybe_run()

        captured = capsys.readouterr()
        assert "sweep complete" in captured.err


# ── Supra session tests ────────────────────────────────────────────────────────

class TestSupraSessionSweep:
    def test_stale_supra_moved_to_archive(self, tmp_path: Path):
        """last_attuned 31d ago → moved to supra/sessions/archive/."""
        dirs = _setup_dirs(tmp_path)
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=31)).isoformat()
        _write_supra_yaml(dirs["supra_sessions"], "old-session", stale_date)

        runner = _make_sweep(dirs)
        moved, _ = runner.sweep_supra()

        assert moved == 1
        archive = dirs["supra_sessions"] / "archive" / "old-session.yaml"
        assert archive.exists()
        assert not (dirs["supra_sessions"] / "old-session.yaml").exists()

    def test_recent_supra_stays(self, tmp_path: Path):
        """last_attuned 10d ago → left in place."""
        dirs = _setup_dirs(tmp_path)
        recent_date = (datetime.now(tz=timezone.utc) - timedelta(days=10)).isoformat()
        _write_supra_yaml(dirs["supra_sessions"], "recent-session", recent_date)

        runner = _make_sweep(dirs)
        moved, skipped = runner.sweep_supra()

        assert moved == 0
        assert (dirs["supra_sessions"] / "recent-session.yaml").exists()

    def test_live_terminal_reference_prevents_move(self, tmp_path: Path):
        """Stale supra YAML referenced by live terminal → NOT moved."""
        dirs = _setup_dirs(tmp_path)
        stale_date = (datetime.now(tz=timezone.utc) - timedelta(days=31)).isoformat()
        _write_supra_yaml(dirs["supra_sessions"], "live-session", stale_date)
        _write_terminal_yaml(dirs["terminals"], "12345", "live-session")

        runner = _make_sweep(dirs)
        moved, skipped = runner.sweep_supra()

        assert moved == 0
        assert skipped == 1
        assert (dirs["supra_sessions"] / "live-session.yaml").exists()

    def test_malformed_supra_yaml_skipped_silently(self, tmp_path: Path, capsys):
        """Malformed YAML → skipped with WARN, no crash."""
        dirs = _setup_dirs(tmp_path)
        bad = dirs["supra_sessions"] / "bad-session.yaml"
        bad.write_text("{ this is: [not valid yaml", encoding="utf-8")

        runner = _make_sweep(dirs)
        moved, skipped = runner.sweep_supra()

        assert moved == 0
        assert skipped == 1
        captured = capsys.readouterr()
        assert "malformed" in captured.err or "WARN" in captured.err
        assert bad.exists()  # Not removed

    def test_missing_last_attuned_skipped(self, tmp_path: Path):
        """last_attuned field missing → skipped silently."""
        dirs = _setup_dirs(tmp_path)
        data = {"supra_session_id": "no-ts-session"}
        p = dirs["supra_sessions"] / "no-ts-session.yaml"
        p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")

        runner = _make_sweep(dirs)
        moved, skipped = runner.sweep_supra()

        assert moved == 0
        assert skipped == 1
        assert p.exists()


# ── Message dir tests ──────────────────────────────────────────────────────────

class TestMessageDirSweep:
    def test_old_messages_dir_moved_to_archive(self, tmp_path: Path):
        """Message dir 8d old → moved to messages/archive/YYYY-MM-DD/."""
        dirs = _setup_dirs(tmp_path)
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=8)).date()
        old_dir = dirs["messages"] / old_date.isoformat()
        old_dir.mkdir()
        (old_dir / "msg.yaml").write_text("content", encoding="utf-8")

        runner = _make_sweep(dirs)
        moved, _ = runner.sweep_messages()

        assert moved == 1
        archive = dirs["messages"] / "archive" / old_date.isoformat()
        assert archive.exists()
        assert not old_dir.exists()

    def test_recent_messages_dir_stays(self, tmp_path: Path):
        """Message dir 3d old → left in place."""
        dirs = _setup_dirs(tmp_path)
        recent_date = (datetime.now(tz=timezone.utc) - timedelta(days=3)).date()
        recent_dir = dirs["messages"] / recent_date.isoformat()
        recent_dir.mkdir()

        runner = _make_sweep(dirs)
        moved, _ = runner.sweep_messages()

        assert moved == 0
        assert recent_dir.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
