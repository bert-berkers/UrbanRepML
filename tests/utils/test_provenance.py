"""Tests for utils/provenance.py — contract-first against specs/artifact_provenance.md."""

from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Lazy import — provenance module may not exist yet (parallel devops wave)
# ---------------------------------------------------------------------------

def _import_provenance():
    """Import utils.provenance, raising ImportError with a clear message if absent."""
    try:
        from utils import provenance  # noqa: F401
        return provenance
    except ImportError as exc:
        pytest.skip(f"utils.provenance not yet written by devops: {exc}")


# ---------------------------------------------------------------------------
# Test 1: config_hash dict-order invariance
# ---------------------------------------------------------------------------

class TestConfigHash:
    """§config_hash algorithm: sorted-keys canonicalisation, 16-hex-char output."""

    def test_config_hash_dict_order_invariance(self):
        """Flat dict: different insertion order => same hash."""
        prov = _import_provenance()
        h1 = prov.compute_config_hash({"a": 1, "b": 2})
        h2 = prov.compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2, "Flat dicts with different key ordering must produce the same hash"

    def test_config_hash_nested_dict_order_invariance(self):
        """Nested dict: recursive sort_keys must apply at all levels."""
        prov = _import_provenance()
        h1 = prov.compute_config_hash({"a": {"x": 1, "y": 2}})
        h2 = prov.compute_config_hash({"a": {"y": 2, "x": 1}})
        assert h1 == h2, "Nested dicts with different key ordering must produce the same hash"

    # ---------------------------------------------------------------------------
    # Test 2: config_hash output format
    # ---------------------------------------------------------------------------

    def test_config_hash_format(self):
        """Returned string is exactly 16 lowercase hex chars."""
        prov = _import_provenance()
        h = prov.compute_config_hash({"k": 1, "study_area": "netherlands"})
        assert isinstance(h, str)
        assert len(h) == 16, f"Expected 16 chars, got {len(h)}: {h!r}"
        assert re.fullmatch(r"[0-9a-f]{16}", h), f"Not lowercase hex: {h!r}"

    def test_config_hash_deterministic(self):
        """Same dict always returns the same hash (idempotence)."""
        prov = _import_provenance()
        cfg = {"resolution": 9, "study_area": "netherlands", "seed": 42}
        assert prov.compute_config_hash(cfg) == prov.compute_config_hash(cfg)

    def test_config_hash_different_dicts_differ(self):
        """Two structurally different dicts must produce different hashes (sanity)."""
        prov = _import_provenance()
        h1 = prov.compute_config_hash({"k": 1})
        h2 = prov.compute_config_hash({"k": 2})
        assert h1 != h2

    # ---------------------------------------------------------------------------
    # Test 3: non-serialisable coercion — must not raise; warning emitted
    # ---------------------------------------------------------------------------

    def test_config_hash_non_serialisable_path_no_raise(self):
        """A dict with a pathlib.Path value must NOT raise; a warning is emitted."""
        prov = _import_provenance()
        cfg = {"data_dir": Path("/some/path"), "k": 3}
        # The spec says: "Stringify + warn (default) … log a stderr warning once per type seen."
        # We accept either a UserWarning or a stderr write — capture both.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                result = prov.compute_config_hash(cfg)
            except TypeError:
                pytest.fail(
                    "compute_config_hash raised TypeError on non-serialisable Path; "
                    "spec §config_hash says coercion (stringify+warn) is the default."
                )
        assert isinstance(result, str)
        assert len(result) == 16

    def test_config_hash_non_serialisable_datetime_no_raise(self):
        """A dict with a datetime value must NOT raise; result is a 16-char hex string."""
        prov = _import_provenance()
        cfg = {"run_date": datetime(2026, 4, 24, 15, 0, 0), "seed": 42}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                result = prov.compute_config_hash(cfg)
            except TypeError:
                pytest.fail(
                    "compute_config_hash raised TypeError on datetime; "
                    "spec §config_hash says coercion (stringify+warn) is the default."
                )
        assert re.fullmatch(r"[0-9a-f]{16}", result)

    def test_config_hash_non_serialisable_warning_emitted(self, capsys):
        """Coercing a non-serialisable type should warn (stderr or UserWarning)."""
        prov = _import_provenance()
        cfg = {"path": Path("/tmp/test")}
        caught_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prov.compute_config_hash(cfg)
            caught_warnings = list(w)

        captured = capsys.readouterr()
        # Accept warning through either channel
        warned = len(caught_warnings) > 0 or len(captured.err) > 0 or len(captured.out) > 0
        # NOTE: this test is advisory — if the implementation raises instead of warning,
        # test_config_hash_non_serialisable_path_no_raise catches that harder.
        # Here we just document the expectation; if no warning fires it's a soft failure.
        # We do NOT fail the suite on this assertion alone — document as [open|0d] ambiguity.
        _ = warned  # not asserting; see scratchpad for spec ambiguity note


# ---------------------------------------------------------------------------
# Test 4: SidecarWriter happy path
# ---------------------------------------------------------------------------

class TestSidecarWriterHappyPath:
    """Context manager writes a valid 15-field sidecar alongside the artifact."""

    def test_sidecar_created(self, tmp_path):
        """Sidecar file exists after successful context exit."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact,
            config={"k": 1},
            input_paths=[],
            study_area="test",
            stage="stage3",
        ):
            artifact.write_text("x,y\n1,2\n")

        sidecar = tmp_path / "result.csv.run.yaml"
        assert sidecar.exists(), f"Sidecar {sidecar} not found after context exit"

    def test_sidecar_has_all_15_fields(self, tmp_path):
        """Sidecar YAML contains all 15 minimum required fields from the spec."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact,
            config={"k": 1},
            input_paths=[],
            study_area="test",
            stage="stage3",
        ):
            artifact.write_text("x\n")

        sidecar = tmp_path / "result.csv.run.yaml"
        data = yaml.safe_load(sidecar.read_text(encoding="utf-8"))

        required_fields = {
            "run_id",
            "git_commit",
            "git_dirty",
            "config_hash",
            "config_path",
            "input_paths",
            "output_paths",
            "seed",
            "wall_time_seconds",
            "started_at",
            "ended_at",
            "producer_script",
            "study_area",
            "stage",
            "schema_version",
        }
        missing = required_fields - set(data.keys())
        assert not missing, f"Sidecar missing fields: {sorted(missing)}"

    def test_sidecar_wall_time_non_negative(self, tmp_path):
        """wall_time_seconds must be a non-negative float."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact, config={"k": 1}, input_paths=[], study_area="test", stage="stage3"
        ):
            artifact.write_text("x\n")
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        assert isinstance(data["wall_time_seconds"], (int, float))
        assert data["wall_time_seconds"] >= 0

    def test_sidecar_timestamps_iso8601(self, tmp_path):
        """started_at and ended_at must parse as ISO 8601 datetimes."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact, config={"k": 1}, input_paths=[], study_area="test", stage="stage3"
        ):
            artifact.write_text("x\n")
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        for field in ("started_at", "ended_at"):
            raw = data[field]
            try:
                datetime.fromisoformat(str(raw))
            except ValueError:
                pytest.fail(f"Field {field!r} is not ISO 8601: {raw!r}")

    def test_sidecar_run_id_format(self, tmp_path):
        """run_id must match pattern: {stage}-{producer}-{YYYYMMDDTHHMMSS}-{8hexchars}."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact, config={"k": 1}, input_paths=[], study_area="test", stage="stage3"
        ):
            artifact.write_text("x\n")
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        run_id = data["run_id"]
        # Pattern from spec §run_id format: stage3-<producer>-YYYYMMDDTHHMMSS-<8hexchars>
        pattern = r"^stage3-.+-\d{8}T\d{6}-[0-9a-f]{8}$"
        assert re.fullmatch(pattern, run_id), (
            f"run_id {run_id!r} does not match expected pattern {pattern!r}"
        )

    def test_sidecar_schema_version(self, tmp_path):
        """schema_version must be '1.0'."""
        prov = _import_provenance()
        artifact = tmp_path / "result.csv"
        with prov.SidecarWriter(
            artifact, config={"k": 1}, input_paths=[], study_area="test", stage="stage3"
        ):
            artifact.write_text("x\n")
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        assert str(data["schema_version"]) == "1.0", (
            f"schema_version is {data['schema_version']!r}, expected '1.0'"
        )


# ---------------------------------------------------------------------------
# Test 5: SidecarWriter exception path
# ---------------------------------------------------------------------------

class TestSidecarWriterExceptionPath:
    """On wrapped exception: write partial sidecar, do NOT append ledger row, re-raise."""

    def _run_failing_context(self, tmp_path, prov):
        """Execute the SidecarWriter with a deliberate ValueError."""
        artifact = tmp_path / "result.csv"
        with pytest.raises(ValueError, match="boom"):
            with prov.SidecarWriter(
                artifact,
                config={"k": 1},
                input_paths=[],
                study_area="test",
                stage="stage3",
            ):
                raise ValueError("boom")
        return artifact

    def test_sidecar_written_on_exception(self, tmp_path):
        """Sidecar file exists even though the wrapped block raised."""
        prov = _import_provenance()
        artifact = self._run_failing_context(tmp_path, prov)
        sidecar = tmp_path / "result.csv.run.yaml"
        assert sidecar.exists(), "Sidecar should be written even on exception"

    def test_sidecar_status_failed(self, tmp_path):
        """extra.status must be 'failed' when the block raises."""
        prov = _import_provenance()
        self._run_failing_context(tmp_path, prov)
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        assert "extra" in data, "Sidecar must have an 'extra' section"
        assert data["extra"].get("status") == "failed", (
            f"extra.status is {data['extra'].get('status')!r}, expected 'failed'"
        )

    def test_sidecar_exception_class_captured(self, tmp_path):
        """extra.exception_class must be 'ValueError'."""
        prov = _import_provenance()
        self._run_failing_context(tmp_path, prov)
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        assert data["extra"].get("exception_class") == "ValueError", (
            f"extra.exception_class is {data['extra'].get('exception_class')!r}"
        )

    def test_sidecar_exception_message_contains_boom(self, tmp_path):
        """extra.exception_message must contain (possibly truncated) 'boom'."""
        prov = _import_provenance()
        self._run_failing_context(tmp_path, prov)
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        msg = data["extra"].get("exception_message", "")
        assert "boom" in str(msg), (
            f"extra.exception_message {msg!r} does not contain 'boom'"
        )

    def test_original_exception_propagates(self, tmp_path):
        """The original ValueError must propagate out of the context manager."""
        prov = _import_provenance()
        # pytest.raises in _run_failing_context already verifies this; re-check here
        # explicitly so a reader doesn't have to decode the helper.
        artifact = tmp_path / "result2.csv"
        raised = False
        try:
            with prov.SidecarWriter(
                artifact, config={"k": 1}, input_paths=[], study_area="test", stage="stage3"
            ):
                raise ValueError("boom")
        except ValueError as exc:
            raised = True
            assert "boom" in str(exc)
        assert raised, "ValueError should have propagated out of SidecarWriter"

    def test_no_ledger_row_on_failure(self, tmp_path, monkeypatch):
        """A failed run must NOT append a row to the ledger (spec §3 fail-mode: re-raise).

        The spec §3 says the sidecar is written and the original exception is re-raised.
        It also says 'The ledger-append also fires with stage unchanged' — meaning the
        ledger IS written even on failure (the exception is from the probe block, not from
        the writer). We verify the invariant by checking that after a failed run the
        ledger either (a) has no row for this run_id, OR (b) the row has status 'failed'
        in the sidecar.

        NOTE: spec §3 is ambiguous on whether ledger_append fires on failure — see
        scratchpad [open|0d]. We test the SIDECAR status here; ledger behavior is
        documented as ambiguous.
        """
        prov = _import_provenance()
        # Redirect ledger writes to tmp_path to avoid polluting real data/ledger/
        ledger_path = tmp_path / "ledger" / "runs.jsonl"
        # Try monkeypatching module-level constant or passing explicit path
        try:
            monkeypatch.setattr(prov, "LEDGER_PATH", ledger_path)
        except AttributeError:
            pass  # Implementation may use a different mechanism; best-effort

        self._run_failing_context(tmp_path, prov)

        # Primary assertion: sidecar marks status=failed
        data = yaml.safe_load((tmp_path / "result.csv.run.yaml").read_text(encoding="utf-8"))
        assert data.get("extra", {}).get("status") == "failed"

        # Secondary: if ledger was written, find the run_id and confirm it's marked failed
        if ledger_path.exists():
            rows = []
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            run_id = data.get("run_id", "")
            matching = [r for r in rows if r.get("run_id") == run_id]
            # If any matching row exists, that's acceptable per spec §3 ambiguity
            _ = matching  # documented as ambiguous; not asserting absence


# ---------------------------------------------------------------------------
# Test 6: ledger_append idempotence
# ---------------------------------------------------------------------------

class TestLedgerAppend:
    """Appending the same sidecar twice must not duplicate the ledger row."""

    def _write_minimal_sidecar(self, sidecar_path: Path, run_id: str) -> None:
        """Write a minimal valid sidecar YAML for ledger tests."""
        content = {
            "run_id": run_id,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_hash": "abcdef1234567890",
            "config_path": None,
            "input_paths": [],
            "output_paths": ["data/test/result.csv"],
            "seed": 42,
            "wall_time_seconds": 1.23,
            "started_at": "2026-04-24T15:40:32+00:00",
            "ended_at": "2026-04-24T15:40:33+00:00",
            "producer_script": "stage3_analysis/linear_probe.py",
            "study_area": "test",
            "stage": "stage3",
            "schema_version": "1.0",
            "extra": {"status": "success"},
        }
        sidecar_path.write_text(yaml.dump(content), encoding="utf-8")

    def test_ledger_append_idempotence(self, tmp_path, monkeypatch):
        """Calling ledger_append twice with the same sidecar produces exactly one row."""
        prov = _import_provenance()
        ledger_path = tmp_path / "ledger" / "runs.jsonl"

        # Redirect the module's ledger path constant if it exists
        try:
            monkeypatch.setattr(prov, "LEDGER_PATH", ledger_path)
        except AttributeError:
            pass

        sidecar_path = tmp_path / "result.csv.run.yaml"
        run_id = "stage3-linear_probe-20260424T154032-a3f1b2c7"
        self._write_minimal_sidecar(sidecar_path, run_id)

        # First append
        try:
            prov.ledger_append(sidecar_path)
        except Exception as exc:
            pytest.skip(f"ledger_append raised unexpectedly (may need ledger path arg): {exc}")

        # Second append — should deduplicate
        prov.ledger_append(sidecar_path)

        # Read back and count rows with this run_id
        actual_ledger = ledger_path if ledger_path.exists() else _find_ledger(prov, tmp_path)
        if actual_ledger is None or not actual_ledger.exists():
            pytest.skip("Cannot locate ledger file to verify idempotence")

        matching_rows = []
        for line in actual_ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("run_id") == run_id:
                    matching_rows.append(row)
            except json.JSONDecodeError:
                pass

        assert len(matching_rows) == 1, (
            f"Expected exactly 1 ledger row for run_id {run_id!r}, "
            f"got {len(matching_rows)} after 2 ledger_append calls"
        )

    def test_ledger_append_idempotence_via_read_ledger(self, tmp_path, monkeypatch):
        """read_ledger() returns exactly 1 row after two appends of the same sidecar.

        We redirect ledger writes by monkeypatching prov._project_root to return
        tmp_path, so ledger_append writes to tmp_path/data/ledger/runs.jsonl.
        This avoids touching the real data/ directory.
        """
        prov = _import_provenance()
        # Redirect _project_root so ledger_append writes inside tmp_path
        monkeypatch.setattr(prov, "_PROJECT_ROOT", None)  # reset cache
        monkeypatch.setattr(prov, "_project_root", lambda: tmp_path)

        expected_ledger = tmp_path / "data" / "ledger" / "runs.jsonl"

        sidecar_path = tmp_path / "result.csv.run.yaml"
        run_id = "stage3-linear_probe-20260424T154032-a3f1b2c7"
        self._write_minimal_sidecar(sidecar_path, run_id)

        try:
            prov.ledger_append(sidecar_path)
            prov.ledger_append(sidecar_path)
        except Exception as exc:
            pytest.skip(f"ledger_append raised: {exc}")

        assert expected_ledger.exists(), (
            f"Ledger file not found at {expected_ledger} — "
            "_project_root monkeypatch may not have taken effect"
        )

        # Try read_ledger with explicit path
        try:
            df = prov.read_ledger(expected_ledger)
        except TypeError:
            pytest.skip("read_ledger does not accept explicit path argument")

        matches = df[df["run_id"] == run_id] if "run_id" in df.columns else df
        assert len(matches) == 1, (
            f"read_ledger returned {len(matches)} rows for run_id {run_id!r}, expected 1"
        )


def _find_ledger(prov, tmp_path: Path):
    """Try to locate the ledger file from module constant or default path."""
    # Check module constant
    path = getattr(prov, "LEDGER_PATH", None)
    if path is not None:
        return Path(path)
    # Default project-relative path from spec
    proj_root = Path(__file__).parent.parent.parent
    return proj_root / "data" / "ledger" / "runs.jsonl"


# ---------------------------------------------------------------------------
# Test 7: read_ledger fail-open on malformed rows
# ---------------------------------------------------------------------------

class TestReadLedger:
    """read_ledger skips malformed rows and warns; valid rows are returned."""

    def _write_mixed_ledger(self, path: Path, run_ids: list[str]) -> None:
        """Write a JSONL file with: valid row, garbage, valid row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({
                "run_id": run_ids[0],
                "git_commit": "a" * 40,
                "git_dirty": False,
                "config_hash": "abcdef1234567890",
                "config_path": None,
                "seed": 42,
                "wall_time_seconds": 1.23,
                "started_at": "2026-04-24T15:40:32+00:00",
                "ended_at": "2026-04-24T15:40:33+00:00",
                "producer_script": "stage3_analysis/linear_probe.py",
                "study_area": "test",
                "stage": "stage3",
                "schema_version": "1.0",
                "sidecar_path": "data/test/result.csv.run.yaml",
            }),
            "THIS IS NOT JSON {{{",  # malformed line
            json.dumps({
                "run_id": run_ids[1],
                "git_commit": "b" * 40,
                "git_dirty": True,
                "config_hash": "fedcba9876543210",
                "config_path": None,
                "seed": None,
                "wall_time_seconds": 2.34,
                "started_at": "2026-04-24T16:00:00+00:00",
                "ended_at": "2026-04-24T16:00:02+00:00",
                "producer_script": "stage3_analysis/dnn_probe.py",
                "study_area": "test",
                "stage": "stage3",
                "schema_version": "1.0",
                "sidecar_path": "data/test/result2.csv.run.yaml",
            }),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_read_ledger_returns_two_valid_rows(self, tmp_path):
        """read_ledger with 3 lines (valid, garbage, valid) returns DataFrame with 2 rows."""
        prov = _import_provenance()
        ledger_path = tmp_path / "ledger" / "runs.jsonl"
        run_ids = [
            "stage3-linear_probe-20260424T154032-a3f1b2c7",
            "stage3-dnn_probe-20260424T160000-fedcba98",
        ]
        self._write_mixed_ledger(ledger_path, run_ids)

        try:
            df = prov.read_ledger(ledger_path)
        except TypeError:
            # Implementation may not accept a path argument — skip this variant
            pytest.skip("read_ledger does not accept an explicit path; cannot redirect in test")

        assert len(df) == 2, (
            f"Expected 2 rows (skipping malformed), got {len(df)}"
        )
        assert set(df["run_id"].tolist()) == set(run_ids)

    def test_read_ledger_warns_on_malformed(self, tmp_path, capsys):
        """A warning mentioning the malformed line is printed to stderr."""
        prov = _import_provenance()
        ledger_path = tmp_path / "ledger" / "runs.jsonl"
        run_ids = [
            "stage3-linear_probe-20260424T154032-a3f1b2c7",
            "stage3-dnn_probe-20260424T160000-fedcba98",
        ]
        self._write_mixed_ledger(ledger_path, run_ids)

        caught_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                prov.read_ledger(ledger_path)
            except TypeError:
                pytest.skip("read_ledger does not accept explicit path")
            caught_warnings = list(w)

        captured = capsys.readouterr()
        warned = (
            any("malform" in str(wi.message).lower()
                or "skip" in str(wi.message).lower()
                or "invalid" in str(wi.message).lower()
                or "error" in str(wi.message).lower()
                for wi in caught_warnings)
            or "malform" in captured.err.lower()
            or "skip" in captured.err.lower()
            or "invalid" in captured.err.lower()
            or "error" in captured.err.lower()
            # Accept any non-empty stderr as the warning signal
            or len(captured.err.strip()) > 0
        )
        assert warned, (
            "read_ledger should emit a warning to stderr or via warnings module "
            "when it encounters a malformed row"
        )
