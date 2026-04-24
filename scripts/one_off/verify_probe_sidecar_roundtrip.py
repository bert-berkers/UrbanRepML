"""
Verify cluster-2 W2a probe sidecar wiring end-to-end (synthetic round-trip).

Lifetime: temporary (30d). Exercises stage3 LinearProbeRegressor.save_results() path
with mock TargetResult objects to confirm: (1) primary metrics_summary.csv.run.yaml
sidecar is written alongside the artifact; (2) data/ledger/runs.jsonl gets a row;
(3) the sidecar + row carry the 15 required fields.

Runs in-process with a tempdir project root fake is overkill — instead we point
the probe at a throwaway output_dir and let ledger_append touch the real
data/ledger/ (rows are data, not tracked). This mirrors a real probe run.

Usage:
    uv run python scripts/one_off/verify_probe_sidecar_roundtrip.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3_analysis.linear_probe import (  # noqa: E402
    FoldMetrics,
    LinearProbeConfig,
    LinearProbeRegressor,
    TargetResult,
)
from utils.provenance import read_ledger  # noqa: E402


def make_mock_result(target: str, n_rows: int = 50) -> TargetResult:
    rng = np.random.default_rng(seed=hash(target) % (2**32))
    preds = rng.standard_normal(n_rows).astype(np.float64)
    actual = preds + 0.1 * rng.standard_normal(n_rows)
    region_ids = np.array([f"8a{target}{i:010x}ffff" for i in range(n_rows)], dtype=object)
    return TargetResult(
        target=target,
        target_name=f"Mock {target}",
        best_alpha=1.0,
        best_l1_ratio=0.5,
        fold_metrics=[
            FoldMetrics(fold=k, rmse=0.1, mae=0.08, r2=0.5, n_train=40, n_test=10)
            for k in range(5)
        ],
        overall_r2=0.52,
        overall_rmse=0.11,
        overall_mae=0.09,
        coefficients=rng.standard_normal(8),
        intercept=0.0,
        feature_names=[f"emb_{i}" for i in range(8)],
        oof_predictions=preds,
        actual_values=actual,
        region_ids=region_ids,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "lp_synthetic"
        config = LinearProbeConfig(
            study_area="netherlands",
            year=2022,
            h3_resolution=10,
            modality="alphaearth",
            target_name="leefbaarometer",
            n_folds=5,
            run_descriptor="",  # flat dir; we supply output_dir directly below
        )

        probe = LinearProbeRegressor(config, project_root=PROJECT_ROOT)
        probe.feature_names = [f"emb_{i}" for i in range(8)]
        probe._loaded_embedding_path = (
            PROJECT_ROOT
            / "data/study_areas/netherlands/stage1_unimodal/alphaearth/res10_2022.parquet"
        )
        probe.results = {t: make_mock_result(t) for t in ["lbm", "fys"]}

        ledger_before = len(read_ledger())
        final_out = probe.save_results(output_dir=out_dir)

        sidecar_path = out_dir / "metrics_summary.csv.run.yaml"
        assert sidecar_path.exists(), f"missing sidecar: {sidecar_path}"

        sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        required_15 = [
            "run_id", "git_commit", "git_dirty", "config_hash", "config_path",
            "input_paths", "output_paths", "seed", "wall_time_seconds",
            "started_at", "ended_at", "producer_script", "study_area",
            "stage", "schema_version",
        ]
        missing = [k for k in required_15 if k not in sidecar]
        assert not missing, f"sidecar missing keys: {missing}"
        assert sidecar["stage"] == "stage3"
        assert sidecar["study_area"] == "netherlands"
        assert sidecar["seed"] == 42
        assert sidecar["schema_version"] == "1.0"
        assert sidecar["producer_script"] == "stage3_analysis/linear_probe.py"
        assert sidecar["extra"]["probe_type"] == "linear"
        assert sidecar["extra"]["status"] == "success"
        assert len(sidecar["input_paths"]) >= 1
        assert len(sidecar["output_paths"]) >= 3  # csv, coef, config, + preds

        ledger_after = read_ledger()
        assert (
            len(ledger_after) == ledger_before + 1
        ), f"ledger delta != 1 (before={ledger_before}, after={len(ledger_after)})"
        # Look up by run_id rather than assuming sort order — read_ledger sorts by
        # `started_at`, but a single ledger may contain rows written out-of-order
        # relative to wall-clock (e.g. earlier test runs with different clocks).
        matches = ledger_after[ledger_after["run_id"] == sidecar["run_id"]]
        assert len(matches) == 1, (
            f"expected exactly 1 ledger row with run_id={sidecar['run_id']}, "
            f"got {len(matches)}"
        )
        new_row = matches.iloc[0]
        assert new_row["stage"] == "stage3"
        assert new_row["producer_script"] == "stage3_analysis/linear_probe.py"
        assert new_row["config_hash"] == sidecar["config_hash"]
        assert new_row["sidecar_path"].endswith("metrics_summary.csv.run.yaml")

        print("OK  sidecar written at:", sidecar_path.relative_to(Path(tmp)))
        print("OK  run_id:", sidecar["run_id"])
        print("OK  output_paths count:", len(sidecar["output_paths"]))
        print("OK  ledger delta: +1 row")
        print("OK  synthetic round-trip verified")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
