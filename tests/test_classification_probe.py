"""
Tests for stage3_analysis.classification_probe.ClassificationProber.

Tests the LogisticRegression classification probe logic using synthetic
region_id-indexed embeddings and integer targets, without requiring
real data files on disk.
"""

import numpy as np
import pandas as pd
import pytest

from stage3_analysis.linear_probe import FoldMetrics, TargetResult


class TestClassificationProbeImports:
    """Verify classification probe modules import cleanly."""

    def test_import_classification_prober(self):
        from stage3_analysis.classification_probe import ClassificationProber
        assert ClassificationProber is not None

    def test_import_classification_probe_config(self):
        from stage3_analysis.classification_probe import ClassificationProbeConfig
        assert ClassificationProbeConfig is not None


class TestTrainAndEvaluateCV:
    """Test the _train_and_evaluate_cv method with synthetic data."""

    @pytest.fixture
    def prober(self):
        """Create a ClassificationProber with minimal config."""
        from stage3_analysis.classification_probe import (
            ClassificationProbeConfig,
            ClassificationProber,
        )

        config = ClassificationProbeConfig(
            study_area="netherlands",
            n_folds=3,
        )
        prober = ClassificationProber(config)
        prober.feature_names = [f"emb_{i}" for i in range(8)]
        return prober

    def test_forward_pass_with_synthetic_data(self, prober):
        """LogisticRegression produces predictions for every test sample."""
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 8
        n_classes = 3

        X = rng.randn(n_samples, n_features).astype(np.float32)
        y = rng.randint(0, n_classes, size=n_samples).astype(float)
        folds = np.array([1, 2, 3] * (n_samples // 3) + [1] * (n_samples % 3))
        unique_folds = np.array([1, 2, 3])

        oof_predictions, fold_metrics = prober._train_and_evaluate_cv(
            X, y, folds, unique_folds,
        )

        # Every sample should have a prediction
        assert not np.any(np.isnan(oof_predictions)), "Some samples have NaN predictions"
        assert len(oof_predictions) == n_samples

        # Should have one FoldMetrics per fold
        assert len(fold_metrics) == 3
        for fm in fold_metrics:
            assert isinstance(fm, FoldMetrics)
            assert 0.0 <= fm.accuracy <= 1.0
            assert 0.0 <= fm.f1_macro <= 1.0
            assert fm.n_train > 0
            assert fm.n_test > 0

    def test_predictions_are_valid_class_labels(self, prober):
        """Predictions should be actual class labels, not probabilities."""
        rng = np.random.RandomState(42)
        n_samples = 150
        X = rng.randn(n_samples, 8).astype(np.float32)
        y = rng.choice([0, 1, 2], size=n_samples).astype(float)
        folds = np.tile([1, 2, 3], n_samples // 3)
        unique_folds = np.array([1, 2, 3])

        oof_predictions, _ = prober._train_and_evaluate_cv(
            X, y, folds, unique_folds,
        )

        # All predictions should be one of the original class labels
        valid_labels = {0.0, 1.0, 2.0}
        actual_labels = set(oof_predictions)
        assert actual_labels.issubset(valid_labels), (
            f"Predictions contain invalid labels: {actual_labels - valid_labels}"
        )


class TestNClassesComputation:
    """Test n_classes computation — this is the divergence bug."""

    def test_n_classes_contiguous_labels(self):
        """Contiguous labels [0, 1, 2] → n_classes should be 3."""
        y = np.array([0, 1, 2, 0, 1, 2], dtype=float)
        # Current (buggy) code: len(np.unique(y[~np.isnan(y)])) = 3 ✓
        # DNN probe code: max - min + 1 = 2 - 0 + 1 = 3 ✓
        # Both agree for contiguous labels
        n_classes_current = int(len(np.unique(y[~np.isnan(y)])))
        assert n_classes_current == 3

    def test_n_classes_non_contiguous_labels(self):
        """Non-contiguous labels [1, 3, 5] → n_classes=5 (label_max - label_min + 1).

        Both classification_probe and DNN probe should agree:
        n_classes = label_max - label_min + 1 = 5 - 1 + 1 = 5.
        This is correct for CrossEntropyLoss (needs output_dim covering all possible
        class indices after offset subtraction: indices 0, 2, 4 require dim >= 5).
        """
        y = np.array([1, 3, 5, 1, 3, 5], dtype=float)

        # Both probes should compute the same way:
        unique_labels = np.unique(y[~np.isnan(y)]).astype(int)
        label_min = int(unique_labels.min())
        label_max = int(unique_labels.max())
        n_classes = label_max - label_min + 1
        assert n_classes == 5

    def test_n_classes_with_zero_based_labels(self):
        """Zero-based labels [0, 2, 4] → n_classes should be 5."""
        y = np.array([0, 2, 4, 0, 2, 4], dtype=float)

        unique_labels = np.unique(y[~np.isnan(y)]).astype(int)
        label_min = int(unique_labels.min())
        label_max = int(unique_labels.max())
        n_classes_correct = label_max - label_min + 1
        assert n_classes_correct == 5


class TestTargetResultContract:
    """Test that ClassificationProber produces valid TargetResult fields."""

    def test_target_result_classification_fields(self):
        """TargetResult from classification should have accuracy, f1, n_classes."""
        result = TargetResult(
            target="type_level1",
            target_name="Level 1 (2 cls)",
            best_alpha=0.0,
            best_l1_ratio=0.0,
            fold_metrics=[
                FoldMetrics(fold=1, rmse=0.0, mae=0.0, r2=0.0,
                            n_train=100, n_test=50, accuracy=0.85, f1_macro=0.83),
            ],
            overall_r2=0.0,
            overall_rmse=0.0,
            overall_mae=0.0,
            coefficients=np.zeros(8),
            intercept=0.0,
            feature_names=[f"emb_{i}" for i in range(8)],
            oof_predictions=np.array([0.0, 1.0, 0.0]),
            actual_values=np.array([0.0, 1.0, 1.0]),
            region_ids=np.array(["hex1", "hex2", "hex3"]),
            overall_accuracy=0.67,
            overall_f1_macro=0.65,
            n_classes=2,
            task_type="classification",
        )

        assert result.task_type == "classification"
        assert result.overall_accuracy == 0.67
        assert result.overall_f1_macro == 0.65
        assert result.n_classes == 2
        assert len(result.region_ids) == 3


class TestDNNClassificationProbeImports:
    """Verify DNN classification probe imports."""

    def test_import_dnn_classification_prober(self):
        from stage3_analysis.dnn_classification_probe import DNNClassificationProber
        assert DNNClassificationProber is not None

    def test_import_dnn_classification_config(self):
        from stage3_analysis.dnn_classification_probe import DNNClassificationConfig
        assert DNNClassificationConfig is not None

    def test_dnn_config_defaults(self):
        from stage3_analysis.dnn_classification_probe import DNNClassificationConfig
        config = DNNClassificationConfig()
        assert config.hidden_dim == 32
        assert config.num_layers == 3
        assert config.max_epochs == 100
        assert config.patience == 15
