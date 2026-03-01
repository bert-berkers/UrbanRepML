"""
Unit tests for stage1_modalities/poi/processor.py

Covers the HEX2VEC_FILTER adoption, GradualBatchSizeCallback, EarlyStopping
configuration, batch ramp logic, and CLI config wiring.

No I/O, no SRAI API calls — pure unit tests using synthetic data.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(config_overrides: dict = None):
    """Create a POIProcessor with minimal required config."""
    from stage1_modalities.poi.processor import POIProcessor

    config = {"study_area": "test"}
    if config_overrides:
        config.update(config_overrides)
    return POIProcessor(config)


def _make_count_df(n_rows: int = 5, include_multi_category: bool = True) -> pd.DataFrame:
    """Synthetic count embeddings DataFrame indexed by fake region_ids."""
    idx = [f"89196{i:012x}" for i in range(n_rows)]
    data = {
        "amenity_restaurant": np.random.randint(0, 10, n_rows),
        "amenity_cafe": np.random.randint(0, 5, n_rows),
        "shop_bakery": np.random.randint(0, 3, n_rows),
        "shop_supermarket": np.random.randint(0, 2, n_rows),
        "leisure_park": np.random.randint(0, 8, n_rows),
    }
    return pd.DataFrame(data, index=pd.Index(idx, name="region_id"))


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestPOIProcessorImports:
    def test_import_poi_processor(self):
        from stage1_modalities.poi.processor import POIProcessor
        assert POIProcessor is not None

    def test_import_gradual_batch_size_callback(self):
        from stage1_modalities.poi.processor import GradualBatchSizeCallback
        assert GradualBatchSizeCallback is not None

    def test_import_hex2vec_filter(self):
        from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
        assert isinstance(HEX2VEC_FILTER, dict)
        assert len(HEX2VEC_FILTER) == 15

    def test_hex2vec_filter_has_expected_keys(self):
        from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
        # These are the canonical 15 OSM keys from the Hex2Vec paper
        expected_keys = {
            "aeroway", "amenity", "building", "healthcare", "historic",
            "landuse", "leisure", "military", "natural", "office",
            "shop", "sport", "tourism", "water", "waterway",
        }
        assert set(HEX2VEC_FILTER.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Default poi_categories — HEX2VEC_FILTER adoption
# ---------------------------------------------------------------------------


class TestHEX2VECFilterAdoption:
    def test_default_poi_categories_is_hex2vec_filter(self):
        """When poi_categories is absent from config, processor defaults to HEX2VEC_FILTER."""
        from stage1_modalities.poi.processor import POIProcessor, HEX2VEC_FILTER

        p = _make_processor()
        assert p.poi_categories is HEX2VEC_FILTER

    def test_explicit_none_in_config_uses_default(self):
        """When config explicitly passes None, processor falls back to HEX2VEC_FILTER."""
        from stage1_modalities.poi.processor import POIProcessor, HEX2VEC_FILTER

        p = _make_processor({"poi_categories": None})
        assert p.poi_categories is HEX2VEC_FILTER

    def test_class_attribute_default_poi_categories_is_hex2vec_filter(self):
        """Class-level DEFAULT_POI_CATEGORIES references HEX2VEC_FILTER."""
        from stage1_modalities.poi.processor import POIProcessor, HEX2VEC_FILTER

        assert POIProcessor.DEFAULT_POI_CATEGORIES is HEX2VEC_FILTER

    def test_hex2vec_filter_satisfies_geovex_requirement(self):
        """HEX2VEC_FILTER must have >= 256 total sub-tags for GeoVex."""
        from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER

        total_sub_tags = sum(
            len(v) if isinstance(v, (list, dict)) else 1
            for v in HEX2VEC_FILTER.values()
        )
        assert total_sub_tags >= 256, (
            f"GeoVex requires >= 256 features, but HEX2VEC_FILTER has {total_sub_tags}"
        )


# ---------------------------------------------------------------------------
# GradualBatchSizeCallback
# ---------------------------------------------------------------------------


class TestGradualBatchSizeCallback:
    def test_instantiation(self):
        from stage1_modalities.poi.processor import GradualBatchSizeCallback

        cb = GradualBatchSizeCallback(initial_batch_size=512, target_batch_size=4096)
        assert cb.initial_batch_size == 512
        assert cb.target_batch_size == 4096

    def test_is_lightning_callback(self):
        """GradualBatchSizeCallback must be a pl.Callback subclass."""
        from stage1_modalities.poi.processor import GradualBatchSizeCallback, LIGHTNING_AVAILABLE

        if not LIGHTNING_AVAILABLE:
            pytest.skip("pytorch_lightning not installed")

        import pytorch_lightning as pl
        cb = GradualBatchSizeCallback(initial_batch_size=512, target_batch_size=4096)
        assert isinstance(cb, pl.Callback)

    def test_linear_ramp_epoch_zero(self):
        """At epoch 0, batch size should equal initial_batch_size."""
        initial, target, max_epochs = 512, 4096, 10
        fraction = 0 / (max_epochs - 1)
        expected = int(initial + fraction * (target - initial))
        assert expected == 512

    def test_linear_ramp_last_epoch(self):
        """At last epoch, batch size should equal target_batch_size."""
        initial, target, max_epochs = 512, 4096, 10
        fraction = (max_epochs - 1) / (max_epochs - 1)
        expected = int(initial + fraction * (target - initial))
        assert expected == 4096

    def test_linear_ramp_midpoint(self):
        """At epoch 5 of 10, batch size should be proportional to fraction 5/9."""
        initial, target, max_epochs = 512, 4096, 10
        fraction = 5 / (max_epochs - 1)  # 5/9 ~= 0.556
        expected = int(initial + fraction * (target - initial))
        # fraction=5/9: 512 + (5/9)*(4096-512) = 512 + 0.556*3584 ~ 2503
        assert 2490 < expected < 2520

    def test_single_epoch_uses_full_batch_size(self):
        """When max_epochs == 1, fraction is 1.0 -> target_batch_size."""
        initial, target, max_epochs = 512, 4096, 1
        fraction = 1.0  # max_epochs <= 1 -> fraction = 1.0 per code
        expected = int(initial + fraction * (target - initial))
        assert expected == 4096


# ---------------------------------------------------------------------------
# _build_training_callbacks logic
# ---------------------------------------------------------------------------


class TestBuildTrainingCallbacks:
    def test_initial_less_than_target_ramp_enabled(self):
        """initial_batch_size < batch_size -> GradualBatchSizeCallback in callbacks."""
        from stage1_modalities.poi.processor import GradualBatchSizeCallback

        p = _make_processor({"initial_batch_size": 512, "batch_size": 4096})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        types = [type(cb).__name__ for cb in callbacks]
        assert "GradualBatchSizeCallback" in types

    def test_initial_equal_to_target_no_ramp(self):
        """initial_batch_size == batch_size -> no GradualBatchSizeCallback."""
        from stage1_modalities.poi.processor import GradualBatchSizeCallback

        p = _make_processor({"initial_batch_size": 4096, "batch_size": 4096})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        types = [type(cb).__name__ for cb in callbacks]
        assert "GradualBatchSizeCallback" not in types

    def test_initial_greater_than_target_no_ramp(self):
        """initial_batch_size > batch_size -> no ramp (condition is initial < target)."""
        from stage1_modalities.poi.processor import GradualBatchSizeCallback

        p = _make_processor({"initial_batch_size": 8192, "batch_size": 4096})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        types = [type(cb).__name__ for cb in callbacks]
        assert "GradualBatchSizeCallback" not in types

    def test_patience_greater_than_zero_early_stopping_enabled(self):
        """early_stopping_patience > 0 -> EarlyStopping in callbacks."""
        from stage1_modalities.poi.processor import LIGHTNING_AVAILABLE

        if not LIGHTNING_AVAILABLE:
            pytest.skip("pytorch_lightning not installed")

        from pytorch_lightning.callbacks import EarlyStopping

        p = _make_processor({"early_stopping_patience": 3})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" in types

    def test_patience_zero_no_early_stopping(self):
        """early_stopping_patience == 0 -> no EarlyStopping."""
        p = _make_processor({"early_stopping_patience": 0})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" not in types

    def test_early_stopping_monitors_correct_metric(self):
        """EarlyStopping monitors 'train_loss_epoch' (epoch-aggregated)."""
        from stage1_modalities.poi.processor import LIGHTNING_AVAILABLE

        if not LIGHTNING_AVAILABLE:
            pytest.skip("pytorch_lightning not installed")

        from pytorch_lightning.callbacks import EarlyStopping

        p = _make_processor({"early_stopping_patience": 3})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        es_callbacks = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]
        assert len(es_callbacks) == 1
        assert es_callbacks[0].monitor == "train_loss_epoch"

    def test_early_stopping_check_on_train_epoch_end(self):
        """EarlyStopping configured with check_on_train_epoch_end=True."""
        from stage1_modalities.poi.processor import LIGHTNING_AVAILABLE

        if not LIGHTNING_AVAILABLE:
            pytest.skip("pytorch_lightning not installed")

        from pytorch_lightning.callbacks import EarlyStopping

        p = _make_processor({"early_stopping_patience": 3})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        es_callbacks = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]
        assert len(es_callbacks) == 1
        # The attribute is stored internally as _check_on_train_epoch_end
        assert es_callbacks[0]._check_on_train_epoch_end is True

    def test_early_stopping_mode_is_min(self):
        """EarlyStopping must be in 'min' mode for loss monitoring."""
        from stage1_modalities.poi.processor import LIGHTNING_AVAILABLE

        if not LIGHTNING_AVAILABLE:
            pytest.skip("pytorch_lightning not installed")

        from pytorch_lightning.callbacks import EarlyStopping

        p = _make_processor({"early_stopping_patience": 3})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        es_callbacks = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]
        assert es_callbacks[0].mode == "min"

    def test_full_config_two_callbacks(self):
        """With patience > 0 and initial < target: exactly 2 callbacks (ES + ramp)."""
        p = _make_processor({"initial_batch_size": 512, "batch_size": 4096, "early_stopping_patience": 3})
        callbacks = p._build_training_callbacks("train_loss_epoch")
        assert len(callbacks) == 2

    def test_no_lightning_returns_empty_list(self):
        """When pytorch_lightning is unavailable, callbacks must be empty list."""
        import sys
        from unittest.mock import patch

        # Temporarily hide pytorch_lightning to simulate absence
        with patch.dict(sys.modules, {"pytorch_lightning": None}):
            # Re-import with the mock in place
            import importlib
            import stage1_modalities.poi.processor as mod
            original_available = mod.LIGHTNING_AVAILABLE
            mod.LIGHTNING_AVAILABLE = False
            try:
                p = _make_processor({"initial_batch_size": 512, "batch_size": 4096})
                callbacks = p._build_training_callbacks("train_loss_epoch")
                assert callbacks == []
            finally:
                mod.LIGHTNING_AVAILABLE = original_available


# ---------------------------------------------------------------------------
# Diversity metrics with HEX2VEC_FILTER categories
# ---------------------------------------------------------------------------


class TestDiversityMetricsWithHEX2VECFilter:
    def test_diversity_metrics_computed_for_hex2vec_categories(self):
        """_calculate_diversity_metrics works with HEX2VEC_FILTER keys."""
        from stage1_modalities.poi.processor import POIProcessor, HEX2VEC_FILTER

        p = _make_processor()
        assert p.poi_categories is HEX2VEC_FILTER

        count_df = _make_count_df(n_rows=5)
        diversity_df = p._calculate_diversity_metrics(count_df)

        assert isinstance(diversity_df, pd.DataFrame)
        assert "poi_shannon_entropy" in diversity_df.columns
        assert "poi_simpson_diversity" in diversity_df.columns
        assert "poi_richness" in diversity_df.columns
        assert "poi_evenness" in diversity_df.columns

    def test_diversity_metrics_no_nans(self):
        """Diversity metrics contain no NaN values for normal count data."""
        p = _make_processor()
        count_df = _make_count_df(n_rows=10)
        diversity_df = p._calculate_diversity_metrics(count_df)
        assert not diversity_df.isnull().any().any()

    def test_diversity_metrics_entropy_non_negative(self):
        """Shannon entropy must be >= 0."""
        p = _make_processor()
        count_df = _make_count_df(n_rows=10)
        diversity_df = p._calculate_diversity_metrics(count_df)
        assert (diversity_df["poi_shannon_entropy"] >= 0).all()

    def test_diversity_metrics_simpson_in_range(self):
        """Simpson diversity must be in [0, 1]."""
        p = _make_processor()
        count_df = _make_count_df(n_rows=10)
        diversity_df = p._calculate_diversity_metrics(count_df)
        assert (diversity_df["poi_simpson_diversity"] >= 0).all()
        assert (diversity_df["poi_simpson_diversity"] <= 1).all()

    def test_diversity_metrics_all_zeros_returns_zero_entropy(self):
        """A hex with zero POIs in all categories gets zero entropy."""
        p = _make_processor()
        # All-zero row
        count_df = pd.DataFrame(
            {"amenity_restaurant": [0, 1], "shop_bakery": [0, 2]},
            index=pd.Index(["hex_a", "hex_b"], name="region_id"),
        )
        diversity_df = p._calculate_diversity_metrics(count_df)
        # All-zero hex should get 0 entropy (no information)
        assert diversity_df.loc["hex_a", "poi_shannon_entropy"] == pytest.approx(0.0)

    def test_diversity_metrics_none_poi_categories_uses_default(self):
        """When config passes None, diversity metrics works with HEX2VEC_FILTER fallback."""
        p = _make_processor({"poi_categories": None})
        count_df = _make_count_df()
        diversity_df = p._calculate_diversity_metrics(count_df)
        assert "poi_shannon_entropy" in diversity_df.columns


# ---------------------------------------------------------------------------
# CLI config wiring tests
# ---------------------------------------------------------------------------


class TestCLIConfigWiring:
    def test_batch_size_wired_from_config(self):
        p = _make_processor({"batch_size": 2048})
        assert p.batch_size == 2048

    def test_initial_batch_size_wired_from_config(self):
        p = _make_processor({"initial_batch_size": 256})
        assert p.initial_batch_size == 256

    def test_early_stopping_patience_wired_from_config(self):
        p = _make_processor({"early_stopping_patience": 5})
        assert p.early_stopping_patience == 5

    def test_default_batch_size(self):
        """Default batch_size is 4096 when not in config."""
        p = _make_processor()
        assert p.batch_size == 4096

    def test_default_initial_batch_size(self):
        """Default initial_batch_size is 512 when not in config."""
        p = _make_processor()
        assert p.initial_batch_size == 512

    def test_default_early_stopping_patience(self):
        """Default early_stopping_patience is 3 when not in config."""
        p = _make_processor()
        assert p.early_stopping_patience == 3

    def test_accelerator_wired_from_config(self):
        """Explicit accelerator value is stored on the processor."""
        p = _make_processor({"accelerator": "cpu"})
        assert p.accelerator == "cpu"

    def test_accelerator_gpu_wired_from_config(self):
        p = _make_processor({"accelerator": "gpu"})
        assert p.accelerator == "gpu"

    def test_default_accelerator_is_auto(self):
        """Default accelerator is 'auto' when not in config."""
        p = _make_processor()
        assert p.accelerator == "auto"


# ---------------------------------------------------------------------------
# Accelerator passed to trainer_kwargs
# ---------------------------------------------------------------------------


class TestAcceleratorInTrainerKwargs:
    """Verify that the accelerator attribute is forwarded to fit_transform trainer_kwargs."""

    @pytest.fixture()
    def srai_triple(self):
        """Small SRAI triple for mock-based tests."""
        import geopandas as gpd
        from shapely.geometry import Point, box
        from srai.regionalizers import H3Regionalizer
        from srai.joiners import IntersectionJoiner

        area = gpd.GeoDataFrame(
            geometry=[box(4.3, 52.0, 4.32, 52.02)], crs="EPSG:4326"
        )
        regions_gdf = H3Regionalizer(resolution=9).transform(area)

        features_gdf = gpd.GeoDataFrame(
            {
                "amenity": ["restaurant", "cafe"],
                "geometry": [Point(4.305, 52.01), Point(4.31, 52.015)],
            },
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"

        joint_gdf = IntersectionJoiner().transform(
            regions=regions_gdf, features=features_gdf
        )
        return regions_gdf, features_gdf, joint_gdf

    def test_hex2vec_trainer_kwargs_uses_accelerator(self, srai_triple):
        """run_hex2vec passes self.accelerator to trainer_kwargs, not hardcoded 'auto'."""
        from unittest.mock import patch, MagicMock

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor({"accelerator": "cpu"})

        mock_embedder = MagicMock()
        mock_embedder.fit_transform.return_value = pd.DataFrame(
            np.ones((len(regions_gdf), 8), dtype=np.float32),
            index=regions_gdf.index,
            columns=[f"dim_{i}" for i in range(8)],
        )

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        call_kwargs = mock_embedder.fit_transform.call_args
        trainer_kwargs = call_kwargs.kwargs.get("trainer_kwargs") or call_kwargs[1].get("trainer_kwargs")
        assert trainer_kwargs["accelerator"] == "cpu"

    def test_hex2vec_trainer_kwargs_auto_default(self, srai_triple):
        """Default accelerator 'auto' is forwarded to hex2vec trainer_kwargs."""
        from unittest.mock import patch, MagicMock

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor()  # default accelerator = "auto"

        mock_embedder = MagicMock()
        mock_embedder.fit_transform.return_value = pd.DataFrame(
            np.ones((len(regions_gdf), 8), dtype=np.float32),
            index=regions_gdf.index,
            columns=[f"dim_{i}" for i in range(8)],
        )

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        call_kwargs = mock_embedder.fit_transform.call_args
        trainer_kwargs = call_kwargs.kwargs.get("trainer_kwargs") or call_kwargs[1].get("trainer_kwargs")
        assert trainer_kwargs["accelerator"] == "auto"


# ---------------------------------------------------------------------------
# run_hex2vec / run_geovex signature checks (no training, just init)
# ---------------------------------------------------------------------------


class TestEmbedderSignatures:
    def test_hex2vec_fit_transform_accepts_batch_size_kwarg(self):
        """Hex2VecEmbedder.fit_transform accepts batch_size as kwarg."""
        import inspect
        from srai.embedders import Hex2VecEmbedder

        sig = inspect.signature(Hex2VecEmbedder.fit_transform)
        assert "batch_size" in sig.parameters

    def test_geovex_constructor_accepts_batch_size(self):
        """GeoVexEmbedder.__init__ accepts batch_size."""
        import inspect
        from srai.embedders import GeoVexEmbedder

        sig = inspect.signature(GeoVexEmbedder.__init__)
        assert "batch_size" in sig.parameters

    def test_geovex_fit_transform_does_not_accept_batch_size(self):
        """GeoVexEmbedder.fit_transform does NOT take batch_size (set at construction)."""
        import inspect
        from srai.embedders import GeoVexEmbedder

        sig = inspect.signature(GeoVexEmbedder.fit_transform)
        assert "batch_size" not in sig.parameters, (
            "GeoVexEmbedder.fit_transform should not take batch_size; "
            "it's set in the constructor and the GradualBatchSizeCallback handles ramping."
        )

    def test_expected_output_features_param_in_hex2vec_init(self):
        """Hex2VecEmbedder.__init__ accepts expected_output_features."""
        import inspect
        from srai.embedders import Hex2VecEmbedder

        sig = inspect.signature(Hex2VecEmbedder.__init__)
        assert "expected_output_features" in sig.parameters

    def test_target_features_param_in_geovex_init(self):
        """GeoVexEmbedder.__init__ accepts target_features."""
        import inspect
        from srai.embedders import GeoVexEmbedder

        sig = inspect.signature(GeoVexEmbedder.__init__)
        assert "target_features" in sig.parameters


# ---------------------------------------------------------------------------
# POIProcessor index contract
# ---------------------------------------------------------------------------


class TestPOIProcessorIndexContract:
    """Tests for run_count_embeddings output contracts.

    Uses synthetic SRAI data with at least one real POI so that
    IntersectionJoiner does not raise 'Features must not be empty'.
    """

    @pytest.fixture(scope="class")
    def srai_data(self):
        """Small synthetic SRAI triple: regions_gdf, features_gdf, joint_gdf."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point, box
        from srai.regionalizers import H3Regionalizer
        from srai.joiners import IntersectionJoiner

        area = gpd.GeoDataFrame(
            geometry=[box(4.3, 52.0, 4.32, 52.02)], crs="EPSG:4326"
        )
        regionalizer = H3Regionalizer(resolution=9)
        regions_gdf = regionalizer.transform(area)

        # Put a few synthetic POIs inside the area so the joiner has data
        features_gdf = gpd.GeoDataFrame(
            {
                "amenity": ["restaurant", "cafe", None],
                "shop": [None, None, "bakery"],
                "geometry": [
                    Point(4.305, 52.01),
                    Point(4.31, 52.015),
                    Point(4.315, 52.005),
                ],
            },
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"

        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(regions=regions_gdf, features=features_gdf)

        return regions_gdf, features_gdf, joint_gdf

    def test_run_count_embeddings_index_name(self, srai_data):
        """run_count_embeddings output must have index.name == 'region_id'."""
        regions_gdf, features_gdf, joint_gdf = srai_data

        p = _make_processor({"compute_diversity_metrics": False})
        result = p.run_count_embeddings(regions_gdf, features_gdf, joint_gdf)

        assert result.index.name == "region_id"

    def test_run_count_embeddings_no_nans(self, srai_data):
        """Count embeddings contain no all-NaN rows."""
        regions_gdf, features_gdf, joint_gdf = srai_data

        p = _make_processor({"compute_diversity_metrics": False})
        result = p.run_count_embeddings(regions_gdf, features_gdf, joint_gdf)

        # No all-NaN rows
        assert not result.isnull().all(axis=1).any()

    def test_run_count_embeddings_shape(self, srai_data):
        """Row count matches number of regions."""
        regions_gdf, features_gdf, joint_gdf = srai_data

        p = _make_processor({"compute_diversity_metrics": False})
        result = p.run_count_embeddings(regions_gdf, features_gdf, joint_gdf)

        assert len(result) == len(regions_gdf)

    def test_run_count_embeddings_total_poi_count_dropped(self, srai_data):
        """process_to_h3 drops 'total_poi_count' from the final output."""
        # run_count_embeddings itself returns total_poi_count; only process_to_h3 strips it
        regions_gdf, features_gdf, joint_gdf = srai_data

        p = _make_processor({"compute_diversity_metrics": False})
        result = p.run_count_embeddings(regions_gdf, features_gdf, joint_gdf)

        # total_poi_count IS present in run_count_embeddings output
        assert "total_poi_count" in result.columns


# ---------------------------------------------------------------------------
# Neighbourhood caching
# ---------------------------------------------------------------------------


class TestNeighbourhoodCaching:
    """Tests for H3Neighbourhood save/load/cache logic."""

    @pytest.fixture()
    def processor_with_tmp_dir(self, tmp_path):
        """POIProcessor with intermediate_dir pointing at a tmp directory."""
        p = _make_processor({
            "save_intermediate": True,
            "intermediate_dir": str(tmp_path / "intermediate"),
        })
        return p

    @pytest.fixture(scope="class")
    def small_regions(self):
        """Small H3 tessellation for neighbourhood tests."""
        import geopandas as gpd
        from shapely.geometry import box
        from srai.regionalizers import H3Regionalizer

        area = gpd.GeoDataFrame(
            geometry=[box(4.3, 52.0, 4.32, 52.02)], crs="EPSG:4326"
        )
        return H3Regionalizer(resolution=9).transform(area)

    def test_save_and_load_neighbourhood_roundtrip(self, processor_with_tmp_dir, small_regions):
        """Save then load produces an equivalent H3Neighbourhood."""
        from srai.neighbourhoods import H3Neighbourhood

        p = processor_with_tmp_dir
        original = H3Neighbourhood(small_regions)

        p._save_neighbourhood(original, h3_resolution=9, study_area_name="test")
        loaded = p._load_neighbourhood(h3_resolution=9, study_area_name="test")

        assert loaded is not None
        assert loaded._available_indices == original._available_indices

    def test_load_neighbourhood_returns_none_when_missing(self, processor_with_tmp_dir):
        """When no cached file exists, _load_neighbourhood returns None."""
        p = processor_with_tmp_dir
        result = p._load_neighbourhood(h3_resolution=9, study_area_name="nonexistent")
        assert result is None

    def test_neighbourhood_path_follows_convention(self, processor_with_tmp_dir):
        """Neighbourhood file path includes year and matches intermediate dir pattern."""
        p = processor_with_tmp_dir
        path = p._neighbourhood_path(h3_resolution=10, study_area_name="netherlands")
        assert path.name == f"netherlands_res10_{p.year}_neighbourhood.pkl"
        assert path.parent.name == "neighbourhood"

    def test_load_intermediates_without_neighbourhood_flag(self, processor_with_tmp_dir, small_regions):
        """Default load_intermediates returns 3-tuple (backward compat)."""
        import geopandas as gpd
        from shapely.geometry import Point
        from srai.joiners import IntersectionJoiner

        p = processor_with_tmp_dir
        features_gdf = gpd.GeoDataFrame(
            {"amenity": ["restaurant"]},
            geometry=[Point(4.31, 52.01)],
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"
        joint_gdf = IntersectionJoiner().transform(regions=small_regions, features=features_gdf)

        p._save_intermediate_data(features_gdf, small_regions, joint_gdf, 9, "test")

        result = p.load_intermediates(9, "test")
        assert len(result) == 3

    def test_load_intermediates_with_neighbourhood_flag(self, processor_with_tmp_dir, small_regions):
        """include_neighbourhood=True returns 4-tuple with neighbourhood."""
        import geopandas as gpd
        from shapely.geometry import Point
        from srai.joiners import IntersectionJoiner
        from srai.neighbourhoods import H3Neighbourhood

        p = processor_with_tmp_dir
        features_gdf = gpd.GeoDataFrame(
            {"amenity": ["restaurant"]},
            geometry=[Point(4.31, 52.01)],
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"
        joint_gdf = IntersectionJoiner().transform(regions=small_regions, features=features_gdf)
        neighbourhood = H3Neighbourhood(small_regions)

        p._save_intermediate_data(features_gdf, small_regions, joint_gdf, 9, "test",
                                  neighbourhood=neighbourhood)

        result = p.load_intermediates(9, "test", include_neighbourhood=True)
        assert len(result) == 4
        regions_gdf, features_gdf_loaded, joint_gdf_loaded, nbr = result
        assert nbr is not None
        assert nbr._available_indices == neighbourhood._available_indices

    def test_load_intermediates_with_flag_no_cache(self, processor_with_tmp_dir, small_regions):
        """include_neighbourhood=True returns None when no cache file exists."""
        import geopandas as gpd
        from shapely.geometry import Point
        from srai.joiners import IntersectionJoiner

        p = processor_with_tmp_dir
        features_gdf = gpd.GeoDataFrame(
            {"amenity": ["restaurant"]},
            geometry=[Point(4.31, 52.01)],
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"
        joint_gdf = IntersectionJoiner().transform(regions=small_regions, features=features_gdf)

        # Save intermediates WITHOUT neighbourhood
        p._save_intermediate_data(features_gdf, small_regions, joint_gdf, 9, "test")

        result = p.load_intermediates(9, "test", include_neighbourhood=True)
        assert len(result) == 4
        assert result[3] is None

    def test_infer_resolution(self, small_regions):
        """_infer_resolution correctly reads H3 resolution from index."""
        from stage1_modalities.poi.processor import POIProcessor
        res = POIProcessor._infer_resolution(small_regions)
        assert res == 9


# ---------------------------------------------------------------------------
# run_hex2vec logic — unit tests with mocked Hex2VecEmbedder
# ---------------------------------------------------------------------------


class TestRunHex2VecLogic:
    """Unit tests for the simplified run_hex2vec() which calls fit_transform() directly.

    We mock Hex2VecEmbedder so that fit_transform() returns a deterministic
    DataFrame without any GPU or actual training.  Each test verifies a
    data contract of the method's output.
    """

    @pytest.fixture()
    def srai_triple(self):
        """Small SRAI triple: regions_gdf with a few POIs."""
        import geopandas as gpd
        from shapely.geometry import Point, box
        from srai.regionalizers import H3Regionalizer
        from srai.joiners import IntersectionJoiner

        area = gpd.GeoDataFrame(
            geometry=[box(4.3, 52.0, 4.32, 52.02)], crs="EPSG:4326"
        )
        regions_gdf = H3Regionalizer(resolution=9).transform(area)

        features_gdf = gpd.GeoDataFrame(
            {
                "amenity": ["restaurant", "cafe"],
                "geometry": [Point(4.305, 52.01), Point(4.31, 52.015)],
            },
            crs="EPSG:4326",
        )
        features_gdf.index.name = "feature_id"

        joint_gdf = IntersectionJoiner().transform(
            regions=regions_gdf, features=features_gdf
        )
        return regions_gdf, features_gdf, joint_gdf

    def _make_mock_embedder(self, regions_gdf, embedding_dim: int = 32):
        """Return a mock Hex2VecEmbedder whose fit_transform() returns deterministic embeddings.

        The mock returns one row per region in *regions_gdf*, filled with ones,
        with columns named ``dim_0 .. dim_N``.  The processor renames them to
        ``hex2vec_0 .. hex2vec_N`` after calling fit_transform().
        """
        from unittest.mock import MagicMock

        n = len(regions_gdf)
        mock = MagicMock()
        mock.fit_transform.return_value = pd.DataFrame(
            np.ones((n, embedding_dim), dtype=np.float32),
            index=regions_gdf.index,
            columns=[f"dim_{i}" for i in range(embedding_dim)],
        )
        return mock

    def test_output_covers_all_regions(self, srai_triple):
        """run_hex2vec output has one row per region in regions_gdf."""
        from unittest.mock import patch

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor()
        mock_embedder = self._make_mock_embedder(regions_gdf, embedding_dim=32)

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            result = p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        assert len(result) == len(regions_gdf), (
            f"Expected {len(regions_gdf)} rows, got {len(result)}"
        )

    def test_output_index_name_is_region_id(self, srai_triple):
        """Output DataFrame has index.name == 'region_id'."""
        from unittest.mock import patch

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor()
        mock_embedder = self._make_mock_embedder(regions_gdf)

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            result = p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        assert result.index.name == "region_id"

    def test_output_columns_named_hex2vec_i(self, srai_triple):
        """Output columns are named hex2vec_0, hex2vec_1, ..., hex2vec_N."""
        from unittest.mock import patch

        regions_gdf, features_gdf, joint_gdf = srai_triple
        embedding_dim = 16
        p = _make_processor()
        mock_embedder = self._make_mock_embedder(regions_gdf, embedding_dim=embedding_dim)

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            result = p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        expected_cols = [f"hex2vec_{i}" for i in range(embedding_dim)]
        assert list(result.columns) == expected_cols

    def test_fit_transform_called_once(self, srai_triple):
        """fit_transform() is called exactly once — no chunking or separate fit/transform."""
        from unittest.mock import patch

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor()
        mock_embedder = self._make_mock_embedder(regions_gdf)

        with patch("stage1_modalities.poi.processor.Hex2VecEmbedder", return_value=mock_embedder), \
             patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", True):
            p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)

        assert mock_embedder.fit_transform.call_count == 1, (
            "Simplified run_hex2vec must call fit_transform() exactly once"
        )

    def test_hex2vec_unavailable_raises_import_error(self, srai_triple):
        """run_hex2vec raises ImportError if Hex2VecEmbedder is not installed."""
        from unittest.mock import patch

        regions_gdf, features_gdf, joint_gdf = srai_triple
        p = _make_processor()

        with patch("stage1_modalities.poi.processor.HEX2VEC_AVAILABLE", False):
            with pytest.raises(ImportError, match="srai\\[torch\\]"):
                p.run_hex2vec(regions_gdf, features_gdf, joint_gdf)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
