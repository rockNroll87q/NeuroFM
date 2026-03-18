"""
tests/test_neurofm.py

Sanity checks for NeuroFM. Covers input resolution, preprocessing,
model construction, and inference output shapes.

These tests are intentionally lightweight — no pretrained weights are
required. The inference tests build each variant with random weights
and verify output shapes and types are correct.

Run with:
    python -m unittest tests/test_neurofm.py -v
"""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurofm.io import (
    _get_stem,
    _normalize,
    get_expected_output_path,
    infer_input_root,
    load_cached_result,
    resolve_inputs,
    save_outputs,
)
from neurofm.model import (
    _BRAIN_HEALTH_INTERNAL,
    BRAIN_HEALTH_KEYS,
    VARIANT_CONFIGS,
    NetworkConfig,
    build_model,
)
from neurofm.weights import DEFAULT_VARIANT, VARIANTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(shape=(16, 16, 16), zooms=(1.0, 1.0, 1.0)):
    """Create a minimal synthetic NIfTI image for testing."""
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.diag(list(zooms) + [1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(zooms)
    return img


def _save_nifti(img, path):
    nib.save(img, path)
    return path


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

class TestResolveInputs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    def _make_scan(self, name="scan.nii.gz"):
        path = os.path.join(self.tmp, name)
        _save_nifti(_make_nifti(), path)
        return path

    def test_single_file(self):
        path = self._make_scan()
        result = resolve_inputs(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0]), os.path.abspath(path))

    def test_directory(self):
        for name in ("a.nii.gz", "b.nii.gz", "c.nii.gz"):
            self._make_scan(name)
        result = resolve_inputs(self.tmp)
        self.assertEqual(len(result), 3)

    def test_directory_recursive(self):
        sub = os.path.join(self.tmp, "sub")
        os.makedirs(sub)
        self._make_scan("top.nii.gz")
        _save_nifti(_make_nifti(), os.path.join(sub, "nested.nii.gz"))
        result = resolve_inputs(self.tmp)
        self.assertEqual(len(result), 2)

    def test_csv(self):
        paths = [self._make_scan(f"s{i}.nii.gz") for i in range(3)]
        csv_path = os.path.join(self.tmp, "subjects.csv")
        pd.DataFrame({"input": paths, "subject_id": ["s1", "s2", "s3"]}).to_csv(
            csv_path, index=False
        )
        result = resolve_inputs(csv_path)
        self.assertEqual(len(result), 3)

    def test_csv_missing_input_column(self):
        csv_path = os.path.join(self.tmp, "bad.csv")
        pd.DataFrame({"path": ["/nonexistent.nii.gz"]}).to_csv(csv_path, index=False)
        with self.assertRaises(ValueError):
            resolve_inputs(csv_path)

    def test_csv_skips_missing_files(self):
        real = self._make_scan("real.nii.gz")
        csv_path = os.path.join(self.tmp, "partial.csv")
        pd.DataFrame({"input": [real, "/nonexistent/fake.nii.gz"]}).to_csv(
            csv_path, index=False
        )
        result = resolve_inputs(csv_path)
        self.assertEqual(len(result), 1)

    def test_unsupported_extension(self):
        path = os.path.join(self.tmp, "scan.mgz")
        open(path, "w").close()
        with self.assertRaises(ValueError):
            resolve_inputs(path)

    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            resolve_inputs("/definitely/does/not/exist.nii.gz")

    def test_empty_directory(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            resolve_inputs(empty)


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    def test_normalize_zero_mean_unit_std(self):
        """z-score normalisation should produce zero mean and unit std."""
        data = np.random.rand(16, 16, 16).astype(np.float32) * 1000
        result = _normalize(data)
        self.assertAlmostEqual(float(np.mean(result)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(result)), 1.0, places=4)

    def test_normalize_constant_volume(self):
        """Constant volume has zero std — scipy returns nan, which is expected."""
        data = np.ones((8, 8, 8), dtype=np.float32)
        result = _normalize(data)
        self.assertTrue(np.all(np.isnan(result)))

    def test_get_stem_nii_gz(self):
        self.assertEqual(_get_stem("/data/sub-01_T1w.nii.gz"), "sub-01_T1w")

    def test_get_stem_nii(self):
        self.assertEqual(_get_stem("/data/sub-01_T1w.nii"), "sub-01_T1w")

    def test_get_stem_no_extension(self):
        self.assertEqual(_get_stem("/data/somefile"), "somefile")

    def test_infer_input_root_common(self):
        paths = [
            "/data/study/sub-01/scan.nii.gz",
            "/data/study/sub-02/scan.nii.gz",
        ]
        root = infer_input_root(paths)
        self.assertIsNotNone(root)
        self.assertTrue("/data/study" in str(root))

    def test_infer_input_root_empty(self):
        self.assertIsNone(infer_input_root([]))


# ---------------------------------------------------------------------------
# Model construction and registry
# ---------------------------------------------------------------------------

class TestModelConstruction(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    def test_default_network_config(self):
        """NetworkConfig should instantiate with defaults without error."""
        config = NetworkConfig()
        self.assertEqual(len(config.num_classes), 4)
        self.assertEqual(len(config.predicted_variable), 4)

    def test_all_variant_configs_exist(self):
        for variant in ("neurofm-s", "neurofm-m", "neurofm-l"):
            self.assertIn(variant, VARIANT_CONFIGS)

    def test_brain_health_arrays_same_length(self):
        """Internal and user-facing key lists must be the same length."""
        self.assertEqual(len(_BRAIN_HEALTH_INTERNAL), len(BRAIN_HEALTH_KEYS))

    def test_brain_health_keys_match_num_classes(self):
        """Number of output heads must match brain health key count."""
        config = NetworkConfig()
        self.assertEqual(len(config.num_classes), len(BRAIN_HEALTH_KEYS))

    def test_default_variant_in_weights_registry(self):
        self.assertIn(DEFAULT_VARIANT, VARIANTS)

    def test_all_variant_configs_in_weights_registry(self):
        """Every variant config must have a corresponding weights entry."""
        for variant in VARIANT_CONFIGS:
            self.assertIn(variant, VARIANTS)
            self.assertIn("filename", VARIANTS[variant])
            self.assertIn("latent_dim", VARIANTS[variant])

    def test_latent_dim_matches_neck_layer_size(self):
        """VARIANTS latent_dim should match neck_layer_size in the config."""
        for variant, info in VARIANTS.items():
            if variant in VARIANT_CONFIGS:
                config = VARIANT_CONFIGS[variant]
                self.assertEqual(
                    info["latent_dim"], config.neck_layer_size,
                    msg=f"latent_dim mismatch for {variant}"
                )

    def test_build_small_model(self):
        """Build a tiny model to verify architecture constructs without error."""
        config = NetworkConfig(
            shape=(32, 32, 32),
            num_conv_layers=2,
            num_initial_filter=4,
            num_neck_layers=1,
            neck_layer_size=16,
            final_stage="avgPool",
        )
        model = build_model(config)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.outputs), len(config.num_classes))

    def test_model_has_multihead_output_layer(self):
        """Latent extraction layer must exist with the expected name."""
        config = NetworkConfig(
            shape=(32, 32, 32),
            num_conv_layers=2,
            num_initial_filter=4,
            num_neck_layers=1,
            neck_layer_size=16,
        )
        model = build_model(config)
        layer_names = [layer.name for layer in model.layers]
        self.assertIn("multihead_output", layer_names)


# ---------------------------------------------------------------------------
# Inference output shapes (random weights — no download required)
# ---------------------------------------------------------------------------

class TestInferenceShapes(unittest.TestCase):
    """
    Smoke tests using randomly initialised weights. Validates output
    shapes and types only — not prediction quality.
    """

    @classmethod
    def setUpClass(cls):
        cls.config = NetworkConfig(
            shape=(32, 32, 32),
            num_conv_layers=2,
            num_initial_filter=4,
            num_neck_layers=1,
            neck_layer_size=16,
        )
        cls.model = build_model(cls.config)
        cls.volume = np.random.rand(1, 32, 32, 32, 1).astype(np.float32)
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    def test_predict_returns_list(self):
        preds = self.model.predict(self.volume, verbose=0)
        self.assertIsInstance(preds, list)

    def test_predict_head_count(self):
        preds = self.model.predict(self.volume, verbose=0)
        self.assertEqual(len(preds), len(self.config.num_classes))

    def test_each_head_is_scalar_per_batch(self):
        """Each regression head should produce shape (batch, 1)."""
        preds = self.model.predict(self.volume, verbose=0)
        for i, p in enumerate(preds):
            self.assertEqual(p.shape[0], 1, msg=f"Head {i} batch dim wrong")

    def test_outputs_are_float32(self):
        preds = self.model.predict(self.volume, verbose=0)
        for i, p in enumerate(preds):
            self.assertEqual(p.dtype, np.float32, msg=f"Head {i} dtype wrong")

    def test_latent_extraction_shape(self):
        """Latent sub-model should output (1, neck_layer_size)."""
        import tensorflow as tf
        latent_output = self.model.get_layer("multihead_output").output
        latent_model = tf.keras.Model(
            inputs=self.model.input, outputs=latent_output
        )
        embedding = latent_model.predict(self.volume, verbose=0)
        self.assertEqual(embedding.shape, (1, self.config.neck_layer_size))


# ---------------------------------------------------------------------------
# Output writing and cache
# ---------------------------------------------------------------------------

class TestOutputWritingAndCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.input_path = os.path.join(self.tmp, "sub-01_T1w.nii.gz")
        _save_nifti(_make_nifti(), self.input_path)
        self.fake_result = {
            "brain_health": np.array(
                [65.2, 0.8, 45000.0, 1350000.0], dtype=np.float32
            ),
            "latent": np.random.rand(16).astype(np.float32),
        }
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    def test_save_and_reload_brain_health(self):
        save_outputs(
            self.fake_result, self.input_path, self.tmp,
            ["brain_health"], output_mode="flat",
        )
        npy_path = os.path.join(self.tmp, "individuals", "sub-01_T1w_brain_health.npy")
        self.assertTrue(os.path.isfile(npy_path))
        loaded = np.load(npy_path)
        np.testing.assert_array_almost_equal(
            loaded, self.fake_result["brain_health"]
        )

    def test_save_and_reload_latent(self):
        save_outputs(
            self.fake_result, self.input_path, self.tmp,
            ["brain_health", "latent"], output_mode="flat",
        )
        lat_path = os.path.join(self.tmp, "individuals", "sub-01_T1w_latent.npy")
        self.assertTrue(os.path.isfile(lat_path))
        loaded = np.load(lat_path)
        np.testing.assert_array_almost_equal(
            loaded, self.fake_result["latent"]
        )

    def test_summary_mode_writes_no_per_file_outputs(self):
        save_outputs(
            self.fake_result, self.input_path, self.tmp,
            ["brain_health", "latent"], output_mode="summary",
        )
        self.assertFalse(
            os.path.isfile(os.path.join(self.tmp, "sub-01_T1w_brain_health.npy"))
        )

    def test_cache_hit_returns_result(self):
        """After saving, load_cached_result should return the saved values."""
        save_outputs(
            self.fake_result, self.input_path, self.tmp,
            ["brain_health", "latent"], output_mode="flat",
        )
        cached = load_cached_result(
            self.input_path, self.tmp, "flat", ["brain_health", "latent"],
        )
        self.assertIsNotNone(cached)
        np.testing.assert_array_almost_equal(
            cached["brain_health"], self.fake_result["brain_health"]
        )

    def test_cache_miss_returns_none(self):
        cached = load_cached_result(
            self.input_path, self.tmp, "flat", ["brain_health"],
        )
        self.assertIsNone(cached)

    def test_cache_miss_if_latent_missing(self):
        """brain_health saved but latent not — partial cache should be a miss."""
        save_outputs(
            self.fake_result, self.input_path, self.tmp,
            ["brain_health"], output_mode="flat",
        )
        cached = load_cached_result(
            self.input_path, self.tmp, "flat",
            ["brain_health", "latent"],
        )
        self.assertIsNone(cached)

    def test_get_expected_output_path(self):
        expected = get_expected_output_path(self.input_path, self.tmp, "flat")
        self.assertTrue(expected.name.endswith("sub-01_T1w_brain_health.npy"))


if __name__ == "__main__":
    unittest.main(verbosity=2)