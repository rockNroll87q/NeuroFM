"""
neurofm/inference.py

Core inference logic. The NeuroFM class is the primary user-facing API.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from loguru import logger

from .io import load_and_preprocess
from .model import _BRAIN_HEALTH_INTERNAL, get_mid_layer, load_neurofm
from .weights import DEFAULT_STD_CONSTS, DEFAULT_VARIANT, VARIANTS, get_weights_path

VALID_OUTPUTS = {"brain_health", "latent"}
LATENT_LAYER_NAME = "multihead_output"

class NeuroFM:
    """
    NeuroFM: Foundation model for individualized brain health estimation.

    Loads a pretrained model variant and runs inference on T1w MRI scans,
    producing brain health predictions and optionally latent embeddings.

    Parameters
    ----------
    variant : str
        Model variant to load. One of 'neurofm-s', 'neurofm-m', 'neurofm-l'.
        Defaults to 'neurofm-s'.
    device : str
        Device to run inference on. One of 'auto', 'cpu', 'gpu'.
        'auto' uses GPU if available, falls back to CPU.
    weights : str | None
        Path to a local weights .h5 file. If None, weights are downloaded
        automatically from HuggingFace.
    cache_dir : str
        Directory for caching downloaded weights.

    Examples
    --------
    >>> from neurofm import NeuroFM
    >>> model = NeuroFM(variant="neurofm-s", device="auto")
    >>> results = model.predict("subject_01_T1w.nii.gz")
    >>> results["brain_health"]  # np.ndarray, shape (4,)

    >>> # With latent features
    >>> results = model.predict("scan.nii.gz", outputs=["brain_health", "latent"])
    >>> results["latent"]  # np.ndarray, shape (D,)
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        device: str = "auto",
        weights: str | None = None,
        cache_dir: str = "~/.cache/NeuroFM",
        standardization_constants: dict = DEFAULT_STD_CONSTS,
        custom_preproc_fn = None,
        custom_postproc_fn = None,
    ):
        self.variant = variant
        self.device = device
        self._latent_dim = VARIANTS[variant]["latent_dim"]
        self.std_consts = DEFAULT_STD_CONSTS
        self.custom_preproc_fn = custom_preproc_fn
        self.custom_postproc_fn = custom_postproc_fn

        _configure_device(device)

        weights_path = get_weights_path(
            variant=variant,
            cache_dir=cache_dir,
            local_path=weights,
        )
        self._model = load_neurofm(weights_path, variant=variant)
        self._latent_model = None  # built lazily if latent output is requested

        logger.info(
            f"NeuroFM ready — variant: {variant}, device: {_active_device()}"
        )

    def predict(
        self,
        input_path: str,
        outputs: list[str] | None = None,
    ) -> dict:
        """
        Run inference on a single NIfTI file.

        Parameters
        ----------
        input_path : str
            Path to a skull-stripped, 1mm isotropic T1w NIfTI file.
        outputs : list[str] | None
            Which outputs to compute. Any subset of ['brain_health', 'latent'].
            Defaults to ['brain_health'].

        Returns
        -------
        dict
            Keys are the requested output names. Values are np.ndarrays.
            'brain_health' shape: (4,) — [brain_age, sex, ventricle_volume, brain_volume]
            'latent' shape: (D,) — latent embedding dimension for the variant
        """
        if outputs is None:
            outputs = ["brain_health"]

        _validate_outputs(outputs)

        volume = load_and_preprocess(input_path, self.custom_preproc_fn)

        if self.custom_postproc_fn is not None:
            volume = self.custom_postproc_fn(volume)

        results = {}

        if "brain_health" in outputs:
            try:
                results["brain_health"] = self.predict_brain_health(volume)
            except Exception as e:
                logger.warning(f"Failed to process brain_health for {input_path}: {e}")
                results["brain_health"] = None

        if "latent" in outputs:
            try:
                results["latent"] = self.predict_latent(volume)
            except Exception as e:
                logger.warning(f"Failed to process latent for {input_path}: {e}")
                results["latent"] = None

        return results

    def predict_batch(
        self,
        input_paths: list[str],
        outputs: list[str] | None = None,
    ) -> list[dict]:
        """
        Run inference on a list of NIfTI files sequentially.

        Returns a list of result dicts in the same order as input_paths.
        Failed scans are returned as None with a logged warning rather than
        raising, so a single bad file doesn't abort a long cohort run.

        Parameters
        ----------
        input_paths : list[str]
            List of paths to NIfTI files.
        outputs : list[str] | None
            Which outputs to compute. Defaults to ['brain_health'].

        Returns
        -------
        list[dict | None]
        """
        all_results = []
        n = len(input_paths)

        for i, path in enumerate(input_paths):
            logger.info(f"[{i + 1}/{n}] Processing: {path}")
            try:
                result = self.predict(path, outputs=outputs)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                result = None
            all_results.append(result)

        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def predict_brain_health(self, volume: np.ndarray) -> np.ndarray:
        preds = self._model.predict(volume, verbose=0)
        # preds shape: (1, 4) — squeeze batch dim

        # initialize before populating
        preds_processed = [0] * len(_BRAIN_HEALTH_INTERNAL)

        # each item in the order we set in the config
        for output_name in _BRAIN_HEALTH_INTERNAL:
            if output_name == 'PatientSex':
                continue
            out_idx = _BRAIN_HEALTH_INTERNAL.index(output_name)
            # get rid of extra dim and then de-standardize
            pred_raw = np.squeeze(preds[out_idx])
            pred_scaled = self._unstandardize(pred_raw, output_name)
            preds_processed[out_idx] = pred_scaled

        # sex is the outlier, it just needs arg-maxed. We will
        # leave the class as a binary int/float
        sex_idx = _BRAIN_HEALTH_INTERNAL.index("PatientSex")
        sex_pred = np.argmax(preds[sex_idx])

        preds_processed[sex_idx] = sex_pred
        return np.squeeze(preds_processed).astype(np.float32)

    def predict_latent(self, volume: np.ndarray) -> np.ndarray:
        if self._latent_model is None:
            self._latent_model = get_mid_layer(
                self._model, LATENT_LAYER_NAME
            )
        embedding = self._latent_model.predict(volume, verbose=False)
        return np.squeeze(embedding).astype(np.float32)

    def _unstandardize(self, value, var_name):
        mu = self.std_consts[var_name]['mean']
        std = self.std_consts[var_name]['std']
        return (value * std) + mu

# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

def _configure_device(device: str) -> None:
    """Configure TensorFlow device visibility before model load."""
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        logger.info("Device set to CPU (GPU disabled).")
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logger.warning(
                "No GPU detected. Falling back to CPU. "
                "Check your CUDA/cuDNN installation or use --device cpu explicitly."
            )
        else:
            logger.info(f"GPU detected: {[g.name for g in gpus]}")
    elif device == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Auto-selected GPU: {[g.name for g in gpus]}")
        else:
            logger.info("No GPU detected. Running on CPU.")
    else:
        raise ValueError(f"Unknown device '{device}'. Use 'auto', 'cpu', or 'gpu'.")


def _active_device() -> str:
    gpus = tf.config.get_visible_devices("GPU")
    return f"GPU ({gpus[0].name})" if gpus else "CPU"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_outputs(outputs: list[str]) -> None:
    invalid = set(outputs) - VALID_OUTPUTS
    if invalid:
        raise ValueError(
            f"Unknown output(s): {invalid}. Valid options are: {VALID_OUTPUTS}."
        )