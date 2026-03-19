#!/usr/bin/env python3
"""
=============================================================================
Austin Dibble
University of Glasgow
2026

neurofm/weights.py

Variant registry and automatic weight download from HuggingFace.
Zenodo is the canonical archive for citation; HuggingFace is used
for programmatic access.
=============================================================================
"""
from __future__ import annotations

import os
from typing import Optional

from huggingface_hub import hf_hub_download
from loguru import logger

HF_REPO_ID = "rocknroll87q/NeuroFM"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/NeuroFM")
DEFAULT_VARIANT = "neurofm-s"

DEFAULT_STD_CONSTS = {
    'PatientAge': {'mean': 63.0270771713598, 'std': 10.9996084993436},
    'brain_volume': {'mean': 1469707.60713924, 'std': 187472.342357356},
    'ventricular_volume': {'mean': 89309.9694533562, 'std': 47429.5375731711}
}

# Registry of all available model variants.
# Keys are the user-facing variant names (passed via --model).
VARIANTS = {
    "neurofm-s": {
        "filename": "neurofm-s.h5",
        "savedmodel": "neurofm-l_savedmodel.tar.gz",
        "params": "484k",
        "latent_dim": 161,
        "description": "Small - fast CPU inference, suitable for large cohorts.",
    },
    "neurofm-m": {
        "filename": "neurofm-m.h5",
        "savedmodel": "neurofm-l_savedmodel.tar.gz",
        "params": "6.5M",
        "latent_dim": 256,
        "description": "Medium - balanced representation and speed.",
    },
    "neurofm-l": {
        "filename": "neurofm-l.h5",
        "savedmodel": "neurofm-l_savedmodel.tar.gz",
        "params": "10.8M",
        "latent_dim": 512,
        "description": "Large - maximum representation, GPU recommended.",
    },
}

def get_weights_path(
    variant: str = DEFAULT_VARIANT,
    cache_dir: str = DEFAULT_CACHE_DIR,
    local_path: Optional[str] = None,
) -> str:
    """
    Resolve the path to model weights for a given variant.

    If `local_path` is provided, it is used directly (no download).
    Otherwise, weights are downloaded from HuggingFace if not already cached.

    Parameters
    ----------
    variant : str
        One of the keys in VARIANTS (e.g. 'neurofm-s').
    cache_dir : str
        Local directory for caching downloaded weights.
    local_path : str | None
        If provided, skip download and use this path directly.

    Returns
    -------
    str
        Absolute path to the weights .h5 file.
    """
    # if user-dir symbol ~ is used, we need to expand it. Should be harmless to always ensure.
    cache_dir = os.path.expanduser(cache_dir)

    if local_path is not None:
        local_path = os.path.expanduser(local_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Weights file not found: {local_path}")
        logger.info(f"Using local weights: {local_path}")
        return local_path

    _validate_variant(variant)
    filename = VARIANTS[variant]["filename"]

    logger.info(
        f"Loading weights for {variant} ({VARIANTS[variant]['params']} params). "
        f"Downloading from HuggingFace if not cached..."
    )

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        cache_dir=cache_dir,
    )

    logger.info(f"Weights ready: {path}")
    return path


def _validate_variant(variant: str) -> None:
    """Ensure the selected variant exists"""
    if variant not in VARIANTS:
        valid = ", ".join(VARIANTS.keys())
        raise ValueError(
            f"Unknown model variant '{variant}'. Valid options are: {valid}."
        )


def list_variants() -> None:
    """Print a summary of available model variants to stdout."""
    print(f"\nAvailable NeuroFM variants (HuggingFace repo: {HF_REPO_ID}):\n")
    header = f"  {'Variant':<14} {'Params':<10} {'Latent dim':<14} Description"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, info in VARIANTS.items():
        default_tag = " (default)" if name == DEFAULT_VARIANT else ""
        print(
            f"  {name + default_tag:<14} {info['params']:<10} "
            f"{info['latent_dim']:<14} {info['description']}"
        )
    print()