#!/usr/bin/env python3
"""
=============================================================================
Austin Dibble
University of Glasgow
2026

neurofm/io.py

Input resolution, preprocessing, and output writing.
Accepts a single file, directory, or CSV with an 'input' column.

Output modes
------------
flat     All outputs written to a single flat directory. Default. 
         Note that one subdirectory is created by default: individuals.
         This contains all individual-image outputs, separate from the 
         summary files in the output root.
mirror   Output directory tree mirrors the input directory structure.
summary  Summary CSV and aggregate latent .npy only - no per-file outputs.
         Useful when disk space is a concern or only aggregate results needed.
         Individual-level directory by default follows 'flat'.

Caching
-------
Per-file brain_health outputs double as a cache. If an output file already
exists and --overwrite is not set, the file is loaded from disk and included
in the final aggregate rather than re-running inference. Logged at DEBUG level.
=============================================================================
"""
from __future__ import annotations

import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from nibabel.processing import conform as nibabel_conform
from scipy import stats

from .custom_conform import pad_orient_conform
from .model import _BRAIN_HEALTH_INTERNAL, BRAIN_HEALTH_KEYS  # noqa: F401

SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")
OUTPUT_MODES = ("flat", "mirror", "summary")

# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_inputs(source: str, input_col:str = None) -> List[str]:
    """
    Resolve a user-provided source to a flat list of NIfTI file paths.

    Parameters
    ----------
    source : str
        Path to a single NIfTI file, a directory, or a CSV file with an
        'input' column containing NIfTI paths.

    Returns
    -------
    list[str]
        Resolved and validated list of absolute file paths.
    """
    source = os.path.expanduser(source)
    source = Path(source)

    if source.name.endswith(".csv"):
        paths = _resolve_csv(source)
    elif source.is_dir():
        paths = _resolve_directory(source)
    elif source.is_file():
        paths = _resolve_single(source)
    else:
        raise FileNotFoundError(f"Input not found: {source}")

    if not paths:
        raise ValueError(f"No supported NIfTI files found in: {source}")

    logger.info(f"Resolved {len(paths)} input file(s).")
    return paths


def _resolve_single(path) -> List[str]:
    """Give us the absolute path to the given input path as an array, 
    to fit with the expected formatting.
    """
    path = Path(path)
    if not any(path.name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise ValueError(
            f"Unsupported file type: {path}. Expected one of {SUPPORTED_EXTENSIONS}."
        )
    return [Path(os.path.abspath(path))]


def _resolve_directory(directory: str) -> List[str]:
    """Get all the relevant .nii.gz files in the directory."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(directory, "**", f"*{ext}")
        paths.extend(glob(pattern, recursive=True))
    seen = set()
    unique = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            unique.append(Path(os.path.abspath(p)))
    return unique


def _resolve_csv(csv_path, input_col:str = "input") -> list:
    """Get the paths from the given .csv file. Returns the discovered paths as a list."""
    df = pd.read_csv(csv_path)
    if input_col not in df.columns:
        raise ValueError(
            f"CSV must contain an '{input_col}' column. Found columns: {list(df.columns)}"
        )
    paths = df[input_col].dropna().tolist()
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        logger.warning(
            f"{len(missing)} path(s) in CSV do not exist and will be skipped:\n"
            + "\n".join(missing[:5])
            + ("\n  ..." if len(missing) > 5 else "")
        )
        paths = [p for p in paths if os.path.isfile(p)]
    return [Path(os.path.abspath(p)) for p in paths]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

TARGET_SHAPE = (256, 256, 256) # Model input shape is fixed 256^3, so we pad
TARGET_ZOOMS = (1.0, 1.0, 1.0) # We want the MRI in 1mm iso
TARGET_ORIENTATION = "LIA" # all brains should be conformed to LIA, which was the training space

def load_and_preprocess(path: str, custom_preproc_fn = None) -> np.ndarray:
    """
    Load a NIfTI file, apply orientation correction and resampling,
    and return a preprocessed numpy array ready for model input.

    Parameters
    ----------
    path : str
        Path to a skull-stripped T1w NIfTI file.

    Returns
    -------
    np.ndarray
        Preprocessed volume, shape (1, X, Y, Z, 1) - batch and channel dims
        included for direct model input.
    """
    img = nib.load(path)
    return preprocess_nifti(img, custom_preproc_fn = custom_preproc_fn)

def preprocess_nifti(img, custom_preproc_fn=None):
    img = _resample(img)   # reorients internally if resampling occurs
    img = _reorient(img)   # no-op if already correct; fallback if resample short-circuited
    data = img.get_fdata(dtype=np.float32)
    if custom_preproc_fn is not None:
        data = custom_preproc_fn(data)
    data = _normalize(data)
    return data[np.newaxis, ..., np.newaxis]


def _reorient(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient the given nifti image into the correct orientation space, as needed.
    Image should already be in 256^3 1mm iso space by this point.

    Args:
        img (nib.Nifti1Image): Input volume

    Returns:
        nib.Nifti1Image: Reoriented volume.
    """
    current_orientation = "".join(nib.aff2axcodes(img.affine))
    if current_orientation == TARGET_ORIENTATION:
        return img

    logger.debug(f"Reorienting from '{current_orientation}' to '{TARGET_ORIENTATION}'.")
    try:
        reoriented, _, _, _ = pad_orient_conform(
            img,
            out_shape=img.shape[:3],
            orientation=TARGET_ORIENTATION,
            pad=False,
        )
        return reoriented
    except Exception as e:
        logger.warning(
            f"Reorientation from '{current_orientation}' to '{TARGET_ORIENTATION}' "
            f"failed ({e}). Proceeding with original orientation - results may be degraded."
        )
        return img

def _resample(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Conform image to TARGET_SHAPE and TARGET_ZOOMS.

    Three cases are handled in order of cost:

    1. Correct shape and zooms - no-op, return immediately.
    2. Correct zooms, wrong shape - pad and reorient only (fast path, matches
       training which assumed pre-conformant inputs from FreeSurfer).
    3. Wrong zooms - full cubic resample via nibabel_conform (slow path). Emits
       a one-time UserWarning recommending upstream pre-processing with
       `mri_convert --conform --resample_type cubic` for batch use.

    Reorientation to TARGET_ORIENTATION is handled as part of both the fast
    and slow paths here. _reorient is still called after this function in
    preprocess_nifti as a fallback for any residual orientation mismatch.

    Parameters
    ----------
    img : nib.Nifti1Image
        Input volume in any shape, resolution, or orientation.

    Returns
    -------
    nib.Nifti1Image
        Volume conformed to TARGET_SHAPE and TARGET_ZOOMS where possible.
        On failure, returns the original image and logs a warning.
    """
    current_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    current_shape = img.shape[:3]

    if current_zooms == TARGET_ZOOMS and current_shape == TARGET_SHAPE:
        return img

    logger.debug(
        f"Conforming: shape {current_shape} -> {TARGET_SHAPE}, "
        f"zooms {current_zooms} -> {TARGET_ZOOMS}."
    )

    if current_zooms != TARGET_ZOOMS:
        # Full resample required - slow path
        warnings.warn(
            "Input volume requires resampling due to incorrect voxel size. "
            "This is slower and may reduce accuracy. For batch analyses, consider "
            "pre-processing with: mri_convert --conform --resample_type cubic <input> <output>",
            UserWarning,
            stacklevel=2,
        )
        try:
            return nibabel_conform(
                img,
                out_shape=TARGET_SHAPE,
                voxel_size=TARGET_ZOOMS,
                order=3,
                orientation=TARGET_ORIENTATION,
            )
        except Exception as e:
            logger.warning(
                f"Resampling failed ({e}). Proceeding with original resolution - "
                f"results may be degraded."
            )
            return img
    else:
        # Zooms are correct, only padding/reorientation needed - fast path
        try:
            reoriented, _, _, _ = pad_orient_conform(
                img,
                out_shape=TARGET_SHAPE,
                orientation=TARGET_ORIENTATION,
                pad=True,
            )
            return reoriented
        except Exception as e:
            logger.warning(
                f"Padding/reorientation failed ({e}). Proceeding with original shape - "
                f"results may be degraded."
            )
            return img


def _normalize(data: np.ndarray) -> np.ndarray:
    """z-score for our volumes. Same as during training"""
    return stats.zscore(data, axis=None)


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def get_output_dir(
    input_path,
    output_root,
    output_mode: str,
    input_root: Optional[str] = None,
    individuals_dir: bool = True
) -> str:
    """
    Resolve the output directory for a given input file.

    Note that for output_mode of 'flat' or 'summary', the path 'individuals' is added by default.
    This is to group individual-level outputs within a directory, while the summary files will
    be saved in the output root. This can be disabled with individuals_dir = False.

    Parameters
    ----------
    input_path : str
        Absolute path to the input NIfTI file.
    output_root : str
        Root output directory specified by the user.
    output_mode : str
        One of 'flat', 'mirror', 'summary'.
    input_root : str | None
        For mirror mode - the common root of all input paths, used to
        reconstruct relative directory structure. If None, uses the
        input file's parent directory.

    Returns
    -------
    str
        Directory where outputs for this input file should be written.
    """
    if output_mode in ("flat", "summary"):
        if individuals_dir:
            return Path(output_root) / 'individuals'
        else:
            return Path(output_root)

    # mirror mode: reconstruct relative path under output_root
    if input_root is None:
        input_root = Path(input_path).parent

    try:
        rel = Path(input_path).parent.relative_to(input_root)
        return Path(output_root) / rel
    except ValueError:
        # relative_to raises ValueError when path is not under input_root (e.g. different drives on Windows)
        logger.warning(
            "Could not compute relative path for mirror mode "
            f"(input: {input_path}, root: {input_root}). Falling back to flat."
        )
        return Path(output_root)


def get_expected_output_path(
    input_path,
    output_root,
    output_mode,
    input_root:Optional[str] = None,
) -> str:
    """
    Return the expected path of the brain_health .npy for a given input.
    Used by the CLI to check whether cached outputs exist before running inference.
    """
    out_dir = get_output_dir(input_path, output_root, output_mode, input_root)
    stem = _get_stem(input_path)
    return out_dir / f"{stem}_brain_health.npy"


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cached_result(
    input_path,
    output_root,
    output_mode,
    requested_outputs: List[str],
    input_root: Optional[str] = None,
):
    """
    Attempt to load previously saved outputs for a given input file.

    Returns a results dict (same structure as NeuroFM.predict()) if all
    requested outputs are found on disk, otherwise returns None.
    """
    out_dir = get_output_dir(input_path, output_root, output_mode, input_root)

    stem = _get_stem(input_path)
    result = {}

    if "brain_health" in requested_outputs:
        bh_path = out_dir / f"{stem}_brain_health.npy"
        if not bh_path.is_file():
            return None
        result["brain_health"] = np.load(bh_path)
        logger.debug(f"Cache hit - loaded brain_health from {bh_path}")

    if "latent" in requested_outputs:
        lat_path = out_dir / f"{stem}_latent.npy"
        if not lat_path.is_file():
            return None
        result["latent"] = np.load(lat_path)
        logger.debug(f"Cache hit - loaded latent from {lat_path}")

    return result if result else None


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def save_outputs(
    results: dict,
    input_path,
    output_root,
    requested_outputs: List[str],
    output_mode: str = "flat",
    input_root = None,
) -> None:
    """
    Save per-file model outputs to disk.

    Skipped entirely in 'summary' mode - aggregate outputs are written
    by save_batch_summary() instead.

    Parameters
    ----------
    results : dict
        Output of NeuroFM.predict(), keyed by output name.
    input_path : str
        Original input file path (used to derive output filename).
    output_root : str
        Root output directory.
    requested_outputs : list[str]
        Which outputs to save, e.g. ["brain_health", "latent"].
    output_mode : str
        One of 'flat', 'mirror', 'summary'.
    input_root : str | None
        For mirror mode - common root of all input paths.
    """
    if output_mode == "summary":
        return

    out_dir = get_output_dir(input_path, output_root, output_mode, input_root)
    out_dir.mkdir(exist_ok=True, parents=True)
    stem = _get_stem(input_path)

    if "brain_health" in requested_outputs and "brain_health" in results:
        npy_path = out_dir / f"{stem}_brain_health.npy"
        np.save(npy_path, results["brain_health"])
        logger.debug(f"Saved brain_health -> {npy_path}")

    if "latent" in requested_outputs and "latent" in results:
        lat_path = out_dir / f"{stem}_latent.npy"
        np.save(lat_path, results["latent"])
        logger.debug(f"Saved latent -> {lat_path}")

def save_batch_summary(
    all_results: List[dict],
    input_paths: list,
    output_dir,
    requested_outputs: List[str],
    input_col:str = "input"
) -> None:
    """
    Write aggregate outputs across all processed scans:
      - results_summary.csv        brain_health values for all scans
      - latent_embeddings.npy      shape (N, D), if latent was requested
      - latent_embeddings_index.csv maps rows of latent_embeddings.npy to input paths

    Parameters
    ----------
    all_results : list[dict]
        One result dict per input path. None entries (failed scans) are skipped.
    input_paths : list[str]
        Input paths in the same order as all_results.
    output_dir : str
        Directory to write aggregate outputs into.
    requested_outputs : list[str]
        Which outputs were requested.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    bh_rows = []
    latent_rows = []
    latent_input_paths = []

    for path, result in zip(input_paths, all_results):
        if result is None:
            continue

        if 'brain_health' in result and result['brain_health'] is None \
            or 'latent' in result and result['latent'] is None:
            continue

        if "brain_health" in requested_outputs and "brain_health" in result:
            row = {input_col: path}
            row.update(dict(zip(BRAIN_HEALTH_KEYS, result["brain_health"].tolist())))
            bh_rows.append(row)

        if "latent" in requested_outputs and "latent" in result:
            latent_rows.append(result["latent"])
            latent_input_paths.append(path)

    if bh_rows:
        summary_path = output_dir / "results_summary.csv"
        df = pd.DataFrame(bh_rows)
        df.to_csv(summary_path, index=False)
        logger.info(f"Summary CSV written -> {summary_path} ({len(bh_rows)} scan(s))")

    if latent_rows:
        latent_array = np.stack(latent_rows, axis=0)  # (N, D)
        latent_path = output_dir / "latent_embeddings.npy"
        np.save(latent_path, latent_array)

        # Index CSV so users can map rows back to input paths
        index_path = output_dir / "latent_embeddings_index.csv"
        pd.DataFrame({input_col: latent_input_paths}).to_csv(index_path, index=False)

        logger.info(
            f"Latent embeddings written -> {latent_path} "
            f"(shape: {latent_array.shape}), index -> {index_path}"
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_stem(path) -> str:
    """Strip directory and NIfTI extensions to get a clean filename stem."""
    base = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base

def infer_input_root(input_paths: list) -> Path:
    """
    For mirror mode - find the common directory prefix across all input paths.
    Returns None if inputs come from completely different trees.
    """
    if not input_paths:
        return None
    dirs = [Path(p).parent for p in input_paths]
    return Path(os.path.commonpath(dirs)) or None