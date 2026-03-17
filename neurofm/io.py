"""
neurofm/io.py

Input resolution, preprocessing, and output writing.
Accepts a single file, directory, or CSV with an 'input' column.
"""
from __future__ import annotations

import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger
from nibabel.processing import conform
from scipy import stats

from .model import _BRAIN_HEALTH_INTERNAL, BRAIN_HEALTH_KEYS

SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def resolve_inputs(source: str) -> list[str]:
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

    if source.endswith(".csv"):
        paths = _resolve_csv(source)
    elif os.path.isdir(source):
        paths = _resolve_directory(source)
    elif os.path.isfile(source):
        paths = _resolve_single(source)
    else:
        raise FileNotFoundError(f"Input not found: {source}")

    if not paths:
        raise ValueError(f"No supported NIfTI files found in: {source}")

    logger.info(f"Resolved {len(paths)} input file(s).")
    return paths


def _resolve_single(path: str) -> list[str]:
    if not any(path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise ValueError(
            f"Unsupported file type: {path}. Expected one of {SUPPORTED_EXTENSIONS}."
        )
    return [os.path.abspath(path)]


def _resolve_directory(directory: str) -> list[str]:
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        # glob doesn't handle .nii.gz with a single *, so handle both
        pattern = os.path.join(directory, "**", f"*{ext}")
        paths.extend(glob(pattern, recursive=True))
    # Deduplicate (a .nii file won't also match .nii.gz, but be safe)
    seen = set()
    unique = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            unique.append(os.path.abspath(p))
    return unique


def _resolve_csv(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if "input" not in df.columns:
        raise ValueError(
            f"CSV must contain an 'input' column. Found columns: {list(df.columns)}"
        )
    paths = df["input"].dropna().tolist()
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        logger.warning(
            f"{len(missing)} path(s) in CSV do not exist and will be skipped:\n"
            + "\n".join(missing[:5])
            + ("\n  ..." if len(missing) > 5 else "")
        )
        paths = [p for p in paths if os.path.isfile(p)]
    return [os.path.abspath(p) for p in paths]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

TARGET_SHAPE = (256, 256, 256)   # MNI152 1mm
TARGET_ZOOMS = (1.0, 1.0, 1.0)  # 1mm isotropic
TARGET_ORIENTATION = "LIA"

def load_and_preprocess(path: str) -> np.ndarray:
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
        Preprocessed volume, shape (1, X, Y, Z, 1) — batch and channel dims
        included for direct model input.
    """
    img = nib.load(path)
    img = _reorient(img)
    img = _resample(img)
    data = img.get_fdata(dtype=np.float32)
    data = _normalize(data)
    return data[np.newaxis, ..., np.newaxis]  # (1, X, Y, Z, 1)

def _reorient(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorient image to TARGET_ORIENTATION (LIA) using nibabel's orientation
    transform utilities.

    Handles oblique acquisitions by snapping to the closest cardinal
    orientation before transforming to LIA. Warns if the input orientation
    is ambiguous or cannot be determined from the affine.
    """
    current_codes = nib.aff2axcodes(img.affine)
    current_orientation = "".join(current_codes)

    if current_orientation == TARGET_ORIENTATION:
        return img

    logger.debug(
        f"Reorienting from '{current_orientation}' to '{TARGET_ORIENTATION}'."
    )

    try:
        current_ornt = nib.orientations.axcodes2ornt(current_codes)
        target_ornt = nib.orientations.axcodes2ornt(tuple(TARGET_ORIENTATION))
        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
        img = img.as_reoriented(transform)
    except Exception as e:
        logger.warning(
            f"Reorientation from '{current_orientation}' to '{TARGET_ORIENTATION}' "
            f"failed ({e}). Proceeding with original orientation -- "
            f"results may be degraded."
        )

    return img


def _resample(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Resample image to TARGET_ZOOMS (1mm isotropic) and conform to TARGET_SHAPE.

    Uses nibabel.processing.conform() which handles both resampling and
    field-of-view cropping/padding in a single step, preserving the affine
    correctly. Interpolation order 1 (trilinear) is standard for T1w data.
    """
    current_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    current_shape = img.shape[:3]

    if current_zooms == TARGET_ZOOMS and current_shape == TARGET_SHAPE:
        return img

    logger.debug(
        f"Resampling: shape {current_shape} -> {TARGET_SHAPE}, "
        f"zooms {current_zooms} -> {TARGET_ZOOMS}."
    )

    try:
        img = conform(
            img,
            out_shape=TARGET_SHAPE,
            voxel_size=TARGET_ZOOMS,
            order=1,        # trilinear interpolation
            cval=0.0,       # fill value for regions outside FOV
            orientation=TARGET_ORIENTATION,
        )
    except Exception as e:
        logger.warning(
            f"Resampling failed ({e}). Proceeding with original resolution -- "
            f"results may be degraded."
        )

    return img


def _normalize(data: np.ndarray) -> np.ndarray:
    # basic z-scoring, as we did when training
    return stats.zscore(data, axis=None)

# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def save_outputs(
    results: dict,
    input_path: str,
    output_dir: str,
    outputs: list[str],
) -> None:
    """
    Save model outputs to disk, mirroring input filename structure.

    Parameters
    ----------
    results : dict
        Output of NeuroFM.predict(), keyed by output name.
    input_path : str
        Original input file path (used to derive output filename).
    output_dir : str
        Root directory to write outputs into.
    outputs : list[str]
        Which outputs to save, e.g. ["brain_health", "latent"].
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = _get_stem(input_path)

    if "brain_health" in outputs and "brain_health" in results:
        _save_brain_health(results["brain_health"], stem, output_dir)

    if "latent" in outputs and "latent" in results:
        out_path = os.path.join(output_dir, f"{stem}_latent.npy")
        np.save(out_path, results["latent"])
        logger.debug(f"Saved latent features → {out_path}")


def _save_brain_health(values: np.ndarray, stem: str, output_dir: str) -> None:
    """Save brain health features as both .npy and a .csv."""
    npy_path = os.path.join(output_dir, f"{stem}_brain_health.npy")
    csv_path = os.path.join(output_dir, f"{stem}_brain_health.csv")

    np.save(npy_path, values)

    df = pd.DataFrame([values.tolist()], columns=BRAIN_HEALTH_KEYS)
    df.insert(0, "input", stem)
    df.to_csv(csv_path, index=False)

    logger.debug(f"Saved brain health → {npy_path}, {csv_path}")

def save_batch_summary(
    all_results: list[dict],
    input_paths: list[str],
    output_dir: str,
) -> None:
    """
    Write a single summary CSV consolidating brain_health results across
    all processed scans. Appended to output_dir/results_summary.csv.
    """
    rows = []
    for path, result in zip(input_paths, all_results, strict=True):
        if "brain_health" not in result:
            continue
        row = {"input": path}
        row.update(dict(zip(BRAIN_HEALTH_KEYS, result["brain_health"].tolist(), strict=True)))
        rows.append(row)

    if not rows:
        return

    summary_path = os.path.join(output_dir, "results_summary.csv")
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(summary_path)
    df.to_csv(summary_path, mode="a", header=write_header, index=False)
    logger.info(f"Summary written → {summary_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stem(path: str) -> str:
    """Strip directory and NIfTI extensions to get a clean filename stem."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base
