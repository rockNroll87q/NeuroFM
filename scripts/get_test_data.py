#!/usr/bin/env python3
"""
=============================================================================
Austin Dibble
University of Glasgow
2026

scripts/get_test_data.py

Utility to fetch test data for debugging NeuroFM inference.

If using the template mode, this script requires templateflow.
To avoid dependency conflicts with the other repo packages, install as: 

    pip install templateflow "numpy<=1.24.3" "typing-extensions<4.6.0"

Two modes:
  --mode template   Download the MNI152 1mm skull-stripped T1w template via
                    TemplateFlow. Real data, correct space, meaningful outputs.
                    Requires internet access (~8MB download, cached after first run).

  --mode synthetic  Generate a synthetic skull-shaped volume using nibabel and
                    numpy. No internet required. Useful for pure pipeline/shape
                    testing, but model outputs will be arbitrary.

Examples
--------
  python scripts/get_test_data.py --output ./test_data/
  python scripts/get_test_data.py --output ./test_data/ --mode synthetic
  python scripts/get_test_data.py --output ./test_data/ --mode template -n 3
  
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Get our argument parser"""
    parser = argparse.ArgumentParser(
        description="Fetch or generate test NIfTI data for NeuroFM debugging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default="./test_data/",
        help="Directory to write test volumes into (default: ./test_data/).",
    )
    parser.add_argument(
        "--mode",
        choices=["template", "synthetic"],
        default="template",
        help="Data source: 'template' (MNI152 via TemplateFlow) or "
             "'synthetic' (generated noise volume). Default: template.",
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=1,
        help="Number of volumes to produce. For 'template' mode, copies of the "
             "same template are written with different filenames (useful for "
             "testing batch/CSV input). Default: 1.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Template mode - MNI152 via TemplateFlow
# ---------------------------------------------------------------------------

def fetch_template(output_dir: str, n: int) -> List[str]:
    """
    Download the MNI152NLin2009cAsym 1mm skull-stripped T1w template.

    This is the standard MNI space: skull-stripped, 1mm isotropic, and the
    space NeuroFM expects. A good sanity check - brain age output should be
    in the plausible adult range, brain volume ~1.2-1.4L.

    Args:
        - n: the number of copies to produce in the output dir.
    """
    try:
        import templateflow.api as tflow
    except ImportError:
        print(
            'TemplateFlow not installed. Install with:\n'
            '  pip install templateflow "numpy<=1.24.3" "typing-extensions<4.6.0"\n'
            'Or use --mode synthetic for offline testing.'
        )
        sys.exit(1)

    print("Fetching MNI152NLin2009cAsym 1mm T1w template from TemplateFlow...")
    template_path = tflow.get(
        "MNI152NLin2009cAsym",
        resolution=1,
        desc="brain",
        suffix="T1w",
        extension=".nii.gz",
    )

    if template_path is None:
        print("TemplateFlow could not fetch the template. Check your internet connection.")
        sys.exit(1)

    print(f"Template downloaded: {template_path}")

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for i in range(n):
        suffix = f"_{i + 1:02d}" if n > 1 else ""
        dest = os.path.join(output_dir, f"MNI152_test{suffix}.nii.gz")
        shutil.copy(str(template_path), dest)
        written.append(dest)
        print(f"  Written: {dest}")

    return written


# ---------------------------------------------------------------------------
# Synthetic mode - noise volume in MNI space
# ---------------------------------------------------------------------------

def generate_synthetic(output_dir: str, n: int) -> List[str]:
    """
    Generate a synthetic skull-shaped volume filled with structured noise.

    Shape and affine match MNI152 1mm space (182x218x182). The volume is
    masked to a rough ellipsoid to vaguely resemble a skull-stripped brain.
    Model outputs will not be meaningful, but the full pipeline will run.

    Args:
        - n: the number of copies to produce in the output dir.
    """
    shape = (182, 218, 182)

    # MNI152 1mm affine (LIA orientation)
    affine = np.array([
        [-1.,  0.,  0.,  90.],
        [ 0.,  1.,  0., -126.],
        [ 0.,  0.,  1., -72.],
        [ 0.,  0.,  0.,   1.],
    ])

    os.makedirs(output_dir, exist_ok=True)
    written = []

    for i in range(n):
        print(f"Generating synthetic volume {i + 1}/{n}...")
        data = _make_synthetic_volume(shape)
        img = nib.Nifti1Image(data, affine)
        img.header.set_zooms((1.0, 1.0, 1.0))

        suffix = f"_{i + 1:02d}" if n > 1 else ""
        dest = os.path.join(output_dir, f"synthetic_test{suffix}.nii.gz")
        nib.save(img, dest)
        written.append(dest)
        print(f"  Written: {dest}  (shape: {data.shape}, dtype: {data.dtype})")

    return written


def _make_synthetic_volume(shape: tuple) -> np.ndarray:
    """
    Generate a float32 volume with smooth Gaussian noise masked to an ellipsoid.
    Intensity roughly normalised to [0, 1] to mimic a skull-stripped T1w scan.
    """
    rng = np.random.default_rng(seed=42)

    # Smooth noise base
    raw = rng.standard_normal(shape).astype(np.float32)

    # Rough brain-shaped ellipsoid mask
    cx, cy, cz = [s // 2 for s in shape]
    rx, ry, rz = shape[0] * 0.42, shape[1] * 0.38, shape[2] * 0.42
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0
    data = np.where(mask, np.abs(raw), 0.0)

    # Normalise to [0, 1]
    p99 = np.percentile(data[mask], 99)
    if p99 > 0:
        data = np.clip(data / p99, 0.0, 1.0)

    return data


# ---------------------------------------------------------------------------
# CSV helper - optional, generates a subjects.csv for batch testing
# ---------------------------------------------------------------------------

def write_test_csv(paths: List[str], output_dir: str) -> str:
    """Write a subjects.csv pointing at the generated test volumes."""
    import pandas as pd
    csv_path = os.path.join(output_dir, "subjects.csv")
    df = pd.DataFrame({"input": paths, "subject_id": [f"sub-{i+1:02d}" for i in range(len(paths))]})
    df.to_csv(csv_path, index=False)
    print(f"  CSV written: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main"""
    args = parse_args()

    print("\nNeuroFM test data utility")
    print(f"  Mode   : {args.mode}")
    print(f"  Output : {args.output}")
    print(f"  N      : {args.n}\n")

    paths = fetch_template(args.output, args.n) if args.mode == "template" else generate_synthetic(args.output, args.n)

    if args.n > 1:
        csv_path = write_test_csv(paths, args.output)

    print("\nDone. Test with:")
    if args.n == 1:
        print(f"  python scripts/run_inference.py --input {paths[0]} --output ./test_results/")
    else:
        print(f"  python scripts/run_inference.py --input {args.output} --output ./test_results/")
        if args.n > 1:
            print(f"  python scripts/run_inference.py --input {csv_path} --output ./test_results/")

if __name__ == "__main__":
    main()