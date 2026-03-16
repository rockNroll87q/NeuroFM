#!/usr/bin/env python3
"""
scripts/run_inference.py

Command-line interface for NeuroFM inference.
Accepts a single NIfTI file, a directory, or a CSV with an 'input' column.

Examples
--------
Single file:
    python scripts/run_inference.py --input scan.nii.gz --output ./results/

Directory:
    python scripts/run_inference.py --input /data/study/ --output ./results/

CSV:
    python scripts/run_inference.py --input subjects.csv --output ./results/

With options:
    python scripts/run_inference.py \\
        --input /data/ \\
        --output ./results/ \\
        --model neurofm-l \\
        --outputs brain_health,latent \\
        --device gpu
"""

import argparse
from loguru import logger
import sys
import os

# Allow running as a script without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurofm import NeuroFM
from neurofm.io import resolve_inputs, save_outputs, save_batch_summary
from neurofm.weights import DEFAULT_VARIANT, list_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroFM — Foundation model inference for T1w MRI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .nii/.nii.gz file, a directory, or a .csv with an 'input' column.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_VARIANT,
        metavar="VARIANT",
        help=f"Model variant to use (default: {DEFAULT_VARIANT}). "
             "Run --list-variants to see all options.",
    )
    parser.add_argument(
        "--outputs",
        default="brain_health",
        help="Comma-separated list of outputs to produce. "
             "Options: brain_health, latent (default: brain_health).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device for inference (default: auto).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help="Path to a local weights .h5 file. "
             "Overrides automatic download.",
    )
    parser.add_argument(
        "--cache-dir",
        default="~/.cache/NeuroFM",
        metavar="DIR",
        help="Directory for caching downloaded weights "
             "(default: ~/.cache/NeuroFM).",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print available model variants and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_variants:
        list_variants()
        sys.exit(0)

    # Parse outputs
    requested_outputs = [o.strip() for o in args.outputs.split(",")]

    # Resolve inputs
    logger.info(f"Resolving inputs from: {args.input}")
    try:
        input_paths = resolve_inputs(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Found {len(input_paths)} scan(s) to process.")

    # Load model
    try:
        model = NeuroFM(
            variant=args.model,
            device=args.device,
            weights=args.weights,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Run inference
    all_results = model.predict_batch(input_paths, outputs=requested_outputs)

    # Save outputs
    successful = 0
    failed = 0
    for path, result in zip(input_paths, all_results):
        if result is None:
            failed += 1
            continue
        try:
            save_outputs(result, path, args.output, requested_outputs)
            successful += 1
        except Exception as e:
            logger.warning(f"Failed to save outputs for {path}: {e}")
            failed += 1

    # Write batch summary CSV if processing more than one scan
    if len(input_paths) > 1:
        completed_paths = [
            p for p, r in zip(input_paths, all_results) if r is not None
        ]
        completed_results = [r for r in all_results if r is not None]
        if completed_results:
            save_batch_summary(completed_results, completed_paths, args.output)

    # Final report
    logger.info(
        f"Done. {successful} scan(s) completed successfully"
        + (f", {failed} failed." if failed else ".")
    )
    if failed:
        logger.warning(
            f"{failed} scan(s) failed. Check logs above for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()