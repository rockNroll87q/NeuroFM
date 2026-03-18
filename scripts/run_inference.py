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
        --output-mode mirror \\
        --device gpu

Resume interrupted run (skip already-processed scans):
    python scripts/run_inference.py --input /data/ --output ./results/

Force reprocess everything:
    python scripts/run_inference.py --input /data/ --output ./results/ --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

# Allow running as a script without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurofm import NeuroFM  # noqa: I001
from neurofm.io import (
    OUTPUT_MODES,
    infer_input_root,
    load_cached_result,
    resolve_inputs,
    save_batch_summary,
    save_outputs,
)
from neurofm.weights import DEFAULT_VARIANT, list_variants


def parse_args() -> argparse.Namespace:
    """Get our arg parser"""
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
        "--output-mode",
        default="flat",
        choices=OUTPUT_MODES,
        help=(
            "How to organise output files (default: flat).\n"
            "  flat    — all outputs in a single directory\n"
            "  mirror  — mirrors the input directory structure\n"
            "  summary — summary CSV and aggregate latent .npy only, no per-file outputs"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs. By default, scans with existing "
             "outputs are loaded from disk and skipped (cache behaviour).",
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
        help="Path to a local weights .h5 file. Overrides automatic download.",
    )
    parser.add_argument(
        "--cache-dir",
        default="~/.cache/NeuroFM",
        metavar="DIR",
        help="Directory for caching downloaded weights (default: ~/.cache/NeuroFM).",
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
    """Main"""
    args = parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if args.list_variants:
        list_variants()
        sys.exit(0)

    requested_outputs = [o.strip() for o in args.outputs.split(",")]
    output_root = os.path.expanduser(args.output)
    output_root = Path(output_root)

    # ------------------------------------------------------------------
    # Resolve inputs
    # ------------------------------------------------------------------
    logger.info(f"Resolving inputs from: {args.input}")
    try:
        input_paths = resolve_inputs(args.input)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Found {len(input_paths)} scan(s) to process.")

    # For mirror mode — find the common root of all inputs so relative
    # paths can be reconstructed correctly
    input_root = infer_input_root(input_paths)
    input_root = Path(input_root)

    # ------------------------------------------------------------------
    # Cache check — split inputs into cached vs needs inference
    # ------------------------------------------------------------------
    all_results: list[dict | None] = [None] * len(input_paths)
    to_run: list[tuple[int, str]] = []   # (original index, path)
    n_cached = 0

    if not args.overwrite and args.output_mode != "summary":
        for idx, path in enumerate(input_paths):
            cached = load_cached_result(
                path, output_root, args.output_mode,
                requested_outputs, input_root,
            )
            if cached is not None:
                all_results[idx] = cached
                n_cached += 1
            else:
                to_run.append((idx, path))
    else:
        to_run = list(enumerate(input_paths))

    if n_cached:
        logger.info(
            f"{n_cached} scan(s) loaded from cache. "
            f"{len(to_run)} scan(s) queued for inference."
        )

    # ------------------------------------------------------------------
    # Load model (only if there's actually inference to run)
    # ------------------------------------------------------------------
    if to_run:
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

        paths_to_run = [path for _, path in to_run]
        batch_results = model.predict_batch(paths_to_run, outputs=requested_outputs)

        for (original_idx, _), result in zip(to_run, batch_results, strict=True):
            all_results[original_idx] = result
    else:
        logger.info("All scans loaded from cache — skipping model load.")

    # ------------------------------------------------------------------
    # Save per-file outputs
    # ------------------------------------------------------------------
    successful = 0
    failed = 0

    for path, result in zip(input_paths, all_results, strict=True):
        if result is None:
            failed += 1
            continue
        try:
            save_outputs(
                result, path, output_root,
                requested_outputs, args.output_mode, input_root,
            )
            successful += 1
        except Exception as e:
            logger.warning(f"Failed to save outputs for {path}: {e}")
            failed += 1

    # ------------------------------------------------------------------
    # Write aggregate summary
    # ------------------------------------------------------------------
    save_batch_summary(
        all_results, input_paths, output_root, requested_outputs,
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    logger.info(
        f"Done. {successful} scan(s) completed"
        + (f" ({n_cached} from cache)" if n_cached else "")
        + (f", {failed} failed." if failed else ".")
    )

    if failed:
        logger.warning(f"{failed} scan(s) failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()