# Usage examples and full options

Here is a more extensive glossary of usage examples, and the full table of options for `run_inference.py` is given at the bottom.

### Directory of scans
```bash
python scripts/run_inference.py \
    --input /data/my_study/ \
    --output ./results/
```
Recursively finds all `.nii.gz` files. Output filenames mirror the input structure.

### CSV with an input column
```bash
python scripts/run_inference.py \
    --input subjects.csv \
    --output ./results/
```
Your CSV must have an `input` column containing paths to NIfTI files. Any other columns are ignored and passed through to the output summary CSV.

### Select a model variant
```bash
python scripts/run_inference.py \
    --input /data/ \
    --output ./results/ \
    --model neurofm-m
```
Defaults to `neurofm-s`. Weights for the requested variant are downloaded automatically if not already cached. See [Model Variants](#model-variants) for a full comparison.

### Select specific outputs
```bash
# Produce brain health estimates and latent features
python scripts/run_inference.py \
    --input /data/ \
    --output ./results/ \
    --outputs brain_health,latent
```
Default produces `brain_health` only. Add `latent` to also extract embeddings.

### GPU inference
```bash
python scripts/run_inference.py \
    --input /data/ \
    --output ./results/ \
    --device gpu
```
Defaults to `auto` (uses GPU if available, falls back to CPU). Force CPU with `--device cpu`.

### Full options

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to a `.nii`/`.nii.gz` file, directory, or `.csv` with an `input` column | - |
| `--output` | Output directory | - |
| `--output-mode` | Output layout: `flat`, `mirror`, `summary` | `flat` |
| `--model` | Model variant: `neurofm-s`, `neurofm-m`, `neurofm-l` | `neurofm-s` |
| `--outputs` | Comma-separated outputs to produce: `brain_health`, `latent` | `brain_health` |
| `--device` | Compute device: `auto`, `cpu`, `gpu` | `auto` |
| `--weights` | Path to local weights file, overrides automatic download | - |
| `--cache-dir` | Directory for caching downloaded weights | `~/.cache/NeuroFM` |
| `--overwrite` | Overwrite existing outputs. Without this flag, partially processed runs are resumed | `False` |
| `--list-variants` | Print available model variants and exit | - |
| `--verbose` | Enable debug logging | `False` |

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*