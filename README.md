# NeuroFM

This is the official repository for the paper "NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation".

> **NeuroFM is a foundation model trained exclusively on healthy brains that organizes structural MRI into population-level representations of brain health, transferring across five neuroscience domains and supporting individual-level profiling without ever seeing a diagnostic label.**

📄 [Paper (bioRxiv)]() &nbsp;|&nbsp; 🖥️ [Website](https://rocknroll87q.github.io/NeuroFM/) &nbsp;|&nbsp; 📦 [Weights (v0.1.0)](https://huggingface.co/NeuroAI-UofG/NeuroFM) &nbsp;|&nbsp; 🐳 [Docker](https://hub.docker.com/r/rocknroll87q/neurofm) &nbsp;|&nbsp; 📓 [Notebooks](./notebooks/) 

---

## What it does

NeuroFM takes a T1w MRI scan and produces:

| Output | Description | Format |
|--------|-------------|--------|
| `brain_health` | Predicted brain health features: brain age, brain volume (GM+WM), lateral ventricle volume, sex | floats, `.npy` or `.csv` |
| `latent` *(optional)* | Latent embedding (dimension depends on model variant) | `.npy` array |

NeuroFM comes in three sizes. The smallest variant is under 10MB and runs comfortably on CPU; the largest is ~150MB and is intended for GPU use or when maximum accuracy is needed.

---

## Quickstart

Get a result in under 5 minutes on CPU, no GPU required.

**1. Install**
```bash
pip install git+https://github.com/rockNroll87q/NeuroFM.git
```

Or clone and install locally:
```bash
git clone https://github.com/rockNroll87q/NeuroFM.git
cd NeuroFM
pip install -e .
```

**2. Run on a single file**
```bash
python scripts/run_inference.py --input /path/to/scan.nii.gz --output /path/to/output/
```

Weights for the default model variant (NeuroFM-S) are downloaded automatically (~10MB) on first run and cached to `~/.cache/NeuroFM/`. Larger variants are downloaded on demand when `--model` is specified.

---

## Installation

### Option 1: pip (recommended)
```bash
pip install git+https://github.com/rockNroll87q/NeuroFM.git
```
Requires Python 3.9–3.10 and TensorFlow 2.13. See [requirements.txt](./requirements.txt) for full dependencies.

### Option 2: Miniforge
```bash
mamba create -n neurofm python=3.11
mamba activate neurofm
pip install git+https://github.com/rockNroll87q/NeuroFM.git"
```

### Option 3: Docker *(zero-friction, recommended if you hit dependency issues)*
```bash
docker pull rocknroll87q/neurofm:latest
docker run --rm -v /path/to/data:/data rocknroll87q/neurofm \
    --input /data/scan.nii.gz --output /data/output/
```

### Option 4: Singularity *(for HPC/cluster environments)*
```bash
singularity pull neurofm.sif docker://rocknroll87q/neurofm:latest
singularity run --bind /path/to/data:/data neurofm.sif \
    --input /data/scan.nii.gz --output /data/output/
```

---

## Usage

### Single file
```bash
python scripts/run_inference.py \
    --input subject_01_T1w.nii.gz \
    --output ./results/
```

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
```
--input          Path to a .nii.gz file, directory, or .csv with an 'input' column
--output         Output directory
--model          Model variant: neurofm-s (default), neurofm-m, neurofm-l
--outputs        Comma-separated list of outputs: brain_health,latent (default: brain_health)
--device         Device: auto (default), cpu, gpu
--weights        Path to local weights file (overrides automatic download)
```

### Python API
```python
from neurofm import NeuroFM

model = NeuroFM(model="neurofm-s", device="auto")
results = model.predict("subject_01_T1w.nii.gz", outputs=["brain_health", "latent"])

results["brain_health"]  # np.ndarray, shape (4,) — [brain_age, brain_vol, ventricle_vol, sex]
results["latent"]        # np.ndarray, shape (D,)
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](./notebooks/01_quickstart.ipynb) | Run inference on a single scan end-to-end |
| [02_batch_inference_csv.ipynb](./notebooks/02_batch_inference_csv.ipynb) | Batch processing a study cohort via CSV |
| [03_latent_features.ipynb](./notebooks/03_latent_features.ipynb) | Extracting and visualizing latent embeddings |

---

## Input requirements

| Parameter | Requirement |
|-----------|-------------|
| Modality | T1-weighted MRI |
| Format | NIfTI (`.nii`, `.nii.gz`) |
| Preprocessing | Skull-stripped |
| Resolution | 1mm isotropic |
| Orientation | LIA recommended; the pipeline will attempt to reorient automatically |

> **Note on preprocessing:** The inference script performs resolution resampling and attempts LIA reorientation internally. Input data must be skull-stripped prior to inference. Preprocessing utilities will be added in a future release.

---

## Outputs

### Brain health features

A 4-element array in the order `[brain_age, sex, ventricle_volume, brain_volume]`.
 
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `brain_age` | Predicted brain age | years | 40.0 - 90.0 |
| `sex` | Predicted biological sex | - | 0.0 (female) - 1.0 (male) |
| `ventricle_volume` | Lateral ventricle volume | mm³ | 0 - 180×10³ |
| `brain_volume` | Total brain volume (GM+WM) | mm³ | 1×10⁶ - 1.9×10⁶ |

### Latent features
A D-dimensional embedding representing the brain health representation space learned by NeuroFM. Useful for downstream tasks including differential diagnosis classification, cognitive score regression, and unsupervised cohort clustering. Extracted from the `multihead_output` layer. The embedding dimension D depends on the model variant (see below).

---

## Model Variants

| Variant | Params | Latent dim | Weights (.h5) | Use case |
|---------|--------|------------|---------------|----------|
| `neurofm-s` | 484k | 161 | ~10MB | Default. Fast CPU inference, large cohorts |
| `neurofm-m` | 6.5M | 256 | ~44MB | Balanced accuracy/speed |
| `neurofm-l` | 10.8M | 512 | ~150MB | Maximum accuracy, GPU recommended |

Only the weights for your requested variant are downloaded. See [Weights & Versioning](#weights--versioning).

---

## Weights & Versioning

Weights are hosted on [HuggingFace]() for programmatic access. They are downloaded automatically on first use per variant — no manual steps required.

| Version | HuggingFace | Notes |
|---------|-------------|-------|
| v0.1.0 | [NeuroAI-UofG/NeuroFM](https://huggingface.co/NeuroAI-UofG/NeuroFM) | Initial release |

The auto-download pulls from HuggingFace by default. To point to a locally downloaded file:
```bash
python scripts/run_inference.py --input scan.nii.gz --weights /path/to/weights.h5
```

For long-term reproducibility and citation in publications, please reference the bioarxiv rather than the HuggingFace mirror.
---

## Finetuning

For inference scenarios, the scripts load the NeuroFM saved `.h5` weights as these are smaller and sufficient in most cases. For finetuning, the full saved model directory is available on HuggingFace. Finetuning is likely needed for inference tasks on other MRI modalities (T2-weighted, for example).

For now, finetuning is not officially supported by this repository but may be added at a later date.

---

## Troubleshooting

**TensorFlow not finding GPU**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If the list is empty, check your CUDA/cuDNN versions against the [TF 2.13 compatibility table](https://www.tensorflow.org/install/source#gpu). The Docker/Singularity containers have the correct drivers pre-configured and are the easiest path for GPU use.

**Apple Silicon (M1/M2/M3/M4)**
The Docker and Singularity containers will crash with `Illegal Instruction` on Apple Silicon — `tensorflow:2.13.0` is compiled with AVX2/AVX-512 instructions that Rosetta 2 does not fully emulate. Use a local conda environment instead:
 
```bash
mamba create -n neurofm python=3.11
mamba activate neurofm
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal        # optional — enables GPU via Metal
pip install -e .
```
 
Then run inference directly:
```bash
python scripts/run_inference.py --input scan.nii.gz --output ./results/ --device cpu
```

**Out of memory on GPU**
Switch to a smaller variant (`--model neurofm-s`), or use `--device cpu`.

**Reorientation warnings**
If your data is in an unusual orientation and automatic reorientation fails, you may get a warning. Results may still be usable but accuracy can degrade — we recommend preprocessing to LIA orientation using `fslreorient2std` or equivalent.

---

## Citation

If you use NeuroFM in your research, please cite:

```bibtex
@article{yourname2026neurofm,
  title   = {NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation},
  author  = {[Authors]},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

NeuroFM is released under the [LICENSE NAME] license. See [LICENSE](./LICENSE) for details.

---

## Contributing

This repository is in an early-release state accompanying the manuscript. Bug reports and questions are welcome via [GitHub Issues](). Please open an issue before submitting a pull request.

---

## Acknowledgements

[Funding, institutions, datasets used for training/validation.]