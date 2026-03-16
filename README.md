# NeuroFM

Repository for the paper "NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation"

> **"[headline statement]"**

📄 [Paper (bioRxiv)]() &nbsp;|&nbsp; 📦 [Weights (v0.1.0)]() &nbsp;|&nbsp; 🐳 [Docker]() &nbsp;|&nbsp; 📓 [Notebooks](./notebooks/)

---

## What it does

NeuroFM takes a T1w MRI scan and produces:

| Output | Description | Format |
|--------|-------------|--------|
| `brain_health` | Predicted brain health features (brain age, brain volume (GM+WM), lateral ventricle volume, sex) | floats, `.npy`, or `.csv` |
| `latent` *(optional)* |  latent embedding (size depends on model variant) | `.npy` array |

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

That's it. Weights are downloaded automatically (~200MB) on first run and cached to `~/.cache/NeuroFM/`.

---

## Installation

### Option 1: pip (recommended)
```bash
pip install git+https://github.com/rockNroll87q/NeuroFM.git
```
Requires Python 3.9–3.10 and TensorFlow 2.13. See [requirements.txt](./requirements.txt) for full dependencies.

### Option 2: Conda
```bash
conda env create -f environment.yml
conda activate neurofm
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

### Select specific outputs
```bash
# Only produce brain_age and latent features
python scripts/run_inference.py \
    --input /data/ \
    --output ./results/ \
    --outputs brain_health,latent
```

Default is both outputs. Omitting `latent` skips embedding extraction.

### GPU inference
```bash
python scripts/run_inference.py \
    --input /data/ \
    --output ./results/ \
    --device gpu
```
Defaults to `auto` (uses GPU if available, falls back to CPU). Force CPU with `--device cpu`.

### Python API
```python
from neurofm import InferenceEngine

engine = InferenceEngine(device="auto")
results = engine.predict("subject_01_T1w.nii.gz", outputs=["brain_health", "latent"])

results["brain_health"]  # np.ndarray, shape (4,)
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
| Preprocessing | skull-stripped |
| Resolution | 1mm isotropic, even if it's not native |
| Orientation | LIA recommended, though pipeline will attempt to transform it to the right space |

> **Note on preprocessing:** Currently, no preprocessing is performed by the inference script beyond attempted resolution resampling and LIA transformation. Data is expected to be skull-stripped.

---

## Outputs

### Brain health features
[Description, units, interpretation guidance. What does a "good" output look like?]

| Feature | Description | Unit | value range
|--------|-------------|--------|--------|
| `brain age` | Predicted brain age | years | float (40.0 - 90.0) |
| `brain volume` | Predicted total brain volume (GM+WM) | mm^3 | float (1e6 - 1.9e6) |
| `ventricle volume` | Predicted lateral ventricle volume | mm^3 | float (0 - 180e3) |

### Latent features
A D-dimensional embedding representing the brain health representation space encoded by NeuroFM. Useful for downstream classification tasks (e.g., differential diagnosis), regression tasks (e.g. cognition score prediction), unsupervised clustering, etc. Extracted from `multihead_output` layer of the network. The latent dimension size depends on the selected model variant (see [Variants](#model-variants)).

---

## Model Variants

| Name | Params | Dimensions |
|---------|-------|------|
| NeuroFM-S | 484k | 161 |
| NeuroFM-M | 6.5M | 256 |
| NeuroFM-L | 10.8M | 512 |

---

## Weights & versioning

Model weights are distributed via GitHub Releases and HuggingFace and archived on Zenodo with a citable DOI.

| Version | Release | Zenodo DOI | Notes |
|---------|---------|------------|-------|
| v0.1.0 | [Link]() | [DOI]() | Initial release |

Weights are downloaded automatically on first use. To download manually or specify a local path:
```bash
python scripts/run_inference.py --input scan.nii.gz --weights /path/to/weights.h5
```

---

## System requirements

| | Minimum | Recommended |
|-|---------|-------------|
| Python | 3.9 | 3.10 |
| RAM | 4GB | 8GB |
| GPU | — | NVIDIA, 4GB VRAM |
| OS | Linux, macOS, Windows | Linux |

GPU is optional but recommended for large cohorts. Single-scan CPU inference takes approximately [X seconds] on a modern laptop.

---

## Troubleshooting

**TensorFlow not finding GPU**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If empty, check your CUDA/cuDNN versions against the [TF 2.13 compatibility table](https://www.tensorflow.org/install/source#gpu). The Docker/Singularity containers have the correct drivers pre-configured.

**Apple Silicon (M1/M2/M3)**
TF 2.13 does not support MPS acceleration. Use `--device cpu` or use the Docker container. Performance on Apple Silicon CPU is still reasonable for single-scan inference.

**Out of memory on GPU**
Reduce batch size with `--batch-size 1`, or use `--device cpu`.

---

## Citation

If you use [Model Name] in your research, please cite:

```bibtex
@article{yourname2026modelname,
  title   = {[Paper title]},
  author  = {[Authors]},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

[Model Name] is released under the [LICENSE NAME] license. See [LICENSE](./LICENSE) for details.

---

## Contributing

This repository is currently in an early-release state accompanying the manuscript. We welcome bug reports and questions via [GitHub Issues](). Broader contributions welcome — please open an issue before submitting a PR.

---

## Acknowledgements

[Funding, institutions, datasets used for training/validation.]