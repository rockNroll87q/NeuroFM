# NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation

![Main figure](<./media/Fig1 - v0.11_web.png>)

> **NeuroFM is a foundation model trained exclusively on healthy brains that organizes structural MRI into population-level representations of brain health, transferring across five neuroscience domains and supporting individual-level profiling without ever seeing a diagnostic label.**

📄 [Paper (medRxiv)](https://doi.org/10.64898/2026.03.27.26349489) &nbsp;|&nbsp; 🖥️ [Website](https://rocknroll87q.github.io/NeuroFM/) &nbsp;|&nbsp; 📦 [Weights (v0.1.0)](https://huggingface.co/NeuroAI-UofG/NeuroFM) &nbsp;|&nbsp; 🐳 [Docker](https://hub.docker.com/r/rocknroll87q/neurofm) &nbsp;|&nbsp; 📓 [Notebooks](./notebooks/) 


## What it does

NeuroFM takes a T1w MRI scan and produces:

| Output | Description | Format |
|--------|-------------|--------|
| `brain_health` | Predicted brain health features: brain age, brain volume (GM+WM), lateral ventricle volume, sex | floats, `.npy` or `.csv` |
| `latent` *(optional)* | Latent embedding (dimension depends on model variant) | `.npy` array |

To understand the differences in outputs and how they might be used, see [outputs](./docs/outputs.md).

NeuroFM comes in three sizes. The smallest variant is under 10MB and runs comfortably on CPU; the largest is ~150MB and is intended for GPU use or when maximum accuracy is needed.


## Installation

```bash
pip install git+https://github.com/rockNroll87q/NeuroFM.git
```

> **Requires Python 3.9–3.10 and TensorFlow 2.13**. For more installation options if the above doesn't work for you (miniforge, docker, etc.), see the [installation guide](./docs/installation.md). See [requirements.txt](./requirements.txt) for full dependencies.


## Usage

### Single file, default model (NeuroFM-S)
```bash
python scripts/run_inference.py \
    --input subject_01_T1w.nii.gz \
    --output ./results/
```

### Python API
```python
from neurofm import NeuroFM

model = NeuroFM(variant="neurofm-s", device="auto")
results = model.predict("subject_01_T1w.nii.gz", outputs=["brain_health", "latent"])

results["brain_health"]  # np.ndarray, shape (4,) - [brain_age, brain_vol, ventricle_vol, sex]
results["latent"]        # np.ndarray, shape (D,)
```

We recommend reading through the full [usage and options guide](./docs/usage.md).

Weights for the default model variant (NeuroFM-S) are downloaded automatically (~10MB) on first run and cached to `~/.cache/NeuroFM/`. Larger variants are downloaded on demand when `--model` is specified.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](./notebooks/01_quickstart.ipynb) | Run inference on a single scan end-to-end |
| [02_batch_inference.ipynb](./notebooks/02_batch_inference.ipynb) | Batch processing a study cohort via CSV |
| [03_latent_features.ipynb](./notebooks/03_latent_features.ipynb) | Extracting and visualizing latent embeddings |


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

## Model variants & weights

NeuroFM is distributed in three different variants: NeuroFM-S, NeuroFM-M, and NeuroFM-L. While the primary difference is parameter size, they each provide successively deeper networks and larger latent representations. 

To see a more detailed explanation, see [models](./docs/models.md).

For long-term reproducibility and citation in publications, please reference the bioarxiv rather than the HuggingFace mirror.


## Finetuning

For inference scenarios, the scripts load the NeuroFM saved `.h5` weights as these are smaller and sufficient in most cases. For finetuning, the full saved model directory is available on HuggingFace. Finetuning is likely needed for inference tasks on other MRI modalities (T2-weighted, for example).

For now, finetuning is not officially supported by this repository but may be added at a later date.


## Troubleshooting

For troubleshooting, see the [troubleshooting guide](docs/troubleshooting.md).


## Limitations and bias

To understand limitations of our models, see [limitations](docs/limitations.md).


## Citation

If you use NeuroFM in your research, please cite:

```bibtex
@article {DibbleNeuroFM2026,
	author = {Dibble, Austin and Dalby, Connor and Sevegnani, Michele and Fracasso, Alessio and Lyall, Donald M and Harvey, Monika and Svanera, Michele},
	title = {NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation},
	elocation-id = {2026.03.27.26349489},
	year = {2026},
	doi = {10.64898/2026.03.27.26349489},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Precision neuroimaging aims to deliver individualized assessments of brain health, yet a single structural MRI does not yield a multidimensional, quantitative summary of an individual{\textquoteright}s current health or future risk. Existing approaches optimize task-specific objectives, yielding representations entangled with cohort- or disease-specific signals rather than capturing biologically grounded patterns of anatomical variation. Here, we introduce NeuroFM, a foundation model trained exclusively on 100,000 healthy synthetic volumes to predict morphometric and demographic targets. Without exposure to diagnostic labels, NeuroFM organizes brain MRIs into population-level patterns that encode meaningful brain health differences. These representations transfer across five neuroscience domains without adaptation and support simple linear readouts for clinical, cognitive, developmental, socio-behavioural, and image quality control. Evaluated on 136,361 real volumes spanning multiple cohorts, NeuroFM generalizes across domains and enables individual-level brain health profiling, estimating future dementia risk years before diagnosis. Together, these findings establish a disease-naive foundation model paradigm for precision neuroimaging. Code available at: https://rocknroll87q.github.io/NeuroFM/},
	URL = {https://www.medrxiv.org/content/early/2026/03/31/2026.03.27.26349489},
	eprint = {https://www.medrxiv.org/content/early/2026/03/31/2026.03.27.26349489.full.pdf},
	journal = {medRxiv}
}
```


## License

NeuroFM code and model weights are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

For further detail, see [LICENSE](./LICENSE).


## Contributing

This repository is in an early-release state accompanying the manuscript. Bug reports and questions are welcome via [GitHub Issues](https://github.com/rockNroll87q/NeuroFM/issues). Please open an issue before submitting a pull request.


## Acknowledgements

To see a full list of acknowledgements and declarations (funding, support, data sources, etc.), see the [acknowledgements page](docs/acknowledgements.md).
