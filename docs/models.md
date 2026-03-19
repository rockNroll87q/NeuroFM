# Models

---

### Model variants

| Variant | Params | Latent dim | Weights (.h5) | Use case |
|---------|--------|------------|---------------|----------|
| `neurofm-s` | 484k | 161 | ~10MB | Default. Fast CPU inference, large cohorts |
| `neurofm-m` | 6.5M | 256 | ~44MB | Balanced representation/speed |
| `neurofm-l` | 10.8M | 512 | ~150MB | Maximum representation, GPU recommended |

Only the weights for your requested variant are downloaded.

---

### Weights & versioning

Weights are hosted on [HuggingFace](https://huggingface.co/NeuroAI-UofG/NeuroFM) for programmatic access. They are downloaded automatically on first use per variant, no manual steps required.

| Version | HuggingFace | Notes |
|---------|-------------|-------|
| v0.1.0 | [NeuroAI-UofG/NeuroFM](https://huggingface.co/NeuroAI-UofG/NeuroFM) | Initial release |

The auto-download pulls from HuggingFace by default. To point to a locally downloaded file:
```bash
python scripts/run_inference.py --input scan.nii.gz --weights /path/to/weights.h5
```

For long-term reproducibility and citation in publications, please reference the bioarxiv rather than the HuggingFace mirror.

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*