# Model outputs

NeuroFM can produce two distinct sets of outputs: brain health features, and latent features. By default, `run_inference.py` only produces the brain health features. For more guidance on run options, see [usage](./usage.md).

### Brain health features

A 4-element array in the order `[brain_age, sex, ventricle_volume, brain_volume]`.
 
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `brain_age` | Predicted brain age | years | 40.0 - 90.0 |
| `sex` | Predicted biological sex | - | 0.0 (female) - 1.0 (male) |
| `ventricle_volume` | Lateral ventricle volume | mm³ | 0 - 180×10³ |
| `brain_volume` | Total brain volume (GM+WM) | mm³ | 1×10⁶ - 1.9×10⁶ |

These can be used for basic regressive analyses such as brain age gap regression.

For a very basic example, see the [quickstart notebook](../notebooks/01_quickstart.ipynb).

### Latent features

A D-dimensional embedding representing the brain health representation space learned by NeuroFM. Useful for downstream tasks including differential diagnosis classification, cognitive score regression, and unsupervised cohort clustering. Extracted from the `multihead_output` layer. The embedding dimension D depends on the model variant (see [models](./models.md)).

For a very basic example, see the [latent features notebook](../notebooks/03_latent_features.ipynb).

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*