# Limitations and bias

---

Some limitations of our work and models:

- Training data is predominantly from **European cohorts** in the 40–90 year
  age range. Performance on younger adults, children, or non-European
  populations has not been systematically evaluated.
- Brain age estimates reflect the population-level relationship between MRI
  appearance and chronological age in the training data. They should not be
  interpreted as a direct measure of neurological health in individuals.
- Sex classification reflects biological sex as recorded in dataset metadata
  and is a binary prediction. It does not reflect gender identity.
- Scanner and acquisition protocol diversity in the training data is limited
  to T1-weighted 3-Tesla emulation from the LDM100k dataset. Generalisation to 
  substantially different acquisition parameters (e.g. ultra-high field, 
  non-standard contrasts) is not guaranteed.

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*