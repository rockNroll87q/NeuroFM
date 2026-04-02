# Limitations and bias

Some limitations of our work and models:

- Scanner and acquisition protocol diversity in the training data is limited
  to T1-weighted 3-Tesla emulation from the LDM100k dataset. Generalisation to 
  substantially different acquisition parameters (e.g. ultra-high field, 
  non-standard contrasts) is not guaranteed.
- Training data from LDM100k is based on the UK Biobank population and likely represents 
  similar systematic demographic biases. 
- Brain age estimates reflect the population-level relationship between MRI
  appearance and chronological age in the training data. They should not be
  interpreted as a direct measure of neurological health in individuals.
- Sex classification reflects biological sex as recorded in dataset metadata
  and is a binary prediction. It does not reflect gender identity.
- **This model is intended for research use only and has not been validated for 
  clinical decision-making.** NeuroFM outputs should not be used to inform 
  diagnosis, treatment, or any clinical decision affecting individual patient care. 
  The authors and the University of Glasgow accept no liability for consequences 
  arising from unsupported clinical use.
  
---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*