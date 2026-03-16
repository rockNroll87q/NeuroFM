"""
NeuroFM: Foundation model for individualized brain health estimation from T1w MRI.

Basic usage:
    from neurofm import NeuroFM

    model = NeuroFM(variant="neurofm-s", device="auto")
    results = model.predict("subject_01_T1w.nii.gz")
"""

from .inference import NeuroFM

__version__ = "0.1.0"
__all__ = ["NeuroFM"]