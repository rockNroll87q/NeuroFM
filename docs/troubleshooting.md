# Troubleshooting

**TensorFlow not finding GPU**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If the list is empty, check your CUDA/cuDNN versions against the [TF 2.13 compatibility table](https://www.tensorflow.org/install/source#gpu). The Docker/Singularity containers have the correct drivers pre-configured and are the easiest path for GPU use.

**Apple Silicon (M1/M2/M3/M4)**
The Docker and Singularity containers will crash with `Illegal Instruction` on Apple Silicon - `tensorflow:2.13.0` is compiled with AVX2/AVX-512 instructions that Rosetta 2 does not fully emulate. Use a local conda environment instead:
 
```bash
mamba create -n neurofm python=3.11
mamba activate neurofm
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal        # optional - enables GPU via Metal
pip install -e .
```
 
Then run inference directly:
```bash
python scripts/run_inference.py --input scan.nii.gz --output ./results/ --device cpu
```

**Out of memory on GPU**
Switch to a smaller variant (`--model neurofm-s`), or use `--device cpu`.

**Reorientation warnings**
If your data is in an unusual orientation and automatic reorientation fails, you may get a warning. Results may still be usable but accuracy can degrade - we recommend preprocessing to LIA orientation using `fslreorient2std` or equivalent.

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*