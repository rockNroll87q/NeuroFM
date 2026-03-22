# Installation instructions

NeuroFM can be installed or used via the following options. We recommend installing via pip, but if you have docker that may be the easiest and fastest method.

### Option 1: pip (recommended)
```bash
pip install git+https://github.com/rockNroll87q/NeuroFM.git
```

Alternatively: 
```bash
git clone https://github.com/rockNroll87q/NeuroFM.git
cd NeuroFM
pip install .
```

> NeuroFM requires Python 3.9–3.11 and TensorFlow 2.13. **Python 3.12+ is not supported**. If you are on 3.12 or newer, please create a dedicated environment with Python 3.11 using Miniforge (see [Option 2](#option-2-miniforge)) or use [Docker](#option-3-docker-zero-friction-recommended-if-you-hit-dependency-issues). See [requirements.txt](../requirements.txt) for full dependencies.

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

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rockNroll87q/NeuroFM/issues).*