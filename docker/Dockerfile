# NeuroFM - User-facing inference container
#
# Usage:
#   docker pull rocknroll87q/neurofm:latest
#
#   # Single file
#   docker run --rm -v /path/to/data:/data rocknroll87q/neurofm \
#       --input /data/scan.nii.gz --output /data/results/
#
#   # Directory
#   docker run --rm -v /path/to/data:/data rocknroll87q/neurofm \
#       --input /data/study/ --output /data/results/
#
#   # GPU
#   docker run --rm --gpus all -v /path/to/data:/data rocknroll87q/neurofm \
#       --input /data/scan.nii.gz --output /data/results/ --device gpu
#
# Build:
#   docker build -t rocknroll87q/neurofm:latest -f docker/Dockerfile .

FROM tensorflow/tensorflow:2.13.0-gpu

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Suppress TF startup noise; set to 0 for full verbosity when debugging
ENV TF_CPP_MIN_LOG_LEVEL=2

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/neurofm

# Install Python dependencies first to cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install the package
COPY . .
RUN pip install --no-cache-dir .

ENV NEUROFM_CACHE_DIR=/root/.cache/NeuroFM

# Default entrypoint runs the inference script.
# All arguments after the image name are passed through to run_inference.py.
ENTRYPOINT ["python", "/opt/neurofm/scripts/run_inference.py"]
CMD ["--help"]