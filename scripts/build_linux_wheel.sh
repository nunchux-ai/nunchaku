#!/bin/bash
# Modified from https://github.com/sgl-project/sglang/blob/v0.4.10/sgl-kernel/build.sh
set -ex
PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
MAX_JOBS=${4:-} # optional
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

# Set the corresponding versions for TORCHVISION and TORCHAUDIO
if [ "$TORCH_VERSION" == "2.8" ]; then
  TORCHVISION_VERSION="0.23"
  TORCHAUDIO_VERSION="2.8"
elif [ "$TORCH_VERSION" == "2.9" ]; then
  TORCHVISION_VERSION="0.24"
  TORCHAUDIO_VERSION="2.9"
elif [ "$TORCH_VERSION" == "2.10" ]; then
  TORCHVISION_VERSION="0.25"
  TORCHAUDIO_VERSION="2.10"
else
  echo "Unsupported TORCH_VERSION: $TORCH_VERSION"
  exit 1
fi
echo "TORCH_VERSION is $TORCH_VERSION, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"

docker run --rm \
    -v "$(pwd)":/nunchaku \
    pytorch/manylinux2_28-builder:cuda${CUDA_VERSION} \
    bash -c "
    cd /nunchaku && \
    rm -rf build && \
    gcc --version && g++ --version && \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//.} && \
    ${PYTHON_ROOT_PATH}/bin/pip install build ninja wheel setuptools && \
    export NUNCHAKU_INSTALL_MODE=ALL && \
    export NUNCHAKU_BUILD_WHEELS=1 && \
    export MAX_JOBS=${MAX_JOBS} && \
    ${PYTHON_ROOT_PATH}/bin/python -m build --wheel --no-isolation
    "
