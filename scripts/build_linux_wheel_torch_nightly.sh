#!/bin/bash
# Modified from https://github.com/sgl-project/sglang/blob/v0.4.10/sgl-kernel/build.sh
set -ex
PYTHON_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
MAX_JOBS=${4:-} # optional
PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}

# if [ "$TORCH_VERSION" == "2.9" ]; then
#   TORCHVISION_VERSION="0.24"
#   TORCHAUDIO_VERSION="2.9"
#   echo "TORCH_VERSION is 2.8, setting TORCHVISION_VERSION to $TORCHVISION_VERSION and TORCHAUDIO_VERSION to $TORCHAUDIO_VERSION"
# else
#   echo "TORCH_VERSION is not 2.8, no changes to versions."
# fi

docker run --rm \
    -v "$(pwd)":/nunchaku \
    pytorch/manylinux2_28-builder:cuda${CUDA_VERSION} \
    bash -c "
    cd /nunchaku && \
    rm -rf build && \
    gcc --version && g++ --version && \
    ${PYTHON_ROOT_PATH}/bin/pip install --pre --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION//.} && \
    ${PYTHON_ROOT_PATH}/bin/pip install build ninja wheel setuptools && \
    export NUNCHAKU_INSTALL_MODE=ALL && \
    export NUNCHAKU_BUILD_WHEELS=1 && \
    export MAX_JOBS=${MAX_JOBS} && \
    ${PYTHON_ROOT_PATH}/bin/python -m build --wheel --no-isolation
    "
