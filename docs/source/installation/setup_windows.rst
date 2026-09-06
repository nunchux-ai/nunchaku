Windows Setup Guide
===================

Environment Setup
-----------------

1. Install Cuda
^^^^^^^^^^^^^^^^

Download and install the latest CUDA Toolkit from the official `NVIDIA CUDA Downloads <download_cuda_>`_.
After installation, verify the installation:

.. code-block:: bat

   nvcc --version

2. Install Visual Studio C++ Build Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download from the official `Visual Studio Build Tools page <visual_studio_>`_.
During installation, select the following workloads:

- **Desktop development with C++**
- **C++ tools for Linux development**

3. Install Git
^^^^^^^^^^^^^^

Download Git from `this link <download_git_win_>`_ and follow the installation steps.

4. (Optional) Install Conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda helps manage Python environments. You can install either Anaconda or Miniconda from the `official site <download_anaconda_>`_.

5. (Optional) Install ComfyUI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may have various ways to install ComfyUI.
For example, you can use ComfyUI CLI.
Once Python is installed, you can install ComfyUI via the CLI:

.. code-block:: bat

   pip install comfy-cli
   comfy install

To launch ComfyUI:

.. code-block:: bat

   comfy launch

Installing Nunchaku on Windows
-------------------------------

Step 1: Identify Your Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure correct installation, you need to find the Python interpreter used by ComfyUI.
Launch ComfyUI and look for this line in the log:

.. code-block:: text

   ** Python executable: G:\ComfyuI\python\python.exe

Then verify the Python version and installed PyTorch version:

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" --version
   "G:\ComfyuI\python\python.exe" -m pip show torch

Step 2: Install PyTorch (≥2.5) if you haven't
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install PyTorch appropriate for your setup:

- **For most users**:

  .. code-block:: bat

     "G:\ComfyuI\python\python.exe" -m pip install torch==2.6 torchvision==0.21 torchaudio==2.6

- **For RTX 50-series GPUs** (requires PyTorch ≥2.7 with CUDA 12.8):

  .. code-block:: bat

     "G:\ComfyuI\python\python.exe" -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

Step 3: Install Nunchaku
^^^^^^^^^^^^^^^^^^^^^^^^^

Option 1: Use ``install_wheel.json`` Workflow in ComfyUI
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With `ComfyUI-nunchaku <github_comfyui-nunchaku_>`_  v0.3.2+,
you can install Nunchaku using the provided :ref:`comfyui_nunchaku:install-wheel-json` workflow directly in ComfyUI.

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/install_wheel.png

Option 2: Manually Install Prebuilt Wheels
"""""""""""""""""""""""""""""""""""""""""""

You can install Nunchaku wheels from one of the following:

- `GitHub Releases <github_nunchaku_releases_>`_
- `Hugging Face <hf_nunchaku_>`_
- `ModelScope <ms_nunchaku_>`_

Example (for Python 3.11 + PyTorch 2.7):

.. code-block:: bat

   "G:\ComfyUI\python\python.exe" -m pip install https://github.com/nunchux-ai/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl

To verify the installation:

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" -c "import nunchaku"

You can also run a test (requires a Hugging Face token for downloading the models):

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" -m huggingface-cli login
   "G:\ComfyuI\python\python.exe" -m nunchaku.test

Option 3: Build Nunchaku from Source
""""""""""""""""""""""""""""""""""""

Please use CMD instead of PowerShell for building.

Step 1: Install Build Tools

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" -m pip install ninja setuptools wheel build

Step 2: Clone the Repository

.. code-block:: bat

   git clone --recurse-submodules https://github.com/nunchux-ai/nunchaku.git
   cd nunchaku

Step 3: Set Up Visual Studio Environment

Locate the ``VsDevCmd.bat`` script on your system. Example path:

.. code-block:: text

   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat

Then run:

.. code-block:: bat

   "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
   set DISTUTILS_USE_SDK=1

Step 4: Build Nunchaku

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" setup.py develop

Verify with:

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" -c "import nunchaku"

You can also run a test (requires a Hugging Face token):

.. code-block:: bat

   "G:\ComfyuI\python\python.exe" -m huggingface-cli login
   "G:\ComfyuI\python\python.exe" -m nunchaku.test

(Optional) Step 5: Building wheel for Portable Python

If building directly with portable Python fails:

.. code-block:: bat

   set NUNCHAKU_INSTALL_MODE=ALL
   "G:\ComfyuI\python\python.exe" python -m build --wheel --no-isolation

Use Nunchaku in ComfyUI
-----------------------

1. Install the Plugin
^^^^^^^^^^^^^^^^^^^^^

Clone the `ComfyUI-nunchaku <github_comfyui-nunchaku_>`_ plugin into the ``custom_nodes`` folder:

.. code-block:: bat

   cd ComfyUI/custom_nodes
   git clone https://github.com/nunchux-ai/ComfyUI-nunchaku.git

Alternatively, install it using `ComfyUI-Manager <github_comfyui-manager_>`_ or `comfy-cli <github_comfy-cli_>`_.

2. Download Models
^^^^^^^^^^^^^^^^^^

**Standard FLUX.1-dev Models**

Start by downloading the standard `FLUX.1-dev text encoders <https://huggingface.co/comfyanonymous/flux_text_encoders>`__
and `VAE <https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors>`__.
You can also optionally download the original `BF16 FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors>`__ model. An example command:

.. code-block:: bat

   huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
   huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir models/diffusion_models

**Nunchaku 4-bit FLUX.1-dev Models**

Next, download the Nunchaku 4-bit models to ``models/diffusion_models``:

- For **50-series GPUs**, use the `FP4 model <hf_nunchaku-flux1-dev-fp4_>`_.
- For **other GPUs**, use the `INT4 model <hf_nunchaku-flux1-dev-int4_>`_.

**(Optional): Download Sample LoRAs**

You can test with some sample LoRAs like `FLUX.1-Turbo <hf_lora_flux-turbo_>`_ and `Ghibsky <hf_lora_ghibsky_>`_. Place these files in the ``models/loras`` directory:

.. code-block:: bat

   huggingface-cli download alimama-creative/FLUX.1-Turbo-Alpha diffusion_pytorch_model.safetensors --local-dir models/loras
   huggingface-cli download aleksa-codes/flux-ghibsky-illustration lora.safetensors --local-dir models/loras

3. Set Up Workflows
^^^^^^^^^^^^^^^^^^^

To use the official workflows, download them from the `ComfyUI-nunchaku <github_comfyui-nunchaku_>`_ and place them in your ``ComfyUI/user/default/workflows`` directory. The command can be:

.. code-block:: bat

   # From the root of your ComfyUI folder
   cp -r custom_nodes/ComfyUI-nunchaku/example_workflows user/default/workflows/nunchaku_examples

You can now launch ComfyUI and try running the example workflows.
