#!/bin/bash
source /home/olegk/venv/vllm/bin/activate
cd /home/olegk/Nikola/src/imagegen
python ImageEditServer.py \
    --port 4500 \
    --model /home/olegk/Nikola/models/Z-Image-Turbo \
    --optimized-model /home/olegk/Nikola/models/nunchaku-z-image-turbo/svdq-fp4_r32-z-image-turbo.safetensors \
    --backend zimage \
    --steps 8 \
    --guidance-scale 0.0 \
    --uma
