import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from dfloat11 import compress_model

def main():
    parser = argparse.ArgumentParser("Compress OmniGen2 MLLM (Qwen2.5-VL) using DFloat11")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='The path to the OmniGen2 model (containing "mllm" folder) or direct path to MLLM checkpoint'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./OmniGen2-mllm-DF11',
        help='The path to save the compressed model'
    )
    parser.add_argument(
        '--save_single_file',
        action='store_true',
        help='Save the compressed model as a single .safetensors file'
    )
    parser.add_argument(
        '--check_correctness',
        action='store_true',
        help='Check the correctness of the compressed weights during compression'
    )
    parser.add_argument(
        '--block_range',
        type=int,
        nargs=2,
        default=(0, 100),
        help='The range of transformer blocks to compress (for parallel compression over multiple CPU cores)'
    )
    args = parser.parse_args()

    # Determine MLLM path
    import os
    mllm_path = args.model_path
    if os.path.isdir(os.path.join(args.model_path, "mllm")):
        mllm_path = os.path.join(args.model_path, "mllm")
    
    print(f"Loading MLLM from: {mllm_path}")

    # Load the Qwen2.5-VL model in bfloat16 precision
    # Use trust_remote_code=True same as in inference.py
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mllm_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Untie weights to avoid safetensors error about shared memory
    # safetensors.torch.save_file dies if tensors share memory.
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        print("Untying lm_head weights to avoid safetensors shared memory error...")
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    # Compress the model using DFloat11 compression
    # Pattern updated to match Qwen2.5-VL internal structure (model.language_model.layers...)
    compress_model(
        model=model,
        pattern_dict={
            r"model\.language_model\.layers\.\d+": (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ),
        },
        save_path=args.save_path,
        save_single_file=args.save_single_file, # Force single file to use state_dict keys (model.language_model...)
        check_correctness=args.check_correctness,
        block_range=args.block_range,
    )

if __name__ == "__main__":
    main()
