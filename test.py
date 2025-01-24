import os
import sys
import argparse
import torch
from transformers import ViTForImageClassification
from utils import preprocess_image, save_overlay_image
from gmar import GMAR

def gmar_test(input, output, pretrained, norm):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViTForImageClassification.from_pretrained(pretrained)
    model = model.to(device)
    model.eval()

    image_tensor = preprocess_image(input)

    target_layer = model.vit.encoder.layer[-1].layernorm_before
    gmar = GMAR(model, target_layer)

    # Generate Multi-head Gradient Residual Attention Rollout
    if norm == "l1":
        # l1 norm
        head_weights = gmar.weight_generate(image_tensor.to(device), "l1")
    elif norm == "l2":
        # l1 norm
        head_weights = gmar.weight_generate(image_tensor.to(device), "l2")
    else:
        print(f"Error: Invalid norm type '{norm}'. Supported types are 'l1' and 'l2'.", file=sys.stderr)
        sys.exit(1)  # Exit with error code 1
    
    gmar_rollout = gmar.compute_attention_rollout_with_head_weights(image_tensor.to(device), head_weights)

    save_overlay_image(input, output, gmar_rollout)


def main():
    parser = argparse.ArgumentParser(description='GMAR')
    
    parser.add_argument('--input', type=str, default="./input/test.JPEG", help='input path')
    parser.add_argument('--output', type=str, default="./output/gmar_output.png", help='output path')
    parser.add_argument('--pretrained', type=str, required=True, help='pretrained model(HuggingFace Model)')
    parser.add_argument('--norm', type=str, default="l1", help='regularization type')
    
    args = parser.parse_args()
    
    gmar_test(args.input, args.output, args.pretrained, args.norm)

if __name__ == "__main__":
    main()