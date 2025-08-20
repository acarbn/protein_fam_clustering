"""
Download and save ESM protein language models locally.

This script allows you to download a specified ESM model from Hugging Face
and store it on your local machine for offline use. It saves both the tokenizer
and the model weights in a directory under ./model/. The model is saved in 
float16 precision using safetensors for faster loading and reduced disk space.

Usage:
    python3 -m src.download_esm_model <model_name>

Example:
    python3 -m src.download_esm_model esm2_t6_8M_UR50D
"""

from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import os

# Parse the model name
parser = argparse.ArgumentParser(description="Process a model file.")

parser.add_argument(
        "model_name",
        type=str,
        help="Model name given to download "
    )
args = parser.parse_args()

# Create local directories
os.makedirs("./model", exist_ok=True)

model_name="facebook/"+args.model_name
local_dir = "./model/" + model_name.split("/")[-1].replace("UR50D", "local")

# Download & save tokenizer (small, fast)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)
print("Tokenizer is loaded.")

# Download model in float16 with safetensors (faster + smaller)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # half-precision
    device_map="auto",           # automatically put layers on GPU if available
    use_safetensors=True         # faster load format
)
print("The model is loaded.")

# Save locally for future use
model.save_pretrained(local_dir)

print(args.model_name, " model is downloaded under ", local_dir)
