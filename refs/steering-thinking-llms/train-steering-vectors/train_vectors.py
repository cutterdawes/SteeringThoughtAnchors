# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer
import torch
import re
from nnsight import NNsight
from collections import defaultdict
import os
import random
import json
import utils
from utils import process_saved_responses_batch
import math
import gc
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description="Generate annotations and train steering vectors for model reasoning")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--save_every", type=int, default=1, 
                    help="Save checkpoints every n batches")
parser.add_argument("--responses_path", type=str, default=None,
                    help="Path to JSON file containing responses")
parser.add_argument("--n_samples", type=int, default=100,
                    help="Number of samples to process")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for processing messages")
args, _ = parser.parse_known_args()

# python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n_samples 500 --max_tokens 1000 --batch_size 4 --save_every 1 --load_from_json --update_annotation

# %%
def extract_thinking_process(response):
    """Extract thinking process from response"""
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    return response[think_start:think_end].strip()

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, index):
    """Update mean vectors for overall and individual labels"""
    # Calculate overall thinking section boundaries
    all_positions = [pos for positions in label_positions.values() for pos in positions]
    if all_positions:
        min_pos = min(start for start, _ in all_positions)
        max_pos = max(end for _, end in all_positions)
        
        # Update overall mean
        overall_vectors = layer_outputs[:, min_pos:max_pos].mean(dim=1)
        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (overall_vectors - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

        if torch.isnan(mean_vectors['overall']['mean']).any():
            print(f"NaN in mean_vectors['overall']['mean'] at index {index}")
    
    # Update individual labels
    for label, positions in label_positions.items():
        for position in positions:
            start, end = position
            vectors = layer_outputs[:, start-1:min(end-1, start+10)].mean(dim=1)
            if torch.isnan(vectors).any():
                print(f"NaN in mean_vectors['{label}']['mean'] at index {index}")
                print(f"Layer outputs: {layer_outputs[:, start-1:min(end-1, start+2)]}")
                print(f"Layer outputs shape: {layer_outputs.shape}")
                print(f"Positions: {positions}")
                print(f"Index: {index}")
                print(f"Label: {label}")
                print(f"Start: {start}")
                print(f"End: {end}")
                print(f"Vectors: {vectors}")
                print(f"Current count: {mean_vectors[label]['count']}")
                print(f"Current mean: {mean_vectors[label]['mean']}")
                
                continue
            
            current_count = mean_vectors[label]['count']
            current_mean = mean_vectors[label]['mean']
            mean_vectors[label]['mean'] = current_mean + (vectors - current_mean) / (current_count + 1)
            mean_vectors[label]['count'] += 1

# %% Main execution
model_name = args.model

# Create directories
os.makedirs('results/vars', exist_ok=True)

save_every = args.save_every
save_path = f"results/vars/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

# Default responses path if not provided
responses_json_path = args.responses_path or f"results/vars/responses_{model_name.split('/')[-1].lower()}.json"

if not os.path.exists(responses_json_path):
    raise FileNotFoundError(f"Responses file not found at {responses_json_path}. Please generate responses first.")

# Load model using utils function
print(f"Loading model {model_name}...")
model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name, load_in_8bit=args.load_in_8bit)

mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# Load existing responses
print(f"Loading responses from {responses_json_path}")
with open(responses_json_path, 'r') as f:
    responses_data = json.load(f)

random.seed(args.seed)
random.shuffle(responses_data)

# Process in batches to update annotations and vectors
num_batches = math.ceil(min(len(responses_data), args.n_samples) / args.batch_size)

for batch_idx in tqdm(range(num_batches), desc="Processing responses"):
    start_idx = batch_idx * args.batch_size
    end_idx = min(start_idx + args.batch_size, min(len(responses_data), args.n_samples))
    
    batch_responses = responses_data[start_idx:end_idx]
    thinking_processes = [data["thinking_process"] for data in batch_responses]
    batch_full_responses = [data["full_response"] for data in batch_responses]
    batch_indices = list(range(start_idx, end_idx))
    
    # Generate annotations
    annotated_responses = utils.process_batch_annotations(thinking_processes)
    
    # Update annotation fields in the JSON
    for i, (response_data, annotated) in enumerate(zip(batch_responses, annotated_responses)):
        responses_data[start_idx + i]["annotated_thinking"] = annotated
    
    # Process saved responses to calculate vectors
    batch_layer_outputs = process_saved_responses_batch(batch_full_responses, tokenizer, model)
    
    # Update vectors based on annotations
    for i, (response_data, layer_outputs) in enumerate(zip(batch_responses, batch_layer_outputs)):
        if annotated_responses[i]:  # Use the new annotations
            label_positions = utils.get_label_positions(annotated_responses[i], response_data["full_response"], tokenizer)
            update_mean_vectors(mean_vectors, layer_outputs, label_positions, batch_indices[i])
            
    del batch_layer_outputs
    
    if batch_idx % save_every == 0:
        # Save updated JSON
        with open(responses_json_path, "w") as f:
            json.dump(responses_data, f, indent=2)
        # Save updated vectors
        save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
        torch.save(save_dict, save_path)

    torch.cuda.empty_cache()
    gc.collect()

# Save final results
with open(responses_json_path, "w") as f:
    json.dump(responses_data, f, indent=2)
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final annotations and vectors")