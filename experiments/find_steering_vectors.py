import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import NNsight
import json
import os
import sys
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_model_and_vectors 

def find_thought_anchor_steering_vectors(model_name: str, data_path: str, output_dir: str):
    """
    Implements Experiment #2: Finding thought anchor steering vectors.

    Args:
        model_name (str): The name of the pre-trained model to use (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").
        data_path (str): Path to the dataset containing (prompt, CoT, answer, activations) tuples with thought anchor annotations.
        output_dir (str): Directory to save the computed steering vectors.
    """
    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer, _ = load_model_and_vectors(model_name=model_name, device="mps")
    print("Model and tokenizer loaded.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load the dataset ---
    # This dataset is expected to contain:
    # - original prompts
    # - generated CoTs
    # - answers
    # - activations (or a way to re-generate them)
    # - annotations for thought anchor sentences within the CoTs
    print(f"Loading dataset from {data_path}...")
    try:
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} data points.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure Experiment #1 (Preliminaries) is completed.")
        return

    steering_vectors = {}

    # Iterate through the dataset to find thought anchors and compute steering vectors
    for i, data_point in enumerate(dataset):
        prompt = data_point['prompt']
        cot = data_point['cot']
        thought_anchor_sentence = data_point.get('thought_anchor_sentence') # Assuming annotation exists

        if not thought_anchor_sentence:
            print(f"Skipping data point {i} due to missing thought anchor annotation.")
            continue

        print(f"Processing data point {i+1}: Prompt='{prompt[:50]}...', Thought Anchor='{thought_anchor_sentence[:50]}...'")

        # --- Step 2: Extract a_anchor (average activation of the thought anchor sentence) ---
        # This is a placeholder. Actual implementation would involve:
        # 1. Tokenizing the thought_anchor_sentence.
        # 2. Running the model with nnsight to capture activations at the relevant layer(s).
        # 3. Averaging activations across tokens of the thought anchor sentence.
        # For now, we'll use a dummy tensor.
        
        # Placeholder for actual activation extraction
        # Example: Using nnsight to get activations
        # with NNsight(model, tokenizer) as model_nnsight:
        #     with model_nnsight.generate(inputs=thought_anchor_sentence, max_new_tokens=1) as _:
        #         # Assuming we want activations from the last hidden state of a specific layer
        #         # The 'most causally relevant layer' needs to be determined (e.g., via patching experiments)
        #         # For demonstration, let's pick a dummy layer index
        #         layer_idx = 10 # This should be determined by Experiment #2.1.ii
        #         a_anchor_activations = model_nnsight.model.layers[layer_idx].output.save()
        # a_anchor = a_anchor_activations.value.mean(dim=1).squeeze() # Average over tokens and batch

        # Dummy a_anchor for demonstration
        a_anchor = torch.randn(model.config.hidden_size) # Assuming hidden_size is the dimension of activations

        # --- Step 3: Generate a_counter (average activations of sampled counterfactual replacement sentences) ---
        # This is also a placeholder and is the most complex part. It would involve:
        # 1. Resampling just before the thought anchor sentence.
        # 2. Conditioning on a minimum threshold of dissimilarity to the thought anchor.
        # 3. Generating multiple counterfactual sentences.
        # 4. Extracting activations for each counterfactual.
        # 5. Averaging these activations.
        # For now, we'll use a dummy tensor.

        # Placeholder for actual counterfactual generation and activation extraction
        # Example:
        # counterfactual_sentences = generate_counterfactuals(model, tokenizer, prompt, thought_anchor_sentence)
        # a_counter_activations_list = []
        # for cf_sentence in counterfactual_sentences:
        #     with NNsight(model, tokenizer) as model_nnsight:
        #         with model_nnsight.generate(inputs=cf_sentence, max_new_tokens=1) as _:
        #             a_cf_activations = model_nnsight.model.layers[layer_idx].output.save()
        #     a_counter_activations_list.append(a_cf_activations.value.mean(dim=1).squeeze())
        # a_counter = torch.stack(a_counter_activations_list).mean(dim=0)

        # Dummy a_counter for demonstration
        a_counter = torch.randn(model.config.hidden_size)

        # --- Step 4: Compute the steering vector v = a_anchor - a_counter ---
        steering_vector = a_anchor - a_counter

        # --- Step 5: Normalize the steering vector (optional, but recommended) ---
        steering_vector = torch.nn.functional.normalize(steering_vector, dim=0)

        # Store the steering vector, perhaps associated with the thought anchor or prompt ID
        steering_vectors[f"prompt_{i}_anchor_vector"] = steering_vector.tolist()
        print(f"Computed steering vector for data point {i+1}.")

    # Save the computed steering vectors
    output_file = os.path.join(output_dir, f"steering_vectors_{model_name.replace('/', '-')}.json")
    with open(output_file, 'w') as f:
        json.dump(steering_vectors, f, indent=4)
    print(f"All steering vectors saved to {output_file}")

if __name__ == "__main__":
    # Example usage:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # This data_path should point to the output of Experiment #1, which includes thought anchor annotations
    data_path = "/Users/cutterdawes/Desktop/Research/SteeringThoughtAnchors/generated_data/generated_data_annotated_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B.json"
    output_dir = "/Users/cutterdawes/Desktop/Research/SteeringThoughtAnchors/steering-thinking-llms/train-steering-vectors/results/steering_vectors"

    find_thought_anchor_steering_vectors(model_name, data_path, output_dir)
