# %%
import dotenv
dotenv.load_dotenv("../.env")

import argparse
import json
import random
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import torch

from utils import chat, steering_config, process_batch_annotations
from messages import messages

# Model Configuration
MODEL_CONFIG = {
    # API Models: model_id to display name mapping
    'API_MODELS': {
        'gpt-4o': 'GPT-4o',
        'claude-3-opus': 'Claude-3-Opus',
        'claude-3-7-sonnet': 'Claude-3-7-Sonnet',
        'gemini-2-0-think': 'Gemini-2-0-Think',
        'gemini-2-0-flash': 'Gemini-2-0-Flash',
        'deepseek-v3': 'DeepSeek-V3',
        'deepseek-r1': 'DeepSeek-R1',
        'deepseek/deepseek-r1-distill-llama-8b': 'DeepSeek-R1-Llama-8B',
        'deepseek/deepseek-r1-distill-llama-70b': 'DeepSeek-R1-Llama-70B',
        'deepseek/deepseek-r1-distill-qwen-1.5b': 'DeepSeek-R1-Qwen-1.5B',
        'deepseek/deepseek-r1-distill-qwen-14b': 'DeepSeek-R1-Qwen-14B',
        'deepseek/deepseek-r1-distill-qwen-32b': 'DeepSeek-R1-Qwen-32B',
        'meta-llama/llama-3.1-8b-instruct': 'Llama-3.1-8B',
        'meta-llama/llama-3.3-70b-instruct': 'Llama-3.3-70B',
    },
    
    # Local Models: model_id to display name mapping
    'LOCAL_MODELS': {
        'Qwen/Qwen2.5-14B-Instruct': 'Qwen-2.5-14B',
        'Qwen/Qwen2.5-1.5B-Instruct': 'Qwen-2.5-1.5B',
        'Qwen/Qwen2.5-32B-Instruct': 'Qwen-2.5-32B',
    },
    
    # Thinking Models (for visualization grouping)
    'THINKING_MODELS': [
        'deepseek-r1-distill-llama-8b',
        'deepseek-r1-distill-llama-70b',
        'deepseek-r1-distill-qwen-1.5b',
        'deepseek-r1-distill-qwen-14b',
        'deepseek-r1-distill-qwen-32b',
        'claude-3-7-sonnet',
        'gemini-2-0-think',
        'deepseek-r1'
    ]
}

def get_model_display_name(model_id):
    """Convert model ID to display name using configuration"""
    # Check API models first
    if model_id in MODEL_CONFIG['API_MODELS']:
        return MODEL_CONFIG['API_MODELS'][model_id]
    
    # Check local models
    for local_id, display_name in MODEL_CONFIG['LOCAL_MODELS'].items():
        if local_id in model_id:
            return display_name
    
    # Default case: format the model ID
    return model_id.title()

def is_api_model(model_name):
    """Check if the model is an API model"""
    return model_name in MODEL_CONFIG['API_MODELS']

def is_thinking_model(model_name):
    """Check if the model is a thinking model"""
    # Convert model_name to lowercase for case-insensitive comparison
    model_name = model_name.lower()
    
    return model_name in MODEL_CONFIG['THINKING_MODELS']

def is_local_model(model_name):
    """Check if the model is a local model"""
    return model_name in MODEL_CONFIG['LOCAL_MODELS']

def extract_thinking_process(response_text):
    """Extracts the thinking process from between <think> and </think> tags."""
    think_start = 0
    think_end = len(response_text)
    if "\nassistant\n" in response_text:
        response_text = response_text.split("\nassistant\n")[1]
    if '<think>' in response_text:
        think_start = response_text.index('<think>') + len('<think>')
    if '</think>' in response_text:
        think_end = response_text.index('</think>')
    return response_text[think_start:think_end].strip()

# Parse arguments
parser = argparse.ArgumentParser(description="Compare reasoning abilities between models")
parser.add_argument("--model", type=str, default="gemini-2-0-think", 
                    help="Model to evaluate (e.g., 'gpt-4o', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B')")
parser.add_argument("--n_examples", type=int, default=10, 
                    help="Number of examples to use for evaluation")
parser.add_argument("--compute_from_json", action="store_true", 
                    help="Recompute scores from existing json instead of generating new responses")
parser.add_argument("--re_compute_scores", action="store_true", 
                    help="Recompute scores from existing json instead of generating new responses")
parser.add_argument("--re_annotate_responses", action="store_true", 
                    help="Re-annotate responses with new annotations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_tokens", type=int, default=100, help="Number of max tokens")
parser.add_argument("--skip_viz", action="store_true", help="Skip visualization at the end")
parser.add_argument("--ignore-common-labels", action="store_true", help="Ignore initializing and deduction labels")
args, _ = parser.parse_known_args()

# %%
def get_label_counts(thinking_process, labels, existing_annotated_response=None):
    if existing_annotated_response is None:
        # Get annotated version using chat function
        annotated_response = process_batch_annotations([thinking_process])[0]
    else:
        annotated_response = existing_annotated_response
    
    # Initialize token counts for each label
    label_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["?([\w-]+)"?\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    # Get tokens for the entire thinking process
    total = 0
    
    # Count tokens for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in labels:
            # Count tokens in this section
            label_counts[label] += 1
            total += 1
    
    return label_counts, annotated_response

def process_chat_response(message, model_name, model, tokenizer, labels):
    """Process a single message through chat function or model"""
    # Build prompts
    question = message["content"]
    no_thinking_prompt = f"""Please answer the following question:

{question}

It requires a few steps of reasoning. So first, think step by step, and only then give the final answer."""
    thinking_prompt = f"""Please answer the following question:

{question}"""

    if is_api_model(model_name):
        prompt = thinking_prompt if is_thinking_model(model_name) else no_thinking_prompt
        response = chat(prompt, model=model_name, max_tokens=args.max_tokens)
        print(response)

    elif is_local_model(model_name): 
        message["content"] = thinking_prompt if is_thinking_model(model_name) else no_thinking_prompt
        input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
                        
        with model.generate(
            {
                "input_ids": input_ids, 
                "attention_mask": (input_ids != tokenizer.pad_token_id).long()
            },
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ) as tracer:
            outputs = model.generator.output.save()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract thinking process
    thinking_process = extract_thinking_process(response)
    
    label_counts, annotated_response = get_label_counts(thinking_process, labels, existing_annotated_response=None)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_counts": label_counts,
        "annotated_response": annotated_response
    }

def _plot_comparison_subplot(ax, results_dict, labels, plot_type='counts', show_legend=True, hide_x_label=False, remove_first_group=False):
    """Helper to plot comparison bar chart on a given subplot axis."""
    
    _plot_labels = ["Total Sentences"] + labels if plot_type == 'counts' else [""] + labels

    model_names = list(results_dict.keys())
    means_dict = {}
    
    thinking_names = [name for name in model_names if is_thinking_model(name)]
    non_thinking_names = [name for name in model_names if not is_thinking_model(name)]
    
    if plot_type == 'counts':
        for model_name in model_names:
            total_sentences_per_response = [sum(ex['label_counts'].values()) for ex in results_dict[model_name]]
            avg_all_sentences = np.mean(total_sentences_per_response) if total_sentences_per_response else 0
            means_for_model = [avg_all_sentences]
            for label in labels:
                label_counts = [ex["label_counts"].get(label, 0) for ex in results_dict[model_name]]
                means_for_model.append(np.mean(label_counts))
            means_dict[model_name] = means_for_model
        
        model_avg_performance = {
            model_name: np.mean(means_dict[model_name][1:]) if len(means_dict[model_name]) > 1 else 0
            for model_name in model_names
        }
    else:  # fractions
        for model_name in model_names:
            model_label_fractions = {label: [] for label in labels}
            for result in results_dict[model_name]:
                label_counts = result.get('label_counts', {})
                total_sentences = sum(label_counts.values())
                
                if total_sentences > 0:
                    for label in labels:
                        fraction = label_counts.get(label, 0) / total_sentences
                        model_label_fractions[label].append(fraction)
                else:
                    for label in labels:
                        model_label_fractions[label].append(0)

            means_dict[model_name] = [0] + [np.mean(model_label_fractions[label]) for label in labels]

        model_avg_performance = {
            model_name: np.mean(means_dict[model_name]) 
            for model_name in model_names
        }
    
    if remove_first_group:
        plot_labels = _plot_labels[1:]
        for model_name in model_names:
            means_dict[model_name] = means_dict[model_name][1:]
    else:
        plot_labels = _plot_labels

    # Custom sorting logic based on model pairs
    # The pairs are defined to group distilled models with their instruction-tuned counterparts.
    model_pairs = [
        ('Deepseek-R1-Distill-Llama-70B', 'Llama-3.3-70B-Instruct'),
        ('Deepseek-R1-Distill-Llama-8B', 'Llama-3.1-8B-Instruct'),
        ('Deepseek-R1-Distill-Qwen-32B',  'Qwen2.5-32B-Instruct'),
        ('Deepseek-R1-Distill-Qwen-14B',  'Qwen2.5-14B-Instruct'),
        ('Deepseek-R1-Distill-Qwen-1.5B',  'Qwen2.5-1.5B-Instruct'),
    ]

    ordered_thinking = []
    ordered_non_thinking = []
    
    # This logic is case-insensitive and ignores hyphens for robust matching.
    available_thinking_map = {name.lower().replace('-', ''): name for name in thinking_names}
    available_non_thinking_map = {name.lower().replace('-', ''): name for name in non_thinking_names}

    for think_model, non_think_model in model_pairs:
        think_key = think_model.lower().replace('-', '')
        non_think_key = non_think_model.lower().replace('-', '')

        if think_key in available_thinking_map:
            ordered_thinking.append(available_thinking_map.pop(think_key))
        if non_think_key in available_non_thinking_map:
            ordered_non_thinking.append(available_non_thinking_map.pop(non_think_key))

    print(f"Ordered thinking: {ordered_thinking}")
    print(f"Ordered non-thinking: {ordered_non_thinking}")

    # Add any remaining models that weren't in pairs, sorted by performance
    remaining_thinking = sorted(available_thinking_map.values(), key=lambda x: model_avg_performance[x], reverse=True)
    print(f"Remaining thinking: {remaining_thinking}")
    remaining_non_thinking = sorted(available_non_thinking_map.values(), key=lambda x: model_avg_performance[x], reverse=True)
    print(f"Remaining non-thinking: {remaining_non_thinking}")
    
    thinking_names = ordered_thinking + remaining_thinking
    non_thinking_names = ordered_non_thinking + remaining_non_thinking

    x = np.arange(len(plot_labels))
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    thinking_colors = ['#1565C0', '#1976D2', '#1E88E5', '#64B5F6', '#BBDEFB']
    non_thinking_colors = ['#E65100', '#F57C00', '#FF9800', '#FFA726', '#FFE0B2']
    
    width = min(0.35, 0.8 / len(model_names))
    
    n_thinking = len(thinking_names)
    n_non_thinking = len(non_thinking_names)
    gap = width * 0.5 if n_thinking > 0 and n_non_thinking > 0 else 0
    
    for i, model_name in enumerate(thinking_names):
        color_idx = i if i < len(thinking_colors) else len(thinking_colors) - 1
        ax.bar(x + width * i, means_dict[model_name], width, 
               label=model_name, color=thinking_colors[color_idx], 
               alpha=0.85, edgecolor='black', linewidth=1)
    
    for i, model_name in enumerate(non_thinking_names):
        color_idx = i if i < len(non_thinking_colors) else len(non_thinking_colors) - 1
        ax.bar(x + width * (i + n_thinking) + gap, means_dict[model_name], width, 
               label=model_name, color=non_thinking_colors[color_idx], 
               alpha=0.85, edgecolor='black', linewidth=1)

    text_fontsize = 16
    if n_thinking > 0:
        for i in range(len(plot_labels)):
            if i == 0 and plot_type == 'fractions' and not remove_first_group:
                continue
            means = [means_dict[name][i] for name in thinking_names]
            group_mean = np.mean(means)
            group_center_x = x[i] + width * (n_thinking - 1) / 2
            max_h = max(means) if means else 0
            text = f"Avg: {group_mean:.1f}" if plot_type == 'counts' else f"Avg: {group_mean*100:.0f}%"
            ax.text(group_center_x, max_h + 0.02, text,
                    ha='center', va='bottom', fontsize=text_fontsize, color='black')

    if n_non_thinking > 0:
        for i in range(len(plot_labels)):
            if i == 0 and plot_type == 'fractions' and not remove_first_group:
                continue
            means = [means_dict[name][i] for name in non_thinking_names]
            group_mean = np.mean(means)
            group_center_x = x[i] + width * (n_thinking + (n_non_thinking - 1) / 2) + gap
            max_h = max(means) if means else 0
            text = f"Avg: {group_mean:.1f}" if plot_type == 'counts' else f"Avg: {group_mean*100:.0f}%"
            ax.text(group_center_x, max_h + 0.02, text,
                    ha='center', va='bottom', fontsize=text_fontsize, color='black')

    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # Adjust xlim to remove extra padding
    if plot_labels:
        num_models = len(thinking_names) + len(non_thinking_names)
        last_bar_pos = width * (num_models - 1) + gap
        ax.set_xlim(-(width + 0.05), len(plot_labels) - 1 + last_bar_pos + width + 0.05)

    ymax = max([max(means) for means in means_dict.values()]) if means_dict else 1
    
    if plot_type == 'counts':
        ax.set_ylim(0, ymax * 1.1)
        ax.set_ylabel('Avg Sentence Count', fontsize=16)
    else:
        ax.set_ylim(0, ymax * 1.1)
        ax.set_ylabel('Avg Sentence Fraction', fontsize=16)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))

    if not hide_x_label:
        ax.set_xlabel("Behavioral patterns", fontsize=16)
    else:
        ax.set_xlabel("")
    
    tick_pos = x + (n_thinking * width + n_non_thinking * width + gap) / 2 - width / 2
    ax.set_xticks(tick_pos)
    formatted_labels = [label.replace('-', ' ').title() for label in plot_labels]
    formatted_labels = [label.replace(' ', '\n') for label in formatted_labels]
    ax.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=16)

    ax.tick_params(axis='y', labelsize=16)
    
    for label_idx in x:
        group_separator = label_idx + width * len(thinking_names) + gap / 2
        ax.axvline(x=group_separator, color='gray', linestyle='--', alpha=0.3, zorder=0)
    
    if show_legend:
        ax.legend(fontsize=16, frameon=True, framealpha=1, 
                  edgecolor='black', bbox_to_anchor=(0.31, 1), 
                  loc='upper center', ncol=2)

def plot_comparison_counts(results_dict, labels):
    """Plot comparison between multiple models' results"""
    os.makedirs('results/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 7))
    _plot_comparison_subplot(ax, results_dict, labels, plot_type='counts', show_legend=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/figures/reasoning_comparison_all_models_counts.pdf',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

def plot_comparison_fractions(results_dict, labels):
    """Plot comparison between multiple models' results as fractions"""
    os.makedirs('results/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 7))
    _plot_comparison_subplot(ax, results_dict, labels, plot_type='fractions', show_legend=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/figures/reasoning_comparison_all_models_fractions.pdf',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

def plot_comparison_counts_and_fractions(results_dict, labels):
    """Plots counts and fractions in two subplots."""
    os.makedirs('results/figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    
    fig.suptitle("Comparison of Reasoning Patterns Across Models", fontsize=20, y=0.98)
    
    _plot_comparison_subplot(ax1, results_dict, labels, plot_type='fractions', show_legend=True, hide_x_label=True)
    ax1.tick_params(labelbottom=True, labelsize=16)
    _plot_comparison_subplot(ax2, results_dict, labels, plot_type='counts', show_legend=False, hide_x_label=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('results/figures/reasoning_comparison_all_models_counts_and_fractions.pdf',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

def plot_comparison_fractions_with_total_count(results_dict, labels):
    """Plots fractions and total sentence count in two subplots."""
    os.makedirs('results/figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [5, 1]})
    
    fig.suptitle("Comparison of Reasoning Patterns Fractions and Total Sentences", fontsize=20, y=0.95)
    
    _plot_comparison_subplot(ax1, results_dict, labels, plot_type='fractions', show_legend=True, hide_x_label=True, remove_first_group=True)
    
    _plot_comparison_subplot(ax2, results_dict, [], plot_type='counts', show_legend=False, hide_x_label=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/figures/reasoning_comparison_all_models_fractions_with_total_count.pdf',
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# %% Parameters
n_examples = args.n_examples
random.seed(args.seed)
model_name = args.model
compute_from_json = args.compute_from_json
re_compute_scores = args.re_compute_scores
re_annotate_responses = args.re_annotate_responses

# Get a shorter model_id for file naming
model_id = model_name.split('/')[-1].lower()

labels = list(list(steering_config.values())[0].keys())
labels.append('initializing')
labels.append('deduction')

if args.ignore_common_labels:
    labels.remove('initializing')
    labels.remove('deduction')

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %% Load model and evaluate
model = None
tokenizer = None

results = []

# %%
if compute_from_json:
    # Load existing results and recompute scores
    print(f"Loading existing results for {model_name}...")
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'r') as f:
        results = json.load(f)
    
    # Re-compute label fractions from loaded results
    if re_compute_scores:
        print("Re-computing label counts for all loaded responses...")
        for result in tqdm(results, desc="Re-computing scores from JSON"):
            thinking_process = result.get('thinking_process', '')
            if thinking_process == '' or thinking_process == 'None':
                thinking_process = extract_thinking_process(result.get('response', ''))
                result['thinking_process'] = thinking_process

            assert thinking_process != '', f"**ERROR** No thinking process found for {result['response']}"
                
            label_counts, annotated_response = get_label_counts(
                thinking_process, 
                labels,
                existing_annotated_response=result.get('annotated_response', None) if not re_annotate_responses else None
            )
            if 'label_fractions' in result:
                del result['label_fractions']
            result['label_counts'] = label_counts
            result['annotated_response'] = annotated_response
        # Save updated results
        with open(f'results/vars/reasoning_comparison_{model_id}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Re-computed {len(results)} examples for {model_name}")
else:
    # Run new evaluation
    print(f"Running evaluation for {model_name}...")

    if not is_api_model(model_name):
        # Load model using the utils function
        import utils
        print(f"Loading model {model_name}...")
        model, tokenizer, _ = utils.load_model_and_vectors(compute_features=False, model_name=model_name)
    
    # Randomly sample evaluation examples
    eval_indices = random.sample(range(len(messages)), n_examples)
    selected_messages = [messages[i] for i in eval_indices]
    
    # Process responses
    for message in tqdm(selected_messages, desc=f"Processing examples for {model_name}"):
        # Process response
        result = process_chat_response(message, model_name, model, tokenizer, labels)
        results.append(result)
    
    # Save results
    with open(f'results/vars/reasoning_comparison_{model_id}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up model to free memory
    if model is not None:
        del model
        torch.cuda.empty_cache()

# %% Generate visualization with all available models
if not args.skip_viz:
    # Load results for all models
    all_results = {}
    result_files = glob.glob('results/vars/reasoning_comparison_*.json')

    # Filter Llama 8B and Qwen Math 1.5B: the responses are too messy
    # result_files = [file for file in result_files if 'llama-8b' not in file and 'llama-3.1-8b' not in file]

    print(f"Found {len(result_files)} model results for visualization")
    
    for file_path in result_files:
        model_id = os.path.basename(file_path).replace('reasoning_comparison_', '').replace('.json', '')
        display_name = get_model_display_name(model_id)
        
        with open(file_path, 'r') as f:
            all_results[display_name] = json.load(f)
    
    # Generate visualization with all models
    if all_results:
        plot_comparison_counts(all_results, labels)
        plot_comparison_fractions(all_results, labels)
        plot_comparison_counts_and_fractions(all_results, labels)
        plot_comparison_fractions_with_total_count(all_results, labels)
    else:
        print("No results found for visualization")

# %%
