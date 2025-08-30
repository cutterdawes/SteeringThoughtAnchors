# %%
import torch
import matplotlib.pyplot as plt
import argparse
from . import utils
import seaborn as sns

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
args, _ = parser.parse_known_args()

model, tokenizer, feature_vectors = utils.load_model_and_vectors(model_name=args.model)

model_id = args.model.split("/")[-1].lower()

# %%
def plot_cosine_similarity_heatmap(feature_vectors, model_id):
    labels = ["initializing", "backtracking", "uncertainty-estimation", "adding-knowledge", "deduction"]
    n_labels = len(labels)
    
    # Create similarity matrix
    similarity_matrix = torch.zeros((n_labels, n_labels))

    for i, label_1 in enumerate(labels):
        for j, label_2 in enumerate(labels):
            
            if label_1 in utils.steering_config[args.model]:
                layer_idx_1 = utils.steering_config[args.model][label_1]["vector_layer"]
            else:
                layer_idx_1 = utils.steering_config[args.model][list(utils.steering_config[args.model].keys())[0]]["vector_layer"]
            
            if label_2 in utils.steering_config[args.model]:
                layer_idx_2 = utils.steering_config[args.model][label_2]["vector_layer"]
            else:
                layer_idx_2 = utils.steering_config[args.model][list(utils.steering_config[args.model].keys())[0]]["vector_layer"]

            similarity_matrix[i, j] = torch.cosine_similarity(
                feature_vectors[label_1][layer_idx_1], 
                feature_vectors[label_2][layer_idx_2], 
                dim=-1
            )
                
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, label='Cosine Similarity', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Remove grid lines
    plt.grid(False)
    
    # Add labels and text annotations
    plt.xticks(range(n_labels), labels, rotation=45, ha='right')
    plt.yticks(range(n_labels), labels)
    
    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            value = similarity_matrix[i, j].item()
            # Choose text color based on similarity value
            text_color = 'white' if abs(value) > 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center',
                    color=text_color,
                    fontsize=9,
                    fontweight='bold')
    
    plt.title('Cosine Similarity Between Feature Vectors', pad=20)
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig(f'results/figures/cosine_similarity_heatmap_{model_id}.pdf', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()

# Plot the aggregated heatmap
plot_cosine_similarity_heatmap(feature_vectors, model_id=model_id)

# %%
def plot_layer_similarities_general(feature_vectors, comparison_matrix, labels, title, model_id):
    """
    Plot layer-wise similarities between feature vectors and a comparison matrix.
    
    Args:
        feature_vectors: Dictionary of feature vectors
        comparison_matrix: Matrix to compare against (e.g., unembed or embed weights)
        labels: List of labels to plot
        title: Title for the plot
        model_id: Model identifier for saving the plot
    """
    num_layers = len(feature_vectors[list(feature_vectors.keys())[0]])
    
    # Define a professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(12, 6))
    
    for idx, label in enumerate(labels):
        similarities = []
        for layer_idx in range(num_layers):
            sim = torch.cosine_similarity(feature_vectors[label][layer_idx].to(comparison_matrix.device), 
                                        comparison_matrix, dim=-1).max().item()
            similarities.append(sim)
        
        plt.plot(range(num_layers), similarities, 
                marker='o', 
                label=label,
                color=colors[idx],
                linewidth=2,
                markersize=6,
                alpha=0.8)
    
    plt.xlabel('Layer Index', labelpad=10)
    plt.ylabel('Cosine Similarity', labelpad=10)
    plt.title(title, pad=20)
    
    # Improve legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Add grid with improved styling
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(-0.5, num_layers - 0.5)
    plt.ylim(min(similarities) - 0.05, max(similarities) + 0.05)
    
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig(f'results/figures/layer_similarities_{title.lower().replace(" ", "_")}_{model_id}.pdf', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()

def plot_layer_similarities(feature_vectors, model, model_id):
    unembed = model.lm_head.weight.data
    embed = model.model.embed_tokens.weight.data
    labels = ["backtracking", "uncertainty-estimation", "adding-knowledge", "deduction", "initializing", "example-testing"]
    
    # Plot unembedding similarities
    plot_layer_similarities_general(
        feature_vectors=feature_vectors,
        comparison_matrix=unembed,
        labels=labels,
        title='Feature Vector Unembedding Similarities Across Layers',
        model_id=model_id
    )
    
    # Plot embedding similarities
    plot_layer_similarities_general(
        feature_vectors=feature_vectors,
        comparison_matrix=embed,
        labels=labels,
        title='Feature Vector Embedding Similarities Across Layers',
        model_id=model_id
    )

# Plot the layer-wise similarities
plot_layer_similarities(feature_vectors, model, model_id)

# %%
