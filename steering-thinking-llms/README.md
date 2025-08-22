# Understanding Reasoning in Thinking Language Models via Steering Vectors

This repository contains the source code for the paper ["Understanding Reasoning in Thinking Language Models via Steering Vectors"](https://arxiv.org/abs/2506.18167).

We provide an interactive [Demo Colab](https://colab.research.google.com/drive/1CXadiO7XZP216QvIyUUfhnJzKgz-EMew) where you can experiment with our steering vectors and observe their effects on model reasoning behavior.


## Overview

This project investigates how steering vectors can be used to understand and control reasoning behavior in large language models, particularly focusing on "thinking" models that exhibit chain-of-thought reasoning patterns. The codebase provides tools for:

- Training steering vectors from model activations
- Evaluating the effects of steering on model reasoning
- Analyzing layer-wise attribution of steering effects
- Comparing reasoning patterns across different model architectures

## Project Structure

- **`train-steering-vectors/`** - Generate model responses and train steering vectors from activations
- **`steering/`** - Evaluate how steering vectors affect model reasoning behavior
- **`vector-layer-attribution/`** - Analyze steering effects across different model layers
- **`compare-base-reasoning/`** - Compare reasoning patterns between steered and unsteered models
- **`messages/`** - Input prompts and evaluation data
- **`utils/`** - Shared utilities and helper functions

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for model inference)
- Conda or Miniconda

### Installation

1. Clone the repository:
```bash
git clone git@github.com:cvenhoff/steering-thinking-llms.git
cd steering-thinking-llms
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate stllms_env
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Set up environment variables (optional):
Create a `.env` file in the root directory with any required API keys or configuration.

## Usage

### 1. Training Steering Vectors

The training pipeline consists of two main steps:

1. **Generate baseline responses**:
```bash
cd train-steering-vectors
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 500 --max_tokens 1000
```

2. **Train steering vectors**:
```bash
python train_vectors.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_samples 500
```

Or run the complete pipeline:
```bash
bash run.sh
```

### 2. Evaluating Steering Effects

Evaluate how steering vectors affect model reasoning:

```bash
cd steering
python evaluate_steering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n_examples 50
```

Or run all evaluations:
```bash
bash run.sh
```

### 3. Layer-wise Analysis

Analyze how steering effects vary across model layers:

```bash
cd vector-layer-attribution
python analyze_layer_effects.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
bash run.sh
```

### 4. Reasoning Comparison

Compare reasoning patterns between different conditions:

```bash
cd compare-base-reasoning
python compare_reasoning.py
```

## Supported Models

The codebase has been tested with the following models:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

## Key Parameters

- `--model`: Model identifier (HuggingFace format)
- `--n_samples`: Number of training samples to process
- `--max_tokens`: Maximum tokens to generate per response
- `--batch_size`: Batch size for processing (adjust based on GPU memory)
- `--seed`: Random seed for reproducibility

## Output

Results are saved in the `results/` directory:
- `figures/`: Generated plots and visualizations
- `vars/`: Saved model states, steering vectors, and processed data

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{venhoff2025understanding,
  title={Understanding Reasoning in Thinking Language Models via Steering Vectors},
  author={Venhoff, Constantin and Arcuschin, Iván and Torr, Philip and Conmy, Arthur and Nanda, Neel},
  booktitle={Workshop on Reasoning and Planning for Large Language Models},
  year={2025},
  url={https://arxiv.org/abs/2506.18167}
}
```

## Dependencies

Key dependencies include:
- PyTorch
- Transformers (HuggingFace)
- nnsight (for model introspection)
- numpy, matplotlib, seaborn (for analysis and visualization)
- tqdm (for progress tracking)

See `environment.yaml` for the complete list of dependencies.

## Contributing

This repository contains research code. For questions or issues, please open a GitHub issue or refer to the paper for technical details.