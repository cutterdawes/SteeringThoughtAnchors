# Steering Thought Anchors

Brief project description and quickstart for the “thought anchors” + activation steering prototype. For details and results, see docs/Bridging Features and Tokens (MATS).pdf and docs/IMPLEMENTATION_NOTES.md.

## Repo Structure
- `scripts/`: data and analysis scripts
  - `generate_data.py`: create (prompt, CoT, answer) tuples from MATH
  - `annotate_data.py`: select thought‑anchor sentences via counterfactuals
  - `find_chunk_activations.py`: per‑chunk mean activation vectors (teacher‑forced)
  - `categorize_chunks.py`: TA‑style function tags for CoT chunks
- `data/`: generated artifacts and figures (JSON, PNG)
- `notebooks/`: exploration and plots for anchors and activations
- `docs/refs/`: external references used for context
- `utils.py`: shared model/prompt/activation utilities

## Quickstart
- Create env: `conda env create -f environment.yml && conda activate steering-thought-anchors`
- Generate: `python scripts/generate_data.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --count 1 --output data`
- Annotate: `python scripts/annotate_data.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max_examples 1`
- Chunk activations: `python scripts/find_chunk_activations.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max_examples 1`
- Categorize chunks: `python scripts/categorize_chunks.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max_examples 1`

Artifacts are written under `data/` with names like:
- `generated_data_{model}.json`, `annotated_data_{model}.json`
- `chunk_activations_{model}.json`, `chunk_categories_{model}.json`
- figures in `data/figures/`

## Results (Summary)
Highlights from the write‑up (see the PDF for figures and details):
- Thought anchors: selected via counterfactual removal and correctness‑KL; anchors are causally salient under our metrics.
- Chunk activations: mean activation vectors over CoT chunks support clustering/visualization and downstream steering experiments.
- Steering behavior: per‑chunk/anchor directions modulate model behavior in controlled ways; KL curves provide interpretable dose–response.

## References & Notes
- Implementation differences vs. refs: `docs/IMPLEMENTATION_NOTES.md`
- External repos mirrored under `docs/refs/`
- API keys via env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`
