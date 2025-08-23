# Repository Guidelines

## Project Structure & Module Organization
- `experiments/`: Task scripts for this research prototype.
  - `generate_data.py`: produce (prompt, CoT, answer, activations) tuples.
  - `annotate_data.py`: add placeholder thought‑anchor annotations.
  - `find_steering_vectors.py`: compute steering vectors (prototype logic).
- `generated_data/`: Saved JSON datasets produced by experiments.
- `utils.py`: Shared helpers for model loading, prompting, and activation processing.
- `refs/`: External references/research repos used for context.
- `environment.yml`: Conda environment with runtime and testing deps.

## Build, Test, and Development Commands
- Create env: `conda env create -f environment.yml && conda activate steering-thought-anchors`.
- Run data gen: `python experiments/generate_data.py` (writes to `generated_data/`).
- Annotate data: `python experiments/annotate_data.py` (writes annotated JSON).
- Find vectors: `python experiments/find_steering_vectors.py` (writes vector JSON).
- Tests (if added): `pytest -q`.
Notes:
- Scripts assume Python path includes repo root; they already append it. Avoid hard‑coded absolute paths when adding new code.
- API keys can be provided via shell env or a `.env`; current code loads env vars if present.

## Coding Style & Naming Conventions
- Python 3.10, PEP8, 4‑space indentation, type hints where helpful.
- Functions/variables: `snake_case`; modules: `lower_snake_case`.
- Generated files: `generated_data_{model}.json`, `steering_vectors_{model}.json`.
- Keep scripts idempotent and avoid side effects outside `generated_data/`.

## Testing Guidelines
- Framework: `pytest` is included; add tests under `tests/` mirroring module names (e.g., `tests/test_utils.py`).
- Name tests `test_*.py`; prefer small, fast unit tests over end‑to‑end runs.
- Run locally with `pytest -q`; add example fixtures for tiny synthetic prompts.

## Commit & Pull Request Guidelines
- Commits: imperative mood and concise (e.g., "Add steering vector export").
- PRs must include:
  - What/why summary and scope of changes.
  - Repro steps/commands (e.g., exact script invocations) and expected outputs.
  - Sample artifacts (paths under `generated_data/`) or logs.
  - Environment details (model name, device: `mps`/`cpu`, relevant env vars).
- Do not commit secrets or large generated artifacts; prefer `.gitignore` entries for datasets if they grow.

## Security & Configuration Tips
- Set provider keys via env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`.
- Keep `.env` files out of version control; export in shell during development.
- When adding models, use explicit names (e.g., `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) and avoid silently changing defaults.

