Implementation Notes and Reference Differences

Overview
- This repo implements a two‑stage workflow:
  1) Data generation (Experiment 1): create (prompt, CoT, answer) tuples from MATH.
  2) Thought‑anchor annotation (Experiment 2): identify the most causally important CoT sentence via counterfactual removal.

This document summarizes how our implementation differs from the reference projects:
- refs/thought-anchors (TA)
- refs/steering-thinking-llms (STL)


Recent Updates (August 2025)
- Resampling parity with base generation:
  - After open resampling, we now run a forced‑answer fallback if no \boxed{…} appears, mirroring base generation.
  - The forced pass is seeded with the full open continuation (prefix + whole resampled continuation), not just the first resampled sentence.
- Token budget alignment:
  - For open resampling, `max_new_tokens_open = max(1, 1000 − prefix_tokens)` so that existing CoT + new continuation ≤ 1000 tokens (matching base generation’s 1000 token cap).
- Counterfactual accuracy handling:
  - We removed the fallback that set per‑chunk counterfactual accuracy to the baseline accuracy when similarity partitions were empty. Accuracy is now always computed from that chunk’s resamples; only KL uses a 0.0 fallback (with a warning) when partitions are empty.
- Answer punctuation cleanup:
  - Added `cleanup_answer_punctuation(answer, gt_answer)`: if GT has no factorial, strip all ‘!’ from the candidate; if GT has factorial, collapse runs of ‘!’ to a single ‘!’. Also strip trailing sentence punctuation (., ?, …).
  - Applied in both base generation (saved answers) and `check_answer` (comparison path) for robustness.


Experiment 1: Data Generation vs. STL
- Prompting:
  - Ours: Use the model’s chat template and expect a "<think>\n" prefix (e.g., DeepSeek‑R1‑Distill‑Qwen). We compute `prompt_len` by subtracting the length of this prefix so that downstream parsing can isolate the generated text.
  - STL: Often uses plain prompts without chat templates and focuses on steering rather than dataset construction.
- Outputs:
  - Ours: Persist `{prompt, raw_response, cot, answer}` (+ optional ground truth). No per‑token activations or steering artifacts are saved here.
  - STL: Provides mean vectors, steering configs, and patching utilities; not centered on MATH data collection.
- Models and vectors:
  - Ours: `utils.load_model_and_vectors` loads an nnsight LanguageModel and, when present, mean vectors from `refs/steering-thinking-llms` for convenience (warning emitted). We do not perform steering during data generation.

- Normalization and cleanup:
  - Ours: Uses `extract_boxed_answers` with nested‑brace handling; answers are cleaned with `cleanup_answer_punctuation` before saving, preventing emphatic artifacts like `7!!!!!!!!`.
  - STL: Focuses on steering; normalization/cleanup is not central to their data path.


Experiment 2: Annotation vs. TA
- Chunking:
  - Ours: `utils.split_solution_into_chunks` strips `<think>` tags and splits CoT (prompt excluded) into sentence/paragraph chunks, merging very small fragments.
  - TA: Similar TA‑style chunking with checks against their chunk artifacts.
- Baseline and pseudo‑GT:
  - Ours: If dataset ground‑truth is absent, derive a pseudo‑GT as the most common normalized answer from baseline forced answers on the full CoT.
  - TA: Uses their rollout dataset and stored GT; may not rely on pseudo‑GT.
- Counterfactual removal protocol:
  - Ours: For each sentence i, remove it, sample multiple continuations from the prefix up to i, store one representative counterfactual answer, and compute per‑chunk metrics.
  - TA: Generates and persists detailed rollout artifacts per chunk (solutions.json), enabling broader analyses.
- Similar vs. dissimilar split:
  - Ours: Cosine similarity between the removed sentence and the first resampled sentence using the reasoning model’s last‑layer mean embeddings; threshold 0.8.
  - TA: Uses SentenceTransformer embeddings (e.g., all‑MiniLM‑L6‑v2) for similarity; threshold also ~0.8.
- Importance metric used for anchor selection:
  - Ours: Counterfactual KL on correctness (P(true/false)) for dissimilar vs. comparator sets. Comparator = similar + next‑sentence (i+1) resamples when available. Laplace smoothing enabled.
  - TA: Also constructs comparator sets and computes KL divergence; defaults to answer‑distribution KL and optionally includes/excludes similar. Smoothing typically off by default.
- Other metrics (not currently used for selection):
  - Ours: Store `counterfactual_accuracies` (aggregate accuracy at each i), `different_trajectories_fraction` (share of dissimilar resamples), and `overdeterminedness` (diversity proxy). We do not compute forced/resampling importance metrics.
  - TA: Additionally analyzes `forced_importance_*` and `resampling_importance_*`, and supports rich plotting/variance analyses using cached rollout artifacts.

- Token budgets and modes:
  - Ours: Open resampling budget is dynamically capped (`1000 − prefix_tokens`) and falls back to a forced‑answer pass when no box is produced. Base generation uses max_new_tokens=1000.
  - TA: Default `max_tokens` for generation is much larger (e.g., 16384 in `generate_rollouts.py`), and they expose separate “default” vs “forced_answer” modes; we now approximate their forced‑answer behavior for resampling while keeping a lighter token budget.


Design Trade‑offs
- Embedding choice (model last‑layer vs. sentence‑transformer): favors fewer dependencies and tighter coupling to the evaluated model at the cost of cross‑model comparability.
- KL over correctness vs. answer distribution: simplifies distributions and stabilizes with Laplace smoothing, but discards fine‑grained answer diversity.
- Artifact footprint: single JSON with per‑chunk aggregates (lightweight) vs. per‑chunk solution logs (heavyweight but richer diagnostics).


Experiment 2.5: Steering Vectors vs. STL
- Vector definition:
  - Ours: `experiments/find_steering_vectors.py` computes v = mean(layer activations on the anchor sentence) − mean(layer activations on sampled counterfactual sentences) at a chosen layer; vectors are L2‑normalized.
  - STL: Provides reference steering vectors and configs keyed to capabilities (e.g., backtracking). Their vectors are generally derived from curated positive/negative sets rather than per‑example anchors.
- Counterfactual sampling:
  - Ours: Samples counterfactual sentences from the prefix up to anchor, filters by dissimilarity to the anchor sentence using last‑layer embeddings, and averages accepted activations.
  - STL: Emphasizes applying precomputed steering directions rather than sampling‑based per‑example vectors.
- Practicalities:
  - Ours: Includes a `--dry_run` mode that produces deterministic pseudo‑vectors (testing only). Real runs rely on nnsight traces to read hidden states.
  - STL: Focused on running/applying steering rather than deriving vectors per example.


Repro Tips
- Always activate the conda env: `conda activate steering-thought-anchors`.
- For annotation speed/variance, tune `--resamples` and consider reducing `max_sentences` for quick sanity checks.
- When comparing to TA exactly, consider swapping in SentenceTransformer embeddings and computing answer‑distribution KL without smoothing.
- When comparing to STL steering behavior, use the provided `steering_config` in `utils.py` and note that our per‑example anchor vectors are conceptually different from their global capability vectors.
