# Implementation Notes and Reference Differences

## Overview
- Two-stage workflow:
  - Data generation (Experiment 1): create (prompt, CoT, answer) tuples from MATH.
  - Thought-anchor annotation (Experiment 2): identify the most causally important CoT sentence via counterfactual removal.

References compared against:
- docs/refs/thought-anchors (TA)
- docs/refs/steering-thinking-llms (STL)

## Recent Updates (August 2025)
- Resampling parity with base generation:
  - After open resampling, run a forced-answer fallback if no \boxed{…} appears, mirroring base generation.
  - Forced pass is seeded with the full open continuation (prefix + resampled continuation).
- Comparator parity with TA:
  - Counterfactual KL requires next-chunk rollouts (next_solutions at i+1). Comparator = next_solutions(i+1) + similar (from current chunk). If next_solutions is empty, KL = 0.0 (warn) — no similar-only fallback.
- Token budget alignment:
  - Open resampling budget: `max_new_tokens_open = max(1, 1000 − prefix_tokens)`; base generation uses `max_new_tokens=1000`.
- Counterfactual accuracy handling:
  - Removed fallback that copied baseline accuracy when partitions were empty; accuracy now always computed from the chunk’s resamples. Only KL can fall back to 0.0 (with a warning).
- Answer punctuation cleanup:
  - `cleanup_answer_punctuation(answer, gt_answer)`: if GT has no factorial, strip ‘!’; if GT has factorial, collapse to single ‘!’. Also strip trailing sentence punctuation.
  - Applied in base generation and `check_answer`.

## Experiment 1: Data Generation vs. STL
- Prompting:
  - Ours: Use the model’s chat template and expect a "<think>\n" prefix (e.g., DeepSeek‑R1‑Distill‑Qwen). Compute `prompt_len` by subtracting this prefix length.
  - STL: Often plain prompts without chat templates; focuses on steering rather than dataset construction.
- Outputs:
  - Ours: Persist `{prompt, raw_response, cot, answer}` (+ optional ground truth). No per‑token activations or steering artifacts here.
  - STL: Provides mean vectors, steering configs, and patching utilities; not centered on MATH data collection.
- Models and vectors:
  - Ours: `utils.load_model_and_vectors` loads an nnsight LanguageModel and, when present, mean vectors from `docs/refs/steering-thinking-llms` (warning emitted). No steering during data generation.
- Normalization and cleanup:
  - Ours: `extract_boxed_answers` with nested‑brace handling; `cleanup_answer_punctuation` prevents emphatic artifacts like `7!!!!!!!!`.
  - STL: Normalization/cleanup not central to their data path.

## Experiment 2: Annotation vs. TA
- Chunking:
  - Ours: `utils.split_solution_into_chunks` strips `<think>` and splits CoT (prompt excluded) into sentence/paragraph chunks, merging very small fragments.
  - TA: Similar TA‑style chunking with checks against their chunk artifacts.
- Baseline and pseudo‑GT:
  - Ours: If dataset GT is absent, derive a pseudo‑GT as the most common normalized answer from baseline forced answers on the full CoT.
  - TA: Uses rollout dataset and stored GT; may not rely on pseudo‑GT.
- Counterfactual removal protocol:
  - Ours: For each sentence i, remove it, sample multiple continuations from the prefix up to i, record one representative counterfactual answer, and compute per‑chunk metrics.
  - TA: Generates and persists detailed rollout artifacts per chunk (solutions.json).
- Similar vs. dissimilar split:
  - Ours: Cosine similarity between the removed sentence and the first resampled sentence using the reasoning model’s last‑layer mean embeddings; threshold 0.8.
  - TA: SentenceTransformer embeddings (e.g., all‑MiniLM‑L6‑v2); threshold ~0.8.
- Importance metric for anchor selection:
  - Ours: Counterfactual KL on correctness (P(true/false)) for dissimilar vs. comparator sets. Comparator = similar + next‑sentence (i+1); Laplace smoothing enabled.
  - TA: Often answer‑distribution KL, optionally includes/excludes similar; smoothing typically off.
  - Parity note: Our comparator matches TA’s structure and requirement for next_solutions; if missing, KL = 0.0 (warn).
- Other metrics (not used for selection):
  - Ours: `counterfactual_accuracies`, `different_trajectories_fraction`, `overdeterminedness`.
  - TA: Adds `forced_importance_*` and `resampling_importance_*` with richer cached artifacts.
- Token budgets and modes:
  - Ours: Open resampling capped (`1000 − prefix_tokens`); forced‑answer fallback when no box. Base generation uses `max_new_tokens=1000`.
  - TA: Much larger `max_tokens` (e.g., 16384); explicit “default” vs “forced_answer” modes.

## Experiment 2.5: Per‑Chunk Activations vs. STL
- Vector definition:
  - Ours: `scripts/find_chunk_activations.py` computes mean activation vectors at a chosen layer over each CoT chunk’s token span (teacher‑forced). L2‑normalized.
  - STL: Reference steering vectors keyed to capabilities (e.g., backtracking), typically from curated positive/negative sets.
- Counterfactual sampling:
  - Ours: No counterfactual sampling in this step; records per‑chunk means for downstream analysis/steering.
  - STL: Emphasizes applying precomputed directions.
- Practicalities:
  - Ours: Standard forward passes; nnsight to read hidden states.
  - STL: Focused on applying steering rather than deriving per‑example vectors.

## Design Trade‑offs
- Model last‑layer embeddings vs. sentence‑transformers: fewer deps, tighter coupling vs. cross‑model comparability.
- Correctness‑KL with smoothing vs. answer distribution: stability vs. granularity.
- Lightweight per‑chunk aggregates vs. heavier per‑chunk solution logs.

## Repro Tips
- Activate env: `conda activate anchorsteering`.
- For speed/variance, tune `--resamples`; optionally limit `--max_sentences`.
- For TA parity, consider SentenceTransformer embeddings and answer‑distribution KL (no smoothing).
- For STL comparisons, see `steering_config` in `utils.py`; note per‑example anchors vs. global capability vectors.
