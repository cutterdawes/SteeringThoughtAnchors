Implementation Notes and Reference Differences

Overview
- This repo implements a two‑stage workflow:
  1) Data generation (Experiment 1): create (prompt, CoT, answer) tuples from MATH.
  2) Thought‑anchor annotation (Experiment 2): identify the most causally important CoT sentence via counterfactual removal.

This document summarizes how our implementation differs from the reference projects:
- refs/thought-anchors (TA)
- refs/steering-thinking-llms (STL)


Experiment 1: Data Generation vs. STL
- Prompting:
  - Ours: Use the model’s chat template and expect a "<think>\n" prefix (e.g., DeepSeek‑R1‑Distill‑Qwen). We compute `prompt_len` by subtracting the length of this prefix so that downstream parsing can isolate the generated text.
  - STL: Often uses plain prompts without chat templates and focuses on steering rather than dataset construction.
- Outputs:
  - Ours: Persist `{prompt, raw_response, cot, answer}` (+ optional ground truth). No per‑token activations or steering artifacts are saved here.
  - STL: Provides mean vectors, steering configs, and patching utilities; not centered on MATH data collection.
- Models and vectors:
  - Ours: `utils.load_model_and_vectors` loads an nnsight LanguageModel and, when present, mean vectors from `refs/steering-thinking-llms` for convenience (warning emitted). We do not perform steering during data generation.


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


Design Trade‑offs
- Embedding choice (model last‑layer vs. sentence‑transformer): favors fewer dependencies and tighter coupling to the evaluated model at the cost of cross‑model comparability.
- KL over correctness vs. answer distribution: simplifies distributions and stabilizes with Laplace smoothing, but discards fine‑grained answer diversity.
- Artifact footprint: single JSON with per‑chunk aggregates (lightweight) vs. per‑chunk solution logs (heavyweight but richer diagnostics).


Repro Tips
- Always activate the conda env: `conda activate steering-thought-anchors`.
- For annotation speed/variance, tune `--resamples` and consider reducing `max_sentences` for quick sanity checks.
- When comparing to TA exactly, consider swapping in SentenceTransformer embeddings and computing answer‑distribution KL without smoothing.

