"""
Experiment 2: Annotation of Thought Anchors

This script annotates each (prompt, CoT, answer) example with a single
"thought anchor" sentence — the sentence whose removal most disrupts the
final answer behavior.

Method summary (our implementation vs. references):
- We split the CoT into sentence/paragraph chunks (prompt excluded) using
  TA-style chunking from utils.split_solution_into_chunks.
- For each sentence i, we remove it and generate multiple stochastic
  continuations from the prefix up to i. We mark each rollout as correct or
  incorrect relative to a ground truth (dataset GT if available; otherwise
  a pseudo-GT derived from baseline forced answers on the original CoT).
- We embed the removed sentence and the first resampled sentence using the
  model's last-layer mean hidden states (via nnsight). We label a resample as
  "dissimilar" if cosine < 0.9; otherwise it is considered "similar".
- Importance score (selection): KL divergence between the correctness
  distribution of dissimilar resamples and a comparator built from the set of
  similar resamples plus the next-sentence (i+1) rollouts when available.
  The sentence with maximum KL is chosen as the thought anchor.

Key differences from refs/thought-anchors:
- Similarity backend: The paper/code uses SentenceTransformer embeddings
  (e.g., all-MiniLM); we use the reasoning model's last-layer embeddings.
- KL over correctness: We compute KL over {true,false} distributions by
  default (use_prob_true=True) with Laplace smoothing. The reference code
  frequently uses answer-distribution KL and may omit smoothing by default.
- Metrics scope: We only compute the counterfactual (removal) metrics used
  for anchor selection. We do not compute or persist the additional
  resampling/forced importance metrics or per-chunk rollout artifacts.

For a complete comparison, see docs/IMPLEMENTATION_NOTES.md.
"""

import json
import os
import re
import argparse
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm, trange

import torch

# Ensure repo root on path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_model_and_vectors,
    extract_thinking_process_and_answer,
    check_answer,
    normalize_answer as utils_normalize_answer,
    split_solution_into_chunks,
    generate_with_model,
    decode_generate_outputs,
)


def generate_forced_answer(
    model,
    tokenizer,
    question: str,
    cot_text: str,
    device: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    variation_seed: Optional[int] = None,
) -> Tuple[str, str]:
    # Build a forced-answer prompt that seeds the CoT and asks for boxed answer
    forced_prefix = (
        "Solve the following problem step by step. You MUST put your final answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:\n<think>\n{cot_text}\n</think>\n\nTherefore, the final answer is \\boxed{{"
    )
    inputs = tokenizer(forced_prefix, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=(inputs["input_ids"] != tokenizer.pad_token_id).long(),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    try:
        outputs = generate_with_model(model, tokenizer, **gen_kwargs)
    except Exception:
        gen_kwargs.update(dict(do_sample=False))
        outputs = generate_with_model(model, tokenizer, **gen_kwargs)
    text = decode_generate_outputs(tokenizer, outputs, skip_special_tokens=True)
    cot, ans = extract_thinking_process_and_answer(text, prompt_len=0)
    return cot, ans

def embed_text_with_model(model, tokenizer, text: str, device: str) -> torch.Tensor:
    if not text:
        return torch.zeros(model.config.hidden_size, dtype=torch.float32)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with model.trace({
        "input_ids": ids["input_ids"],
        "attention_mask": (ids["input_ids"] != tokenizer.pad_token_id).long(),
    }):
        last = model.model.layers[model.config.num_hidden_layers - 1].output[0].save()
    h = last.cpu().detach().to(torch.float32)[0]
    return h.mean(dim=0) if h.numel() else torch.zeros(model.config.hidden_size, dtype=torch.float32)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        a = a.flatten()
        b = b.flatten()
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    except Exception:
        return 0.0

def extract_first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[\.!\?])\s+|\n+", text)
    return parts[0].strip()

def generate_open_rollout_and_answer(
    model,
    tokenizer,
    question: str,
    prefix_text: str,
    device: str,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    variation_seed: Optional[int] = None,  # reserved; sampler randomness is controlled via HF sampling params
    max_new_tokens_open: int = 256,
    max_new_tokens_forced: int = 128,
) -> Tuple[str, str]:
    prompt = (
        "Solve the following problem step by step. You MUST put your final answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:\n<think>\n{prefix_text}"
    )
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=ids["input_ids"],
        attention_mask=(ids["input_ids"] != tokenizer.pad_token_id).long(),
        max_new_tokens=max(1, int(max_new_tokens_open)),
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    try:
        out_ids = generate_with_model(model, tokenizer, **gen_kwargs)
    except Exception:
        gen_kwargs.update(dict(do_sample=False))
        out_ids = generate_with_model(model, tokenizer, **gen_kwargs)
    cont_text = decode_generate_outputs(tokenizer, out_ids, skip_special_tokens=True)
    rollout_text = cont_text[len(prompt):]
    resampled_chunk = extract_first_sentence(rollout_text)
    # Extract boxed answer from open-ended rollout
    try:
        from utils import extract_boxed_answers as _extract
        boxes = _extract(rollout_text)
    except Exception:
        boxes = []
    ans = boxes[0] if boxes else ""
    # Forced-answer fallback to mirror base generation if no boxed answer was found
    if not ans:
        # Use the entire open continuation for the forced fallback to increase parity
        combined_cot = (prefix_text or "").rstrip() + ("\n" if prefix_text else "") + (rollout_text or "")
        _cot2, forced_ans = generate_forced_answer(
            model,
            tokenizer,
            question,
            combined_cot,
            device,
            max_new_tokens=max_new_tokens_forced,
            do_sample=False,
        )
        if forced_ans:
            ans = forced_ans
    return resampled_chunk, ans

def calculate_kl_divergence(
    sols_p: List[Dict], sols_q: List[Dict], laplace_smooth: bool = True, use_prob_true: bool = True
) -> float:
    from math import log
    def to_dist(sols: List[Dict]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        if use_prob_true:
            for s in sols:
                key = "true" if s.get("is_correct", False) else "false"
                counts[key] = counts.get(key, 0) + 1
        else:
            for s in sols:
                key = utils_normalize_answer(s.get("answer", ""))
                if key:
                    counts[key] = counts.get(key, 0) + 1
        if not counts:
            return {}
        if laplace_smooth:
            vocab = list(counts.keys())
            total = sum(counts.values()) + len(vocab)
            return {k: (counts[k] + 1) / total for k in vocab}
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
    P = to_dist(sols_p)
    Q = to_dist(sols_q)
    if not P or not Q:
        return 0.0
    keys = set(P) | set(Q)
    eps = 1e-8
    kl = 0.0
    for k in keys:
        p = P.get(k, eps)
        q = Q.get(k, eps)
        kl += p * (0.0 if p <= 0 else log(p / q))
    return float(kl)

def annotate_data_with_thought_anchors(
    input_data_path: str,
    output_data_path: str,
    model_name: str,
    device: str,
    max_examples: Optional[int] = None,
    max_sentences: Optional[int] = None,
    max_new_tokens_forced: int = 128,
    max_total_cot_tokens: int = 1000,
    resamples: int = 10,
    use_abs_importance: bool = True,
    ground_truth_map: Optional[dict] = None,
):
    """
    Annotate thought anchors via a counterfactual removal test per sentence.
    For each CoT sentence, remove it, force a final answer, and mark if the answer changes.
    Anchor selection uses KL divergence on correctness distributions for
    dissimilar resamples vs. a comparator (similar + next-sentence). The
    sentence with maximum KL is selected as the thought anchor.
    """
    print(f"Loading data from {input_data_path} for annotation...")
    with open(input_data_path, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} data points.")

    print(f"Loading model {model_name} for counterfactual tests on device {device}...")
    model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)
    print("Model loaded.")

    annotated_dataset = []
    if max_examples is not None:
        dataset = dataset[:max_examples]

    total_examples = len(dataset)
    for i, data_point in enumerate(dataset, start=1):
        print(f"\n=== Example {i}/{total_examples} ===")
        question = data_point.get('prompt', '')
        original_cot = data_point.get('cot', '')
        original_answer = data_point.get('answer', '')
        q_preview = (question or "").replace("\n", " ")
        if len(q_preview) > 80:
            q_preview = q_preview[:80] + "..."
        print(f"Prompt: {q_preview}")

        # Ensure we have a baseline final answer; compute via forced pass if missing
        if not original_answer:
            _, original_answer = generate_forced_answer(
                model,
                tokenizer,
                question,
                original_cot,
                model.device,
                max_new_tokens=max_new_tokens_forced,
            )

        # Prepare ground truth if available; else fall back to baseline normalized answer
        gt_answer = None
        if ground_truth_map and question in ground_truth_map:
            gt_answer = ground_truth_map[question]

        # Baseline resamples and accuracy
        base_answers: List[str] = []
        print(f"Baseline forced resamples: {resamples}")
        for r in trange(resamples, desc="Baseline", leave=False):
            # Try sampling; fall back handled inside
            _, ans = generate_forced_answer(
                model,
                tokenizer,
                question,
                original_cot,
                model.device,
                max_new_tokens=max_new_tokens_forced,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                variation_seed=r,
            )
            base_answers.append(ans or "")

        if gt_answer is None:
            # Use the most common baseline normalized answer as pseudo-GT
            norm_counts = {}
            for a in base_answers:
                na = utils_normalize_answer(a)
                norm_counts[na] = norm_counts.get(na, 0) + 1
            pseudo_gt = max(norm_counts.items(), key=lambda x: x[1])[0] if norm_counts else utils_normalize_answer(original_answer)
            gt_answer = pseudo_gt

        base_correct = sum(1 for a in base_answers if check_answer(a, gt_answer))
        base_accuracy = base_correct / max(1, len(base_answers))

        # Split using TA-style chunking
        sentences = split_solution_into_chunks(original_cot)
        print(f"CoT chunks: {len(sentences)}")
        chunk_solutions: Dict[int, List[Dict]] = {i: [] for i in range(len(sentences))}
        cf_importance_kl: List[float] = []
        cf_accuracies: List[float] = []
        cf_answers: List[str] = []
        different_trajectories: List[float] = []
        overdeterminedness: List[float] = []

        if not sentences:
            data_point['thought_anchor_idx'] = None
            data_point['thought_anchor_sentence'] = ""
            annotated_dataset.append(data_point)
            continue

        sentence_range = range(len(sentences)) if max_sentences is None else range(min(len(sentences), max_sentences))
        # Precompute removed chunk embeddings
        removed_embeds = [embed_text_with_model(model, tokenizer, s, model.device) for s in sentences]
        for idx in tqdm(sentence_range, desc="Chunks", unit="chunk", leave=False):
            prefix_text = "\n".join(sentences[:idx])
            removed_text = sentences[idx]
            removed_embed = removed_embeds[idx]
            sols: List[Dict] = []
            # Compute token budget so that prefix CoT + new tokens <= max_total_cot_tokens
            try:
                prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                prefix_tokens = int(prefix_ids.shape[-1])
            except Exception:
                prefix_tokens = 0
            remaining_budget = max(1, int(max_total_cot_tokens) - prefix_tokens)
            for r in trange(resamples, desc=f"Resamples@{idx}", leave=False):
                resampled_chunk, a = generate_open_rollout_and_answer(
                    model, tokenizer, question, prefix_text, model.device,
                    do_sample=True, temperature=0.6, top_p=0.95, variation_seed=r,
                    max_new_tokens_open=remaining_budget,
                    max_new_tokens_forced=max_new_tokens_forced,
                )
                sim = cosine_similarity(removed_embed, embed_text_with_model(model, tokenizer, resampled_chunk, model.device))
                sols.append({
                    "chunk_removed": removed_text,
                    "chunk_resampled": resampled_chunk,
                    "similarity": sim,
                    "answer": a or "",
                    "is_correct": bool(check_answer(a or "", gt_answer)),
                })
            chunk_solutions[idx] = sols
        # Compute metrics per chunk
        for idx in sentence_range:
            current = chunk_solutions.get(idx, [])
            next_solutions = chunk_solutions.get(idx + 1, []) if (idx + 1) in chunk_solutions else []
            dissimilar = [s for s in current if s.get("similarity", 0.0) < 0.8]
            similar = [s for s in current if s.get("similarity", 0.0) >= 0.8]
            diff_frac = (len(dissimilar) / len(current)) if current else 0.0
            different_trajectories.append(diff_frac)
            texts = [s.get("chunk_resampled", "") for s in current]
            overdet = 1.0 - (len(set(texts)) / len(texts)) if texts else 0.0
            overdeterminedness.append(overdet)
            # TA parity: require next_solutions; include similar alongside next_solutions
            comparator = (next_solutions + similar) if next_solutions else []
            # Always compute per-chunk accuracy from current resamples
            acc = sum(1 for s in current if s.get("is_correct", False)) / max(1, len(current)) if current else 0.0
            cf_accuracies.append(acc)
            cf_answers.append(current[0].get("answer", "") if current else "")
            # KL requires non-empty partitions; warn only when comparator is empty (TA requires next_solutions)
            if not comparator or not dissimilar:
                if not comparator:
                    print(f"[warn] Empty comparator at chunk {idx}: next_solutions missing or empty; KL set to 0.0")
                cf_importance_kl.append(0.0)
            else:
                kl = calculate_kl_divergence(dissimilar, comparator, laplace_smooth=True, use_prob_true=True)
                cf_importance_kl.append(kl)

        # Select anchor as first max score
        max_imp = max(cf_importance_kl) if cf_importance_kl else 0.0
        anchor_idx = cf_importance_kl.index(max_imp) if cf_importance_kl else None
        anchor_sentence = sentences[anchor_idx] if anchor_idx is not None else ""
        print(f"Selected anchor idx: {anchor_idx} | KL max: {max_imp:.4f}")
        if anchor_sentence:
            anc_preview = (anchor_sentence or "").replace("\n", " ")
            if len(anc_preview) > 120:
                anc_preview = anc_preview[:120] + "..."
            print(f"Anchor sentence: {anc_preview}")

        # Drop any saved activations from upstream dataset; we don't persist activations here
        if 'activations' in data_point:
            try:
                del data_point['activations']
            except Exception:
                pass

        data_point['thought_anchor_idx'] = anchor_idx
        data_point['thought_anchor_sentence'] = anchor_sentence
        data_point['baseline_accuracy'] = base_accuracy
        data_point['counterfactual_accuracies'] = cf_accuracies
        data_point['counterfactual_importance_kl'] = cf_importance_kl
        data_point['different_trajectories_fraction'] = different_trajectories
        data_point['overdeterminedness'] = overdeterminedness
        data_point['counterfactual_answers'] = cf_answers

        annotated_dataset.append(data_point)

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    print(f"Saving annotated data to {output_data_path}...")
    with open(output_data_path, 'w') as f:
        json.dump(annotated_dataset, f, indent=4)
    print("Annotation complete.")


if __name__ == "__main__":
    # CLI for faster testing and control
    parser = argparse.ArgumentParser(
        description=(
            "Annotate thought-anchor sentences using counterfactual removal tests. "
            "For each CoT chunk the script removes that chunk, resamples continuations, "
            "and measures how often the final boxed answer changes — the sentence that most "
            "often changes the answer is recorded as the thought anchor."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help=(
            "Hugging Face model id used for the counterfactual tests. "
            "Must match the model used to generate the input file so the script can locate "
            "`generated_data/generated_data_{model.replace('/', '-')}.json`. "
            "Default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B."
        ),
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help=(
            "Limit processing to the first N examples in the input file (useful for quick tests). "
            "If omitted, the script processes all examples."
        ),
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help=(
            "Limit how many CoT sentences (chunks) to consider per example. "
            "If set, only the first N chunks are evaluated; otherwise all chunks are used."
        ),
    )
    parser.add_argument(
        "--max_new_tokens_forced",
        type=int,
        default=128,
        help=(
            "Maximum number of tokens to generate when performing forced-answer completions. "
            "Used for computing baseline and counterfactual boxed answers; larger values may capture "
            "longer boxed answers but cost more time. Default: 128."
        ),
    )
    parser.add_argument(
        "--max_total_cot_tokens",
        type=int,
        default=1000,
        help=(
            "Maximum total CoT tokens allowed after resampling (existing prefix + new continuation). "
            "Open resampling uses a dynamic budget of max_total_cot_tokens - prefix_tokens. Default: 1000."
        ),
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=10,
        help=(
            "Number of stochastic resamples to draw per removal condition (per chunk). "
            "Higher values produce more stable importance estimates but increase runtime linearly. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--no_abs_importance",
        action="store_true",
        help=(
            "Reserved flag (currently a no-op): in prior iterations this toggled the "
            "sign of accuracy-difference importance. We now select anchors solely via KL "
            "divergence on correctness distributions; this flag does not affect selection."
        ),
    )
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file mapping {prompt: ground-truth-answer}. "
            "If provided, correctness will be computed against this mapping; otherwise the script "
            "will attempt to auto-detect `generated_data/ground_truth_math.json`."
        ),
    )
    args = parser.parse_args()

    model_name = args.model
    model_tag = model_name.replace('/', '-')
    # Prefer CUDA if available, then MPS, otherwise CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    input_path = os.path.join("generated_data", f"generated_data_{model_tag}.json")
    output_path = os.path.join("generated_data", f"generated_data_annotated_{model_tag}.json")

    gt_map = None
    if args.ground_truth_json and os.path.exists(args.ground_truth_json):
        with open(args.ground_truth_json, 'r') as f:
            gt_map = json.load(f)
    else:
        # Auto-detect ground truth saved by generate_data when using --use_math
        auto_gt = os.path.join("generated_data", "ground_truth_math.json")
        if os.path.exists(auto_gt):
            with open(auto_gt, 'r') as f:
                gt_map = json.load(f)

    annotate_data_with_thought_anchors(
        input_path,
        output_path,
        model_name,
        device,
        max_examples=args.max_examples,
        max_sentences=args.max_sentences,
        max_new_tokens_forced=args.max_new_tokens_forced,
        max_total_cot_tokens=args.max_total_cot_tokens,
        resamples=args.resamples,
        use_abs_importance=(not args.no_abs_importance),
        ground_truth_map=gt_map,
    )
