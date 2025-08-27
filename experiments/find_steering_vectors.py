import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Try importing torch; allow dry-run without it
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Ensure repo root on path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _device() -> str:
    """Prefer CUDA, then MPS, then CPU to match other experiments."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pool_to_hidden(x: "torch.Tensor", hidden_size: int) -> "torch.Tensor":
    """Robustly pool captured activations to a 1D hidden vector.
    Handles shapes [B,S,H], [S,H], [B,H], [H], and fallbacks.
    """
    try:
        if x is None or (hasattr(x, "numel") and x.numel() == 0):
            return torch.zeros(hidden_size, dtype=torch.float32)
        if x.dim() == 3:  # [B,S,H]
            return x[0].to(torch.float32).mean(dim=0)
        if x.dim() == 2:  # [S,H] or [B,H]
            a, b = x.shape
            if b == hidden_size:
                return x.to(torch.float32).mean(dim=0)
            if a == hidden_size:
                return x.to(torch.float32).mean(dim=1)
            # Fallback: flatten then trim/pad
            xf = x.flatten().to(torch.float32)
            out = torch.zeros(hidden_size, dtype=torch.float32)
            n = min(hidden_size, xf.shape[0])
            out[:n] = xf[:n]
            return out
        if x.dim() == 1:  # [H] or other
            if x.shape[0] == hidden_size:
                return x.to(torch.float32)
            out = torch.zeros(hidden_size, dtype=torch.float32)
            n = min(hidden_size, x.shape[0])
            out[:n] = x.to(torch.float32)[:n]
            return out
        # Scalar or unknown shape
        return torch.zeros(hidden_size, dtype=torch.float32)
    except Exception:
        return torch.zeros(hidden_size, dtype=torch.float32)


def _encode(tokenizer, text: str, device: str):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = ids["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    attention_mask = (input_ids != pad_id).long().to(device)
    return input_ids, attention_mask


def mean_layer_activation(model, tokenizer, text: str, layer_idx: int, device: str):
    if torch is None:
        raise RuntimeError("Torch is required.")
    if not text:
        return torch.zeros(model.config.hidden_size, dtype=torch.float32)
    input_ids, attention_mask = _encode(tokenizer, text, device)
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple length: num_layers+1 (embeddings + layers)
    idx = layer_idx + 1 if layer_idx >= 0 else -1
    t = hidden_states[idx]
    return _pool_to_hidden(t, model.config.hidden_size)


def mean_last_layer_embedding(model, tokenizer, text: str, device: str):
    if torch is None:
        raise RuntimeError("Torch is required.")
    if not text:
        return torch.zeros(model.config.hidden_size, dtype=torch.float32)
    input_ids, attention_mask = _encode(tokenizer, text, device)
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    t = outputs.hidden_states[-1]
    return _pool_to_hidden(t, model.config.hidden_size)


def cosine_similarity(a, b) -> float:
    # Support torch tensors or Python lists
    try:
        import math
        if torch is not None and hasattr(torch, 'Tensor') and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            a = a.flatten()
            b = b.flatten()
            if a.numel() == 0 or b.numel() == 0:
                return 0.0
            return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
        # fallback: lists
        ax = list(a)
        bx = list(b)
        if not ax or not bx:
            return 0.0
        n = min(len(ax), len(bx))
        ax = ax[:n]
        bx = bx[:n]
        dot = sum(x*y for x,y in zip(ax,bx))
        na = math.sqrt(sum(x*x for x in ax))
        nb = math.sqrt(sum(y*y for y in bx))
        if na == 0 or nb == 0:
            return 0.0
        return dot/(na*nb)
    except Exception:
        return 0.0


def extract_first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    import re
    parts = re.split(r"(?<=[\.!\?])\s+|\n+", text)
    return parts[0].strip()


def sample_counterfactual_sentence(
    model, tokenizer, question: str, prefix_text: str, device: str, do_sample: bool = True,
    temperature: float = 0.7, top_p: float = 0.9
) -> str:
    prompt = (
        "Solve the following problem step by step. You MUST put your final answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:\n<think>\n{prefix_text}"
    )
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=ids["input_ids"],
        attention_mask=(ids["input_ids"] != tokenizer.pad_token_id).long(),
        max_new_tokens=256,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    # Import robust generators lazily to avoid top-level torch requirements
    from utils import generate_with_model, decode_generate_outputs  # type: ignore
    try:
        out = generate_with_model(model, tokenizer, **gen_kwargs)
    except Exception:
        gen_kwargs.update(dict(do_sample=False))
        out = generate_with_model(model, tokenizer, **gen_kwargs)
    cont_text = decode_generate_outputs(tokenizer, out, skip_special_tokens=True)
    continuation_only = cont_text[len(prompt):]
    return extract_first_sentence(continuation_only)


def compute_anchor_vector_for_example(
    model,
    tokenizer,
    example: Dict,
    layer_idx: int,
    resamples: int,
    min_counterfactuals: int,
    device: str,
) -> Optional[Dict]:
    cot = example.get("cot", "")
    question = example.get("prompt", "")
    anchor_sentence = example.get("thought_anchor_sentence", "")
    anchor_idx = example.get("thought_anchor_idx", None)
    if not anchor_sentence:
        return None

    # Build prefix up to the anchor index if available
    # Try to use project chunker; fallback to simple splitter if unavailable without torch
    try:
        from utils import split_solution_into_chunks  # type: ignore
        chunks = split_solution_into_chunks(cot)
    except Exception:
        import re
        t = cot or ""
        if "<think>" in t:
            t = t.split("<think>",1)[1]
        if "</think>" in t:
            t = t.split("</think>",1)[0]
        chunks = [p.strip() for p in re.split(r"(?<=[\.!\?])\s+|\n\n+", t.strip()) if p.strip()]
    if anchor_idx is None or not (0 <= anchor_idx < len(chunks)):
        # fallback: find first matching chunk
        try:
            anchor_idx = next((i for i, s in enumerate(chunks) if s.strip() == anchor_sentence.strip()), 0)
        except Exception:
            anchor_idx = 0
    prefix_text = "\n".join(chunks[:anchor_idx])

    # Anchor activation at specified layer
    a_anchor = mean_layer_activation(model, tokenizer, anchor_sentence, layer_idx, device)

    # Prepare counterfactual sampling
    anchor_embed = mean_last_layer_embedding(model, tokenizer, anchor_sentence, device)
    cf_vectors: List[torch.Tensor] = []
    attempts = 0
    max_attempts = max(resamples * 2, min_counterfactuals)
    pbar = tqdm(total=max_attempts, desc=f"CF sampling@layer{layer_idx}", unit="try", leave=False)
    while len(cf_vectors) < min_counterfactuals and attempts < max_attempts:
        attempts += 1
        cf_sentence = sample_counterfactual_sentence(model, tokenizer, question, prefix_text, device)
        if not cf_sentence:
            pbar.update(1)
            continue
        cf_embed = mean_last_layer_embedding(model, tokenizer, cf_sentence, device)
        sim = cosine_similarity(anchor_embed, cf_embed)
        # Parity with annotation: treat as dissimilar if similarity < 0.8
        if sim < 0.8:
            # dissimilar enough
            cf_vec = mean_layer_activation(model, tokenizer, cf_sentence, layer_idx, device)
            cf_vectors.append(cf_vec)
        pbar.update(1)
        pbar.set_postfix(accepted=len(cf_vectors))
    pbar.close()

    # If none passed threshold, relax by taking best we have up to resamples
    if not cf_vectors:
        print("[info] No CF passed similarity < 0.8; relaxing filter (take first valid sentences)")
        for _ in tqdm(range(resamples), desc="CF relax", unit="try", leave=False):
            cf_sentence = sample_counterfactual_sentence(model, tokenizer, question, prefix_text, device)
            if cf_sentence:
                cf_vectors.append(mean_layer_activation(model, tokenizer, cf_sentence, layer_idx, device))
    if not cf_vectors:
        return None

    a_counter = torch.stack(cf_vectors, dim=0).mean(dim=0)
    v = a_anchor - a_counter
    v_norm = torch.nn.functional.normalize(v, dim=0)

    hidden_size = int(v.flatten().shape[0]) if hasattr(v, 'shape') else 0
    return {
        "anchor_idx": int(anchor_idx),
        "anchor_sentence": anchor_sentence,
        "layer": int(layer_idx),
        "hidden_size": hidden_size,
        "vector": v_norm.tolist(),
        "counterfactual_count": int(len(cf_vectors)),
    }


def find_thought_anchor_steering_vectors(
    model_name: str,
    annotated_path: str,
    output_dir: str,
    layer_idx: Optional[int] = None,
    resamples: int = 10,
    min_counterfactuals: int = 5,
    max_examples: Optional[int] = None,
):
    """
    Compute steering vectors v = mean(a_anchor) - mean(a_counter) per annotated example,
    using layer activations averaged across tokens in the anchor vs. sampled counterfactual sentences.
    Saves both per-example vectors and an overall average vector.
    """
    device = _device()
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(annotated_path):
        raise FileNotFoundError(f"Annotated data not found: {annotated_path}")

    with open(annotated_path, "r") as f:
        dataset = json.load(f)
    if max_examples is not None:
        dataset = dataset[:max_examples]

    if torch is None:
        raise RuntimeError("Torch not available; install env or use --dry_run.")

    # Lazy import to avoid importing utils when torch is missing
    from utils import load_model_and_vectors  # type: ignore
    print(f"Loading model {model_name}...")
    model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)

    if layer_idx is None:
        layer_idx = model.config.num_hidden_layers - 1
    layer_idx = int(layer_idx)

    per_example: List[Dict] = []
    collected_vectors: List[torch.Tensor] = []

    for i, ex in enumerate(dataset):
        if not ex.get("thought_anchor_sentence"):
            continue
        # Per-example header
        header = f"Example {i+1}/{len(dataset)}"
        preview = (ex.get("prompt","") or "")[:80].replace("\n"," ")
        print(f"\n=== {header} ===\nPrompt: {preview}{'...' if len(ex.get('prompt',''))>80 else ''}")
        anc_full = (ex.get('thought_anchor_sentence','') or '').replace('\n',' ')
        anc_preview = anc_full[:120] + ('...' if len(anc_full) > 120 else '')
        print(f"Anchor idx: {ex.get('thought_anchor_idx', None)} | sentence: {anc_preview}")
        res = compute_anchor_vector_for_example(
            model,
            tokenizer,
            ex,
            layer_idx=layer_idx,
            resamples=resamples,
            min_counterfactuals=min_counterfactuals,
            device=device,
        )
        if res is not None:
            per_example.append({
                "prompt": ex.get("prompt", "")[:2000],
                **res,
            })
            collected_vectors.append(torch.tensor(res["vector"], dtype=torch.float32))
            print(f"Saved vector: layer={res['layer']}, dim={res['hidden_size']}, cf_count={res['counterfactual_count']}")
        else:
            print("[warn] Skipped example: insufficient counterfactuals or missing anchor")

    mean_vector: Optional[List[float]] = None
    if collected_vectors:
        mv = torch.stack(collected_vectors, dim=0).mean(dim=0)
        mv = torch.nn.functional.normalize(mv, dim=0)
        mean_vector = mv.tolist()

    model_tag = model_name.replace('/', '-')
    out_path = os.path.join(output_dir, f"steering_vectors_{model_tag}.json")
    payload = {
        "model": model_name,
        "layer": layer_idx,
        "vectors": per_example,
        "mean_vector": mean_vector,
        "config": {
            "resamples": resamples,
            "min_counterfactuals": min_counterfactuals,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved steering vectors to {out_path}")


def _build_paths(model_name: str) -> Tuple[str, str]:
    model_tag = model_name.replace('/', '-')
    annotated_path = os.path.join("generated_data", f"generated_data_annotated_{model_tag}.json")
    output_dir = os.path.join("generated_data")
    return annotated_path, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute thought-anchor steering vectors (Experiment 2)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--layer", type=int, default=None, help="Layer index for activations (default: last layer)")
    parser.add_argument("--resamples", type=int, default=10, help="Counterfactual resamples per example")
    parser.add_argument("--min_counterfactuals", type=int, default=5, help="Minimum accepted counterfactual sentences")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--annotated_path", type=str, default=None, help="Optional explicit path to annotated data JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory (default: generated_data)")
    args = parser.parse_args()

    annotated_path, default_out = _build_paths(args.model)
    if args.annotated_path:
        annotated_path = args.annotated_path
    out_dir = args.output_dir or default_out

    find_thought_anchor_steering_vectors(
        model_name=args.model,
        annotated_path=annotated_path,
        output_dir=out_dir,
        layer_idx=args.layer,
        resamples=args.resamples,
        min_counterfactuals=args.min_counterfactuals,
        max_examples=args.max_examples,
    )
