import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

# Try importing torch; allow dry-run without it
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from tqdm import tqdm

# Ensure repo root on path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_full(tokenizer, text: str, device: str):
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = ids["input_ids"].to(device)
    offsets = ids["offset_mapping"][0].tolist()  # List[Tuple[int,int]]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    attention_mask = (input_ids != pad_id).long().to(device)
    return input_ids, attention_mask, offsets


def _char_span_to_token_span(char_start: int, char_end_exclusive: int, offsets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Convert character span [char_start, char_end_exclusive) to token span [t0, t1).
    Uses first token whose end > start for t0, and first token whose start >= end for t1.
    Falls back to scanning inside span for robustness.
    """
    n = len(offsets)
    if n == 0:
        return None

    # Find start token: first token with end > char_start
    t0 = None
    for i, (s, e) in enumerate(offsets):
        if e > char_start:
            t0 = i
            break

    if t0 is None:
        return None

    # Find end token (exclusive): first token with start >= char_end_exclusive
    t1 = None
    for i, (s, e) in enumerate(offsets):
        if s >= char_end_exclusive:
            t1 = i
            break
    if t1 is None:
        t1 = n

    if t0 >= t1:
        return None
    return t0, t1


def _pool_hidden(x: "torch.Tensor") -> "torch.Tensor":
    """Mean-pool sequence hidden states to a 1D vector.
    Accepts shapes [S,H] or [1,S,H]. Returns [H] (float32).
    """
    if torch is None:
        raise RuntimeError("Torch is required.")
    if x is None:
        return torch.zeros(0)
    if x.dim() == 3:
        x = x[0]
    if x.dim() != 2:
        return torch.zeros(0)
    return x.to(torch.float32).mean(dim=0)


def _normalize(v: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch is required.")
    if v.numel() == 0:
        return v
    return torch.nn.functional.normalize(v, dim=0)


def compute_chunk_vectors_for_example(
    model,
    tokenizer,
    example: Dict,
    layer_idx: int,
    device: str,
) -> Optional[Dict]:
    """Compute mean activation vectors for all CoT chunks via teacher-forced forward pass.

    For each chunk, we feed the full input text (model_input_prompt + full CoT),
    extract the token span corresponding to the chunk, and mean-pool the selected
    layer's hidden states over that span.
    """
    from utils import split_solution_into_chunks, get_chunk_ranges  # lazy import

    cot = example.get("cot", "") or ""
    model_input_prompt = example.get("model_input_prompt", "") or ""
    if not cot or not model_input_prompt:
        return None

    # Split CoT into chunks and find char spans within CoT text
    chunks = split_solution_into_chunks(cot)
    if not chunks:
        return None
    chunk_ranges_cot = get_chunk_ranges(cot, chunks)
    if not chunk_ranges_cot or len(chunk_ranges_cot) != len(chunks):
        # Fallback: naive splitting by sentences
        chunk_ranges_cot = []
        cursor = 0
        for ch in chunks:
            pos = cot.find(ch, cursor)
            if pos < 0:
                return None
            chunk_ranges_cot.append((pos, pos + len(ch)))
            cursor = pos + len(ch)

    full_text = model_input_prompt + cot
    # Encode with offsets mapping so we can align char spans to token spans
    input_ids, attention_mask, offsets = _encode_full(tokenizer, full_text, device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # hidden_states: tuple (embeddings + layers). Our layer is index+1
    hs = outputs.hidden_states
    idx = layer_idx + 1 if layer_idx >= 0 else -1
    layer_h = hs[idx][0]  # [S,H]
    hidden_size = int(layer_h.shape[-1])

    prefix_len_chars = len(model_input_prompt)
    chunk_payloads: List[Dict] = []

    for ci, (ch, (c0, c1)) in enumerate(zip(chunks, chunk_ranges_cot)):
        # Map CoT char span to full_text char span
        span_start = prefix_len_chars + int(c0)
        span_end = prefix_len_chars + int(c1)
        token_span = _char_span_to_token_span(span_start, span_end, offsets)
        if token_span is None:
            # Skip if we can't align reliably
            continue
        t0, t1 = token_span
        if t0 < 0 or t1 <= t0 or t1 > layer_h.shape[0]:
            continue
        v_mean = _pool_hidden(layer_h[t0:t1, :])
        v_norm = _normalize(v_mean)

        chunk_payloads.append({
            "chunk_index": int(ci),
            "text": ch,
            "token_span": [int(t0), int(t1)],
            "hidden_size": hidden_size,
            "vector": v_norm.cpu().tolist(),
        })

    if not chunk_payloads:
        return None

    return {
        "prompt": example.get("prompt", "")[:2000],
        "thought_anchor_idx": example.get("thought_anchor_idx", None),
        "num_chunks": len(chunks),
        "layer": int(layer_idx),
        "chunks": chunk_payloads,
    }


def find_chunk_steering_vectors(
    model_name: str,
    annotated_path: str,
    output_dir: str,
    layer_idx: Optional[int] = None,
    max_examples: Optional[int] = None,
):
    """
    Compute per-chunk steering vectors as the mean activations at a given layer
    over the token span of each CoT chunk (teacher-forced in-context forward pass).

    Differs from experiments/find_steering_vectors.py by not using counterfactuals
    and returning vectors for all chunks, not just the annotated thought anchor.
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
        raise RuntimeError("Torch not available; install env or activate the provided environment.")

    # Lazy import to avoid top-level torch deps in utils
    from utils import load_model_and_vectors  # type: ignore
    print(f"Loading model {model_name}...")
    model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device)

    if layer_idx is None:
        layer_idx = model.config.num_hidden_layers - 1
    layer_idx = int(layer_idx)

    results: List[Dict] = []

    for i, ex in enumerate(dataset):
        header = f"Example {i+1}/{len(dataset)}"
        preview = (ex.get("prompt", "") or "")[:80].replace("\n", " ")
        print(f"\n=== {header} ===\nPrompt: {preview}{'...' if len(ex.get('prompt',''))>80 else ''}")
        res = compute_chunk_vectors_for_example(
            model,
            tokenizer,
            ex,
            layer_idx=layer_idx,
            device=device,
        )
        if res is not None:
            results.append(res)
            print(f"Saved {len(res['chunks'])} chunk vectors at layer={layer_idx}")
        else:
            print("[warn] Skipped example: could not align chunks or missing fields")

    model_tag = model_name.replace('/', '-')
    out_path = os.path.join(output_dir, f"steering_anchors_{model_tag}.json")
    payload = {
        "model": model_name,
        "layer": layer_idx,
        "examples": results,
        "config": {},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved per-chunk steering vectors to {out_path}")


def _build_paths(model_name: str) -> Tuple[str, str]:
    model_tag = model_name.replace('/', '-')
    annotated_path = os.path.join("generated_data", f"generated_data_annotated_{model_tag}.json")
    output_dir = os.path.join("generated_data")
    return annotated_path, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-chunk steering vectors via mean activations (no CF)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--layer", type=int, default=None, help="Layer index for activations (default: last layer)")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--annotated_path", type=str, default=None, help="Optional explicit path to annotated data JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory (default: generated_data)")
    args = parser.parse_args()

    annotated_path, default_out = _build_paths(args.model)
    if args.annotated_path:
        annotated_path = args.annotated_path
    out_dir = args.output_dir or default_out

    find_chunk_steering_vectors(
        model_name=args.model,
        annotated_path=annotated_path,
        output_dir=out_dir,
        layer_idx=args.layer,
        max_examples=args.max_examples,
    )

