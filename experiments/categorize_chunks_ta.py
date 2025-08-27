import argparse
import json
import os
import re
from typing import Dict, List, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import split_solution_into_chunks, chat  # type: ignore


TA_FUNCTION_TAGS = [
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",
    "final_answer_emission",
    "self_checking",
    "unknown",
]

TA_FUNCTION_TAGS_DOC = """
Function Tags (assign one or more per chunk; use only from this list):
- problem_setup: Parsing or rephrasing the problem (initial reading/comprehension).
- plan_generation: Stating or deciding on a plan of action (meta-reasoning about what to do next).
- fact_retrieval: Recalling facts, formulas, problem details (without immediate computation).
- active_computation: Performing algebra, calculations, manipulations toward the answer.
- result_consolidation: Aggregating intermediate results, summarizing, or preparing final answer.
- uncertainty_management: Expressing confusion, re-evaluating, or proposing alternative strategies (includes backtracking).
- final_answer_emission: Explicit statement of the final boxed answer or a chunk that emits the final answer.
- self_checking: Verifying previous steps, Pythagorean checking, re-confirmations.
- unknown: Use only if the chunk does not fit any of the above tags or is purely stylistic.
"""


def build_annotation_prompt(chunks: List[str]) -> str:
    header = (
        "You are an expert annotator. Given a Chain-of-Thought split into discrete chunks, "
        "label each chunk with function_tags from the provided tag set.\n\n"
    )
    instructions = (
        TA_FUNCTION_TAGS_DOC
        + "\nOutput strictly as minified JSON (no prose), mapping chunk indices (as strings) to an object with a 'function_tags' list. Example: {\"0\": {\"function_tags\":[\"problem_setup\"]}, \"1\": {\"function_tags\":[\"plan_generation\",\"fact_retrieval\"]}}.\n"
        + "Only use the allowed tags; prefer 1-2 tags per chunk; never invent new tags.\n\n"
    )
    listing = [f"[{i}] {chunks[i]}" for i in range(len(chunks))]
    body = "Chunks:\n" + "\n".join(listing) + "\n\nReturn JSON now."
    return header + instructions + body


def _extract_json_blob(s: str) -> Optional[Dict]:
    # Find first JSON-like object in the response
    try:
        # Direct attempt
        return json.loads(s)
    except Exception:
        pass
    # Fallback: extract between first { and last }
    try:
        start = s.find('{')
        end = s.rfind('}')
        if start >= 0 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        return None
    return None


def _sanitize_tags(tags: List[str]) -> List[str]:
    out = []
    for t in tags or []:
        k = re.sub(r"[^a-z_]+", "", (t or "").strip().lower())
        if not k:
            continue
        # Map common aliases
        alias = {
            "final_answer": "final_answer_emission",
            "final_answer_emission": "final_answer_emission",
            "plan": "plan_generation",
            "planning": "plan_generation",
            "retrieve": "fact_retrieval",
            "retrieval": "fact_retrieval",
            "computation": "active_computation",
            "calc": "active_computation",
            "calculation": "active_computation",
            "result": "result_consolidation",
            "consolidation": "result_consolidation",
            "uncertainty": "uncertainty_management",
            "self_check": "self_checking",
        }
        if k in alias:
            k = alias[k]
        if k not in TA_FUNCTION_TAGS:
            # Try to match by prefix
            match = next((v for v in TA_FUNCTION_TAGS if v.startswith(k)), None)
            if match:
                k = match
        if k in TA_FUNCTION_TAGS and k not in out:
            out.append(k)
    if not out:
        out = ["unknown"]
    return out[:3]


def categorize_examples(
    annotated_path: str,
    output_dir: str,
    annotator_model: str = "qwen/qwen2.5-3b-instruct",
    max_examples: Optional[int] = None,
    anchors_path_for_check: Optional[str] = None,
) -> str:
    """Categorize chunks directly from annotated dataset (CoT), verifying chunking against anchors if available.

    - annotated_path: generated_data_annotated_{model}.json
    - anchors_path_for_check: optional path to steering_anchors_{model}.json; used only to sanity-check chunking parity.
    - annotator_model: default to Qwen2.5-3B-Instruct via OpenRouter id 'qwen/qwen2.5-3b-instruct'.
    """
    if not os.path.exists(annotated_path):
        raise FileNotFoundError(f"Missing annotated JSON: {annotated_path}")
    with open(annotated_path, 'r') as f:
        annotated = json.load(f)
    if max_examples is not None:
        annotated = annotated[:max_examples]

    # Optionally load anchors (for parity check only)
    anchors_examples: List[Dict] = []
    if anchors_path_for_check and os.path.exists(anchors_path_for_check):
        try:
            with open(anchors_path_for_check, 'r') as f:
                anc = json.load(f)
            anchors_examples = anc.get('examples', [])
        except Exception as e:
            print(f"[warn] Could not load anchors for parity check: {e}")

    # Attempt to infer model name for output naming
    model_name = ''
    try:
        model_name = (anc.get('model') if anchors_examples else '') or ''
    except Exception:
        model_name = ''

    os.makedirs(output_dir, exist_ok=True)

    results: List[Dict] = []
    for ex_idx, ex in enumerate(annotated):
        cot = ex.get('cot') or ''
        chunks = split_solution_into_chunks(cot)
        if not chunks:
            continue

        # Parity check vs anchors if present
        if anchors_examples and ex_idx < len(anchors_examples):
            a_chunks = anchors_examples[ex_idx].get('chunks', [])
            if len(a_chunks) != len(chunks):
                print(f"[warn] chunk count mismatch at example {ex_idx}: annotated={len(chunks)} vs anchors={len(a_chunks)}")
            else:
                # Spot-check a few positions for normalized equality
                import random as _r
                for j in _r.sample(range(len(chunks)), k=min(3, len(chunks))):
                    ann = re.sub(r"\s+", " ", chunks[j]).strip()
                    anc = re.sub(r"\s+", " ", (a_chunks[j].get('text',''))).strip()
                    if ann != anc:
                        print(f"[warn] chunk text differs at ex {ex_idx}, chunk {j}")
                        break

        prompt = build_annotation_prompt(chunks)
        try:
            resp = chat(prompt, model=annotator_model, max_tokens=2048)  # type: ignore
        except Exception as e:
            print(f"[warn] annotation failed for example {ex_idx}: {e}")
            resp = "{}"
        blob = _extract_json_blob(resp or "{}") or {}
        # Normalize to structure: {chunk_index: {function_tags: [...]}}
        per_chunk = {}
        for i in range(len(chunks)):
            node = blob.get(str(i), blob.get(i, {})) or {}
            tags = node.get('function_tags', [])
            if isinstance(tags, str):
                tags = [tags]
            per_chunk[str(i)] = {"function_tags": _sanitize_tags(tags)}
        results.append({
            "example_index": ex_idx,
            "num_chunks": len(chunks),
            "categories": per_chunk,
        })

    # Derive model tag for filename
    if not model_name:
        # Try to infer from annotated filename
        base = os.path.basename(annotated_path)
        m = re.search(r"generated_data_annotated_(.+)\.json", base)
        model_tag = m.group(1) if m else 'model'
    else:
        model_tag = model_name.replace('/', '-')
    out_path = os.path.join(output_dir, f"chunk_categories_{model_tag}.json")
    payload = {
        "model": model_name,
        "annotated_path": annotated_path,
        "anchors_check_path": anchors_path_for_check,
        "annotator_model": annotator_model,
        "tagset": TA_FUNCTION_TAGS,
        "examples": results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved chunk categories to {out_path}")
    return out_path


def _build_paths(model_name: str) -> (str, str, str):
    model_tag = model_name.replace('/', '-')
    annotated_path = os.path.join('generated_data', f'generated_data_annotated_{model_tag}.json')
    anchors_path = os.path.join('generated_data', f'steering_anchors_{model_tag}.json')
    output_dir = 'generated_data'
    return annotated_path, anchors_path, output_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Categorize CoT chunks with TA-style function tags (from annotated CoT)")
    p.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--annotated_path", type=str, default=None)
    p.add_argument("--anchors_check_path", type=str, default=None, help="Optional anchors JSON to parity-check chunking")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--annotator_model", type=str, default="qwen/qwen2.5-3b-instruct", help="Annotator model (OpenRouter id). Default: Qwen2.5-3B-Instruct")
    p.add_argument("--max_examples", type=int, default=None)
    args = p.parse_args()

    annotated_path, anchors_path, default_out = _build_paths(args.model)
    if args.annotated_path:
        annotated_path = args.annotated_path
    anchors_check = args.anchors_check_path or anchors_path
    out_dir = args.output_dir or default_out

    categorize_examples(
        annotated_path=annotated_path,
        output_dir=out_dir,
        annotator_model=args.annotator_model,
        max_examples=args.max_examples,
        anchors_path_for_check=anchors_check,
    )
