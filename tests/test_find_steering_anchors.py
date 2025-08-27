import os
import re

import pytest


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def test_split_solution_into_chunks_and_ranges():
    # Reference implementation mirroring utils.split_solution_into_chunks/get_chunk_ranges
    def ref_split_solution_into_chunks(solution_text: str):
        # Remove think tags if present
        if "<think>" in solution_text:
            solution_text = solution_text.split("<think>", 1)[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>", 1)[0].strip()

        sentence_ending_tokens = [".", "?", "!"]
        paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

        chunks = []
        current_chunk = ""
        i = 0
        while i < len(solution_text):
            current_chunk += solution_text[i]

            is_paragraph_end = False
            for pattern in paragraph_ending_patterns:
                if i + len(pattern) <= len(solution_text) and solution_text[i:i+len(pattern)] == pattern:
                    is_paragraph_end = True
                    break

            is_sentence_end = False
            if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
                next_char = solution_text[i + 1]
                if next_char == " " or next_char == "\n":
                    is_sentence_end = True

            if is_paragraph_end or is_sentence_end:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            i += 1

        # Merge small chunks (<10 chars)
        i = 0
        while i < len(chunks):
            if len(chunks[i]) < 10:
                if i == len(chunks) - 1:
                    if i > 0:
                        chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                        chunks.pop(i)
                else:
                    chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                    chunks.pop(i)
                    if i == 0 and len(chunks) == 1:
                        break
                    continue
            i += 1
        return chunks

    def ref_get_chunk_ranges(full_text: str, chunks: list):
        ranges = []
        current_pos = 0
        for ch in chunks:
            pos = full_text.find(ch, current_pos)
            assert pos >= 0, "Chunk should be found in full text"
            ranges.append((pos, pos + len(ch)))
            current_pos = pos + len(ch)
        return ranges

    cot = (
        "I think about A. Then I do B.\n"
        "Now conclusion C! Finally D."
    )
    chunks = ref_split_solution_into_chunks(cot)
    ranges = ref_get_chunk_ranges(cot, chunks)

    assert len(chunks) >= 3, "Expected at least 3 chunks from punctuation boundaries"
    assert len(chunks) == len(ranges), "Each chunk should have a char span"

    # Verify the spans slice back to the same text (up to whitespace normalization)
    for ch, (s, e) in zip(chunks, ranges):
        assert 0 <= s < e <= len(cot)
        sliced = cot[s:e]
        assert norm_spaces(sliced) == norm_spaces(ch)


@pytest.mark.parametrize("layer_idx", [0])
def test_compute_chunk_vectors_alignment_tiny_model(layer_idx):
    # Use a very small HF model to keep the test light
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import pytest as _pytest
    transformers = _pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Import the function under test
    from experiments.find_steering_anchors import compute_chunk_vectors_for_example, _encode_full
    from utils import split_solution_into_chunks

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    class HFWrap:
        def __init__(self, m):
            self.model = m

    wrap = HFWrap(model)

    # Build a synthetic example mirroring repo fields
    cot = (
        "I think about A. Then I do B.\n"
        "Now conclusion C! Finally D."
    )
    model_input_prompt = "<User> Solve step by step.\n<Assistant><think>\n"
    ex = {
        "prompt": "Dummy problem?",
        "cot": cot,
        "model_input_prompt": model_input_prompt,
        "thought_anchor_idx": 1,
    }

    device = "cpu"
    res = compute_chunk_vectors_for_example(
        wrap,
        tokenizer,
        ex,
        layer_idx=layer_idx,
        device=device,
    )
    assert res is not None, "Expected result for well-formed example"
    assert "chunks" in res and isinstance(res["chunks"], list)

    chunks = split_solution_into_chunks(cot)
    assert res["num_chunks"] == len(chunks)
    assert len(res["chunks"]) == len(chunks)

    # Verify token spans map back to approximately the chunk text
    full_text = model_input_prompt + cot
    input_ids, attn, offsets = _encode_full(tokenizer, full_text, device)

    for payload in res["chunks"]:
        idx = payload["chunk_index"]
        ch_text = chunks[idx]
        t0, t1 = payload["token_span"]
        assert 0 <= t0 < t1 <= len(offsets)
        c0 = offsets[t0][0]
        c1 = offsets[t1 - 1][1]
        recovered = full_text[c0:c1]

        # After removing the model_input_prompt prefix, recovered should include the chunk text.
        recovered_cot_portion = recovered[len(model_input_prompt):] if recovered.startswith(model_input_prompt) else recovered
        # Normalize whitespace for a robust comparison
        assert norm_spaces(ch_text) in norm_spaces(recovered_cot_portion), (
            f"Token span [{t0},{t1}) did not align with chunk {idx}:\n"
            f"chunk='{ch_text}'\nrecovered='{recovered_cot_portion}'"
        )
