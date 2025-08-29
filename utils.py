import dotenv
dotenv.load_dotenv("../.env")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import time
import random
import torch.nn as nn
import openai
import anthropic
import os
from openai import OpenAI
import json
import re
import numpy as np
import traceback
from typing import List, Tuple, Optional, Dict
from datasets import load_dataset


def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif "deepseek" in model or "gemini" in model or "qwen" in model or "meta-llama" in model:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        raise ValueError(f"Model {model} is not supported. Please use a valid model name.")

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping.get(model, model),
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    if thinking_response is not None:
                        return f"<think>{thinking_response}\n</think>\n{answer_response}"
                    else:
                        return answer_response
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            time.sleep(20)

    return None


def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def get_label_positions(annotated_thinking, response_text, tokenizer):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [category-name] text [end-section]
    pattern = r'["(\S+?)"](.*?)["end-section"]'
    matches = list(re.finditer(pattern, annotated_thinking, re.DOTALL))
    
    # Create character to token mapping once
    char_to_token = get_char_to_token_map(response_text, tokenizer)
    
    for match in matches:
        label = match.group(1).strip()
        text = match.group(2).strip()
        
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        text_pos = response_text.find(text)
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end))
    
    return label_positions

def _configure_nnsight_compile(device: str) -> None:
    """Enable compile for CUDA; disable for MPS/CPU to avoid warnings.
    Controlled via NNSIGHT_COMPILE env var.
    """
    try:
        if isinstance(device, str) and device.lower().startswith("cuda"):
            os.environ["NNSIGHT_COMPILE"] = "1"
        else:
            os.environ["NNSIGHT_COMPILE"] = "0"
    except Exception:
        pass


def _is_nnsight_model(model) -> bool:
    """Best-effort check whether `model` is an nnsight LanguageModel.
    We detect attributes used by nnsight such as `generator` and `generate` context manager.
    """
    try:
        return hasattr(model, "generator") and hasattr(model, "generate") and hasattr(model, "model")
    except Exception:
        return False


def generate_with_model(model, tokenizer, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **gen_kwargs):
    """Run text generation robustly across nnsight/HF backends.

    Returns the raw generation output object (tensor-like or ModelOutput). Use
    `decode_generate_outputs` to obtain text reliably.
    """
    if _is_nnsight_model(model):
        # nnsight path: use context manager and save the generated sequences
        payload = {"input_ids": input_ids, "attention_mask": attention_mask}
        try:
            with model.generate(payload, **gen_kwargs):
                outputs = model.generator.output.save()
            return outputs
        except Exception:
            # As a fallback, try direct call if available
            return model.generate(**{**payload, **gen_kwargs})
    # HF path
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)


def _safe_get_first_sequence(outputs):
    """Extract the first sequence from various output types.
    Supports: HF `GenerateOutput` (has `.sequences`), plain tensors/lists, and nnsight saved outputs.
    """
    # HF GenerateOutput or similar dataclass
    if hasattr(outputs, "sequences"):
        return outputs.sequences[0]
    # List/tuple of sequences or 2D tensor
    try:
        return outputs[0]
    except Exception:
        # Some tracer objects may expose `.item()` or similar; give up to caller
        return outputs


def coerce_ids_to_list(ids_like):
    """Convert a tensor/array-like of token ids to a Python list of ints safely."""
    try:
        if hasattr(ids_like, "detach"):
            return ids_like.detach().to("cpu").to(torch.long).tolist()
        if hasattr(ids_like, "cpu"):
            return ids_like.cpu().detach().to(torch.long).tolist()
        # Iterable fallback
        return [int(x) for x in list(ids_like)]
    except Exception:
        return ids_like


def decode_generate_outputs(tokenizer, outputs, *, skip_special_tokens: bool = True) -> str:
    """Decode the first generated sequence from `outputs` into text robustly."""
    seq = _safe_get_first_sequence(outputs)
    try:
        ids = coerce_ids_to_list(seq)
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    except Exception:
        # Last-resort: let tokenizer try to handle raw `seq`
        try:
            return tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
        except Exception:
            return ""


def load_model_and_vectors(device="cuda:0", load_in_8bit=False, compute_features=True, normalize_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", base_model_name=None):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        normalize_features (bool): If True, normalize the feature vectors
        return_steering_vector_set (bool): If True, return the steering vector set
        model_name (str): Name/path of the model to load
        base_model_name (str): Name/path of the base model to load
    """
    # Configure compilation behavior before constructing the model
    _configure_nnsight_compile(device)
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    # Explicitly disable compile config on unsupported devices to silence warnings
    try:
        if not (isinstance(device, str) and device.lower().startswith("cuda")):
            if hasattr(model, "compile_config"):
                model.compile_config = None
    except Exception:
        pass
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if base_model_name is not None:
        base_model = LanguageModel(base_model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
        try:
            if not (isinstance(device, str) and device.lower().startswith("cuda")):
                if hasattr(base_model, "compile_config"):
                    base_model.compile_config = None
        except Exception:
            pass
    
        base_model.generation_config.temperature=None
        base_model.generation_config.top_p=None
        base_model.generation_config.do_sample=False
        
        base_tokenizer = base_model.tokenizer

        if "llama" in base_model_name.lower():
            base_tokenizer.pad_token_id = base_tokenizer.finetune_right_pad_id
            base_tokenizer.pad_token = base_tokenizer.finetune_right_pad
            base_tokenizer.padding_side = "right"
        else:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
            base_tokenizer.pad_token = base_tokenizer.eos_token
            base_tokenizer.padding_side = "left"

    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # Prefer project-local generated vectors; avoid coupling to refs/
    # Primary path (project-owned artifacts):
    local_vector_path = f"generated_data/mean_vectors_{model_id}.pt"
    # Legacy fallback (reference repo artifacts):
    legacy_vector_path = f"refs/steering-thinking-llms/train-steering-vectors/results/vars/mean_vectors_{model_id}.pt"

    mean_vectors_dict = {}
    feature_vectors = {}

    if os.path.exists(local_vector_path):
        mean_vectors_dict = torch.load(local_vector_path)
    elif os.path.exists(legacy_vector_path):
        print(
            f"Warning: Loading mean vectors from refs ({legacy_vector_path}). "
            f"Consider copying to {local_vector_path} to decouple from refs/."
        )
        mean_vectors_dict = torch.load(legacy_vector_path)
    
    if mean_vectors_dict and compute_features:
        # Compute feature vectors by subtracting overall mean
        feature_vectors = {}
        feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']
        
        for label in ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]:

            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

            if normalize_features:
                for label in feature_vectors:
                    for layer in range(model.config.num_hidden_layers):
                        feature_vectors[label][layer] = feature_vectors[label][layer] * (feature_vectors["overall"][layer].norm() / feature_vectors[label][layer].norm())
    elif not mean_vectors_dict:
        print(f"No mean vectors found for {model_name}. You can save to {local_vector_path}.")

    if base_model_name is not None and compute_features:
        return model, tokenizer, base_model, base_tokenizer, feature_vectors
    elif base_model_name is not None and not compute_features:
        return model, tokenizer, base_model, base_tokenizer, mean_vectors_dict
    elif base_model_name is None and compute_features:
        return model, tokenizer, feature_vectors
    else:
        return model, tokenizer, mean_vectors_dict

def custom_generate_steering(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steering_config, steer_positive=False):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors containing steering_vector_set
        steer_positive: If True, steer towards the label, if False steer away
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if feature_vectors is not None:       
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]
     

            if steer_positive:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in pos_layers:         
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
            else:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in neg_layers:         
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
        
        outputs = model.generator.output.save()
                    
    return outputs


def process_batch_annotations(thinking_processes):
    """Annotate a batch of reasoning chains using the 7-category reasoning framework."""
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = chat(f"""
        Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

        Available labels:
        0. initializing -> The model is rephrasing the given task and states initial thoughts.
        1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
        2. adding-knowledge -> The model is enriching the current approach with recalled facts.
        3. example-testing -> The model generates examples to test its current approach.
        4. uncertainty-estimation -> The model is stating its own uncertainty.
        5. backtracking -> The model decides to change its approach.

        The reasoning chain to analyze:
        {thinking}

        Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
        """)
        annotated_responses.append(annotated_response)
    
    return annotated_responses

def get_batched_message_ids(tokenizer, messages_list, device):
    # First get the max length by encoding each message individually
    max_token_length = max([len(tokenizer.encode(msg, return_tensors="pt")[0]) for msg in messages_list])
    input_ids = torch.cat([
        tokenizer.encode(msg, padding="max_length", max_length=max_token_length, return_tensors="pt").to(device) 
        for msg in messages_list
    ])

    return input_ids

def process_saved_responses_batch(responses_list, tokenizer, model, device):
    """Get layer activations for a batch of saved responses without generation"""
    tokenized_responses = get_batched_message_ids(tokenizer, responses_list, device)
    
    # Process the inputs through the model to get activations
    layer_outputs = []
    with model.trace(
        {
            "input_ids": tokenized_responses, 
            "attention_mask": (tokenized_responses != tokenizer.pad_token_id).long()
        }
    ) as tracer:
        
        # Capture layer outputs
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [x.cpu().detach().to(torch.float32) for x in layer_outputs]

    batch_layer_outputs = []
    
    for batch_idx in range(len(responses_list)):
        # get length of padding tokens
        attention_mask = (tokenized_responses[batch_idx] != tokenizer.pad_token_id).long()
        padding_length = (attention_mask.squeeze() == 0).sum().item()
        
        # Slice out just the non-padded activations for this example across all layers
        example_outputs = torch.stack([
            layer_output[batch_idx][padding_length:] 
            for layer_output in layer_outputs
        ])
        
        batch_layer_outputs.append(example_outputs)
    
    return batch_layer_outputs

def extract_thinking_process_and_answer(response_text: str, prompt_len: int) -> Tuple[str, str]:
    generated_text = response_text[prompt_len:].strip()

    cot = ""
    answer = ""

    # Prefer parsing CoT by <think> tags if present
    think_start_tag = "<think>"
    think_end_tag = "</think>"

    think_start_idx = generated_text.find(think_start_tag)
    think_end_idx = generated_text.find(think_end_tag)

    if think_start_idx != -1 and think_end_idx != -1 and think_end_idx > think_start_idx:
        cot = generated_text[think_start_idx + len(think_start_tag) : think_end_idx].strip()
        tail_for_answer = generated_text[think_end_idx + len(think_end_tag) :]
    elif think_start_idx != -1 and think_end_idx == -1:
        cot = generated_text[think_start_idx + len(think_start_tag) :].strip()
        tail_for_answer = generated_text
    else:
        # No think tags; treat entire generated text as tail for answer search
        tail_for_answer = generated_text

    # Thought Anchors-style: extract \boxed{...} anywhere in the tail (or whole text as fallback)
    boxed = extract_boxed_answers(tail_for_answer)
    if boxed and boxed[0] is not None:
        answer = boxed[0]
    else:
        # Fallback: if we had a closed </think>, use tail as answer; otherwise empty
        if think_start_idx != -1 and think_end_idx != -1 and think_end_idx > think_start_idx:
            answer = tail_for_answer.strip()
        else:
            answer = ""

    return cot, answer

steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "backtracking": {"vector_layer": 17, "pos_layers": [17], "neg_layers": [17], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 15, "pos_layers": [15], "neg_layers": [15], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 18, "pos_layers": [18], "neg_layers": [18], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "backtracking": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1},
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 24, "pos_layers": [24], "neg_layers": [24], "pos_coefficient": 1, "neg_coefficient": 1},
    }
}

def get_chunk_ranges(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0

    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()

        # Try to find the chunk in the full text
        chunk_start = -1

        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()

            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i : i + len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r"\s+", " ", text_window).strip()

                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break

                # If not found with window, try word by word matching
                if i == current_pos + 100:  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(
                            r"\b" + re.escape(chunk_words[0]) + r"\b",
                            full_text[j : j + len(chunk_words[0]) + 5],
                        ):
                            # Check if subsequent words match
                            match_text = full_text[j : j + len(normalized_chunk) + 30]
                            normalized_match = re.sub(r"\s+", " ", match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break

        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue

        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r"\s+", "", chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(
            r"\s+", "", full_text_from_start[: len(chunk) + 50]
        )  # Remove all whitespace

        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break

        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i

        chunk_end += 1  # Include the last character
        current_pos = chunk_end

        chunk_ranges.append((chunk_start, chunk_end))

    return chunk_ranges


def get_chunk_token_ranges(
    text: str, chunk_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    chunk_token_ranges = []

    for chunk_start, chunk_end in chunk_ranges:
        chunk_start_token = tokenizer.encode(text[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        chunk_end_token = tokenizer.encode(text[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        chunk_token_ranges.append((chunk_start_token_idx, chunk_end_token_idx))

    return chunk_token_ranges


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed{", text)]

    if not boxed_starts:
        return [""]

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers if answers else [""]


def normalize_answer(answer: str, use_sympy: bool = False) -> str:
    """
    Get the final normalized and cleaned version of an answer.
    This function combines all normalization steps used in check_answer.

    Args:
        answer: The answer string to normalize
        use_sympy: Whether to use sympy to normalize the answer

    Returns:
        The normalized answer string
    """
    # First apply basic LaTeX normalization
    normalized = normalize_latex(answer)

    # Also prepare the answer for sympy if applicable
    if use_sympy:
        try:
            sympy_ready = prepare_latex_for_sympy(answer)
            if sympy_ready != normalized and len(sympy_ready) > 0:
                return sympy_ready
        except Exception:
            pass

    return normalized

# -----------------------------------------------------------------------------
# Steering/logit utilities for notebooks (3a/3aii/reporting)
# -----------------------------------------------------------------------------

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def forward_with_logits(model, *, input_ids, attention_mask):
    """Return logits from a forward pass, projecting via lm_head if needed.

    Executed under torch.no_grad() to avoid building graphs during analysis.
    """
    import contextlib
    with torch.no_grad() if torch is not None else contextlib.nullcontext():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            head_owner = model if hasattr(model, "lm_head") else (model.model if hasattr(model.model, "lm_head") else None)
            if head_owner is None:
                raise AttributeError("Could not find lm_head on model or model.model")
            logits = head_owner.lm_head(last_hidden)
        return logits


def kl_from_logits(avg_logits_p, avg_logits_q):
    """Token-averaged KL between two average logits tensors (P || Q)."""
    logp = torch.log_softmax(avg_logits_p, dim=-1)
    logq = torch.log_softmax(avg_logits_q, dim=-1)
    p = torch.exp(logp)
    return torch.sum(p * (logp - logq), dim=-1)


def logits_with_steer_full(
    model,
    *,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    beta: float,
    layer_idx: int,
    steer_vec: "torch.Tensor",
    target_pos: int,
):
    """Apply an RMS-scaled steering delta at a single position in a layer, then return logits."""
    backbone = getattr(model.model, "model", model.model)
    target = backbone.layers[int(layer_idx)]
    d = steer_vec.detach().to(torch.float32)
    d = d / (d.norm() + 1e-12)

    def hook(module, inputs, output):
        out = output
        try:
            if isinstance(out, tuple):
                h = out[0].clone(); rest = out[1:]
            else:
                h = out.clone(); rest = tuple()
            pos = int(target_pos)
            if pos < 0 or pos >= h.shape[1]:
                return output
            h_slice_fp32 = h[:, pos:pos+1, :].to(torch.float32)
            rms = torch.sqrt(torch.mean(h_slice_fp32 ** 2) + 1e-20)
            delta_fp32 = (float(beta) * rms) * d.view(1, 1, -1)
            h[:, pos:pos+1, :] = (h_slice_fp32 + delta_fp32).to(h.dtype)
            return (h,) + rest if isinstance(out, tuple) else h
        except Exception:
            return output

    handle = target.register_forward_hook(hook)
    try:
        # no_grad handled inside forward_with_logits
        return forward_with_logits(model, input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()


def find_chunk_start_token(tokenizer, question: str, prefix_text: str, chunk_text: str, device: str):
    """Build and encode the forced prompt, returning tokenized spans and indices."""
    prompt_prefix = (
        "Solve the following problem step by step. You MUST put your final answer in \\boxed{}.\n\n"
        + f"Problem: {question}\n\n"
        + "Solution:\n<think>\n"
        + (prefix_text or "")
    )
    enc_prefix = tokenizer(prompt_prefix, return_tensors="pt")
    ids_pref = enc_prefix["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (getattr(tokenizer, "eos_token_id", 0) or 0)
    am_pref = (ids_pref != pad_id).long().to(device)
    enc_full = tokenizer(prompt_prefix + (chunk_text or ""), return_tensors="pt")
    ids_full = enc_full["input_ids"].to(device)
    am_full = (ids_full != pad_id).long().to(device)
    full_ids = ids_full[0].tolist()
    chunk_seq = tokenizer(chunk_text or "", return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    s_idx = -1
    for pos in range(max(0, ids_pref.shape[-1]-4), len(full_ids)-len(chunk_seq)+1):
        if full_ids[pos:pos+len(chunk_seq)] == chunk_seq:
            s_idx = pos
            break
    if s_idx == -1:
        s_idx = int(ids_pref.shape[-1])
    return ids_pref, am_pref, ids_full, am_full, int(s_idx), len(chunk_seq)


def compute_kl_curve_for_chunk(
    model,
    tokenizer,
    example: Dict,
    anchors_ex: Dict,
    *,
    layer_idx: int,
    betas: "np.ndarray",
    device: str,
    chunk_index: Optional[int] = None,
) -> List[float]:
    """Compute a KL(beta) curve for a specific chunk using its steer vector."""
    import numpy as _np  # local import to avoid top-level dependency
    cot_text = example.get("cot") or ""
    try:
        chunks = split_solution_into_chunks(cot_text)
    except Exception:
        chunks = []
    if not chunks:
        return []
    idx = int(chunk_index) if chunk_index is not None else 0
    if idx < 0 or idx >= len(chunks):
        return []
    prefix_text = "\n".join(chunks[:idx])
    chunk_text = chunks[idx]
    ids_pref, am_pref, ids_full, am_full, s_idx, n_steps = find_chunk_start_token(
        tokenizer, example.get("prompt", ""), prefix_text, chunk_text, device
    )
    logits_full = forward_with_logits(model, input_ids=ids_full, attention_mask=am_full)
    seq_len = int(logits_full.shape[1])
    start = max(0, int(s_idx) - 1)
    n_eff = max(0, min(int(n_steps), int(seq_len - start)))
    if n_eff == 0:
        return []
    base_steps = logits_full[:, start:start + n_eff, :]
    base_avg = base_steps.mean(dim=1)
    # per-chunk vector
    vec_by_idx = {int(ch.get("chunk_index", 0)): ch.get("vector", []) for ch in anchors_ex.get("chunks", [])}
    v = torch.tensor(vec_by_idx.get(int(idx), []), dtype=torch.float32, device=device)
    if v.numel() == 0:
        return []
    y_curve: List[float] = []
    for b in betas:
        logs = logits_with_steer_full(
            model,
            input_ids=ids_full,
            attention_mask=am_full,
            beta=float(b),
            layer_idx=int(layer_idx),
            steer_vec=v,
            target_pos=int(start),
        )
        steered_steps = logs[:, start:start + n_eff, :]
        steered_avg = steered_steps.mean(dim=1)
        kl = kl_from_logits(steered_avg, base_avg)
        y_curve.append(float(kl.item()))
    return y_curve


# -----------------------------------------------------------------------------
# Amplitude normalization helpers
# -----------------------------------------------------------------------------
from typing import Iterable

def normalize_minmax(values: Iterable[float]) -> list[float]:
    """Normalize a 1D iterable to [0,1] via min-max; nans become 0.0.

    If all finite values are equal or no finite values exist, returns all 0.0.
    """
    import math
    vals = [float(v) for v in values]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in vals]
    vmin = min(finite)
    vmax = max(finite)
    rng = vmax - vmin
    if rng <= 0:
        return [0.0 for _ in vals]
    out = []
    for v in vals:
        if math.isfinite(v):
            out.append((v - vmin) / rng)
        else:
            out.append(0.0)
    return out


def normalize_by_group(keys: Iterable[int], values: Iterable[float]) -> list[float]:
    """Normalize values to [0,1] per group key using min-max per key.

    keys: group identifier (e.g., example_index) for each value.
    values: numeric values (e.g., amplitudes) to normalize.
    Returns a list of normalized values aligned to input order.
    """
    import math
    ks = [int(k) for k in keys]
    vs = [float(v) for v in values]
    groups = {}
    for i,(k,v) in enumerate(zip(ks,vs)):
        groups.setdefault(k, []).append(v)
    mins = {}; maxs = {}
    for k, arr in groups.items():
        finite = [x for x in arr if math.isfinite(x)]
        if finite:
            mins[k] = min(finite)
            maxs[k] = max(finite)
        else:
            mins[k] = 0.0; maxs[k] = 0.0
    out=[]
    for k,v in zip(ks,vs):
        rng = maxs[k]-mins[k]
        if not math.isfinite(v) or rng<=0:
            out.append(0.0)
        else:
            out.append((v - mins[k])/rng)
    return out



def check_answer(answer: str, gt_answer: str) -> bool:
    """
    Check if the generated answer matches the ground truth answer
    after normalizing LaTeX formatting.

    Args:
        answer: The generated answer to check
        gt_answer: The ground truth answer to compare against

    Returns:
        True if the answers match after normalization, False otherwise
    """
    # Factorial-aware punctuation cleanup prior to normalization
    answer = cleanup_answer_punctuation(answer, gt_answer)
    gt_answer = cleanup_answer_punctuation(gt_answer, gt_answer)

    # Normalize both answers
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)

    # First check if normalized strings match
    if normalized_answer == normalized_gt_answer:
        return True

    # If string comparison fails, try mathematical equivalence
    try:
        return get_latex_equivalent(answer, gt_answer)
    except Exception as e:
        # If SymPy parsing fails, fall back to string comparison result
        return False


def get_latex_equivalent(answer0, answer1):
    """
    Check if two LaTeX expressions are mathematically equivalent using SymPy.

    Args:
        answer0: First LaTeX expression
        answer1: Second LaTeX expression

    Returns:
        True if expressions are mathematically equivalent, False otherwise
    """
    try:
        import warnings
        # Suppress antlr-related UserWarnings if antlr4 runtime isn't installed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="antlr4.error.ErrorListener module is not installed")
            from sympy.parsing.latex import parse_latex
        import sympy

        # Clean up the LaTeX expressions for parsing
        answer0 = prepare_latex_for_sympy(answer0)
        answer1 = prepare_latex_for_sympy(answer1)

        # Parse the LaTeX expressions
        expr1 = parse_latex(answer0)
        expr2 = parse_latex(answer1)

        # Check if they are mathematically identical
        equals = expr1.equals(expr2)
        # print(f"First: {answer0}, Second: {answer1}: equals={equals}")
        return equals
    except Exception as e:
        # print(f"Error comparing expressions: {e}")
        return False


def prepare_latex_for_sympy(latex_str):
    """
    Prepare a LaTeX string for SymPy parsing by removing unsupported commands
    and simplifying the expression.
    """
    if not isinstance(latex_str, str):
        return str(latex_str)

    # Remove \boxed{} command
    latex_str = re.sub(r"\\boxed\{(.*?)" + "}", r"\1", latex_str)

    # Replace common LaTeX commands that SymPy doesn't support
    replacements = {
        r"\\dfrac": r"\\frac",
        r"\\tfrac": r"\\frac",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\div": r"/",
        r"\\left": r"",
        r"\\right": r"",
        r"\\textbf": r"",
        r"\\text": r"",
        r"\\mathrm": r"",
        r"\\!" : r"",
        r",": r"",
    }

    for old, new in replacements.items():
        latex_str = re.sub(old, new, latex_str)

    return latex_str


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.

    Args:
        latex_str: The LaTeX string to normalize

    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()

    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas
    normalized = normalized.replace("{,}", "")

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)" + "}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)" + "}", r"\1", normalized)

    # Punctuation cleanup (general):
    # - Collapse runs of '!' to a single '!'
    # - Remove trailing '.', '?', or ellipsis characters
    normalized = re.sub(r"!{2,}", "!", normalized)
    normalized = re.sub(r"[\.\?…]+$", "", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized


def cleanup_answer_punctuation(answer: str, gt_answer: Optional[str] = None) -> str:
    """
    Clean punctuation artifacts from model answers.

    - If ground truth has no factorial ('!'), remove all '!' from the candidate.
    - If GT has a factorial, collapse repeated '!' runs to a single '!'.
    - Remove trailing '.', '?', and ellipsis characters.
    """
    if answer is None:
        return ""
    s = str(answer)
    # Trailing sentence punctuation
    s = re.sub(r"[\.\?…]+$", "", s)
    gt_has_factorial = bool(gt_answer and ('!' in str(gt_answer)))
    if gt_has_factorial:
        s = re.sub(r"!{2,}", "!", s)
    else:
        s = s.replace("!", "")
    return s


def split_solution_keep_spacing(solution_text: str) -> List[str]:
    """
    Split a solution into chunks while preserving spacing.
    """
    # Define patterns for chunk boundaries
    sentences = split_solution_into_chunks(solution_text)
    chunk_ranges = get_chunk_ranges(solution_text, sentences)
    sentences_w_spacing = [
        solution_text[chunk_range[0] : chunk_range[1]] for chunk_range in chunk_ranges
    ]
    return sentences_w_spacing


def sanity_check_sentences(sentences_w_spacing, dir_problem, text):
    # Sanity check
    # (the chunks.json removes "\n"also omits the final sentence of the CoT)
    fp_sentences_json = os.path.join(dir_problem, "chunks.json")
    with open(fp_sentences_json, "r") as f:
        sentence_data = json.load(f)

    sentences_og = sentence_data["chunks"]

    assert len(sentences_w_spacing) == len(sentences_og) - 1
    assert (
        sentences_og[-2].strip()
        in sentences_w_spacing[-1].replace("\n", " ").replace("  ", " ").strip()
    )
    for sentence in sentences_w_spacing:
        assert sentence in text


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.

    - Excludes the prompt by stripping the text before/after <think> tags if present.
    - Uses sentence/paragraph boundaries and merges very short fragments.

    Args:
        solution_text: The full solution text (may include <think>…</think>)

    Returns:
        List of chunks (sentences/paragraphs) from CoT content only.
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into chunks
    chunks = []
    current_chunk = ""

    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # # Add the last chunk if not empty
    # if current_chunk.strip():
    #     chunks.append(current_chunk.strip())
    #     chunk_idxs.append(len(solution_text) - 1)  # Add last index

    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            # If this is the last chunk, merge with previous chunk if possible
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            # Otherwise merge with the next chunk
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
                # Don't increment i since we need to check the new merged chunk
            # If we're at the beginning and there's only one chunk, just keep it
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    # chunk_boundaries = [(chunk_idxs[i], chunk_idxs[i + 1]) for i in range(len(chunk_idxs) - 1)]
    # chunk_boundaries.append((chunk_idxs[-1], len(solution_text)))

    # if get_idxs:
    #     return chunks, chunk_boundaries
    # else:
    return chunks


def load_math_problems(
    problem_type: Optional[str] = None,
    level: Optional[str] = None,
    num_problems: Optional[int] = None,
    split: str = "train",
    include_problems: Optional[List[int]] = None,
) -> List[Tuple[int, Dict]]:
    """
    Load problems from the MATH dataset with optional filtering.

    Args:
        problem_type: Type of problems to filter by (if None, use all types)
        level: Level of problems to filter by (if None, use all levels)
        num_problems: Number of problems to sample (if None, use all problems)
        split: Dataset split to use ('train' or 'test')

    Returns:
        List of problems with their original indices
    """
    try:
        # Load from Hugging Face dataset
        math_dataset = load_dataset("fdyrd/math")
        dataset_split = math_dataset[split]

        # Add original indices to problems
        indexed_problems = [
            (
                i,
                {
                    "problem": item["problem"],
                    "level": item["level"],
                    "type": item["type"],
                    "gt_solution": item["solution"],
                },
            )
            for i, item in enumerate(dataset_split)
        ]

        # Extract ground truth answers
        for i, problem in indexed_problems:
            gt_boxed_answers = extract_boxed_answers(problem["gt_solution"])
            gt_answer = gt_boxed_answers[0] if gt_boxed_answers else ""
            problem["gt_answer"] = gt_answer

        # Filter by type if specified
        if problem_type is not None:
            indexed_problems = [
                (i, problem)
                for i, problem in indexed_problems
                if problem.get("type") == problem_type
            ]

        # Filter by level if specified
        if level is not None:
            indexed_problems = [
                (i, problem) for i, problem in indexed_problems if problem.get("level") == level
            ]

        # Sample if needed
        if (
            num_problems is not None
            and include_problems is None
            and num_problems < len(indexed_problems)
        ):
            indexed_problems = random.sample(indexed_problems, num_problems)

        if level:
            print(f"Filtered to level: {level}")
        if problem_type:
            print(f"Filtered to type: {problem_type}")

        return indexed_problems
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []
