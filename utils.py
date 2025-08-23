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
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if base_model_name is not None:
        base_model = LanguageModel(base_model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
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

def extract_boxed_answers(text: str) -> List[str]:
    """Extract answers enclosed in \boxed{...} with basic brace matching.

    Returns a list of extracted contents; empty list if none.
    """
    answers: List[str] = []
    i = 0
    n = len(text)
    target = "\\boxed{"
    while i < n:
        j = text.find(target, i)
        if j == -1:
            break
        # position after opening brace
        k = j + len(target)
        depth = 1
        buf = []
        while k < n and depth > 0:
            ch = text[k]
            if ch == '{':
                depth += 1
                buf.append(ch)
            elif ch == '}':
                depth -= 1
                if depth > 0:
                    buf.append(ch)
            else:
                buf.append(ch)
            k += 1
        # depth reached 0 -> captured one box
        if depth == 0:
            answers.append(''.join(buf).strip())
            i = k
        else:
            # unmatched; stop
            break
    return answers


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

messages = [
    # Mathematical Logic
    {"role": "user", "content": "Using the numbers 4, 7, and 12, create an equation that equals 100."}, 
    {"role": "user", "content": "Find three consecutive numbers that add up to 72."}, 
    {"role": "user", "content": "If you multiply three different prime numbers to get 231, what are they?"}, 
    {"role": "user", "content": "Using the digits 1-9 exactly once, create two numbers that multiply to give the smallest possible result."}, 
    {"role": "user", "content": "What's the smallest number that when divided by 3, 4, and 5 always leaves a remainder of 2?"}, 
    {"role": "user", "content": "Using only the number 4 four times and any mathematical operations, create an expression equal to 17."}, 
    {"role": "user", "content": "Find three different positive numbers whose sum and product are the same."}, 
    {"role": "user", "content": "What's the largest three-digit number that's divisible by both 7 and 8?"}, 
    {"role": "user", "content": "Using the numbers 2, 3, 5, and 7, create an expression that equals 100."}, 
    {"role": "user", "content": "What's the smallest positive number that becomes a perfect square when increased by 14 or decreased by 14?"}, 
    {"role": "user", "content": "What's the smallest number that's both a perfect square and a perfect cube?"}, 
    {"role": "user", "content": "Using the digits 2, 4, 6, and 8 exactly once, create a fraction closest to 1."}, 
    {"role": "user", "content": "Find four different numbers whose sum and product are both 20."}, 
    {"role": "user", "content": "What's the largest number you can make using three 3s and any mathematical operations?"}, 
    {"role": "user", "content": "Find three prime numbers that add up to 100."}, 
    {"role": "user", "content": "What's the smallest number that when divided by 2,3,4,5,6 always leaves remainder 1?"}, 
    {"role": "user", "content": "Create an equation using 1,2,3,4 exactly once that equals 24."}, 
    {"role": "user", "content": "What's the largest two-digit number whose square root is a whole number?"}, 
    {"role": "user", "content": "Find three consecutive even numbers that multiply to give 48."}, 
    {"role": "user", "content": "Using five 5s and any operations, create an expression equal to 120."}, 
    {"role": "user", "content": "What's the smallest number that has exactly 10 factors?"}, 
    {"role": "user", "content": "Create two different fractions that add up to exactly 1 using digits 1-9 once."}, 
    {"role": "user", "content": "Find four consecutive numbers whose product is 3024."}, 
    {"role": "user", "content": "What's the largest three-digit palindrome divisible by 11?"}, 
    {"role": "user", "content": "Using the numbers 1,4,7,8 once each, make 24."}, 
    {"role": "user", "content": "Find three different square numbers that add up to 100."}, 
    {"role": "user", "content": "What's the smallest number that's both triangular and square?"}, 
    {"role": "user", "content": "Create an equation using 5,6,7,8 exactly once that equals 50."}, 
    {"role": "user", "content": "Find two prime numbers that add up to 100."}, 
    {"role": "user", "content": "What's the largest number under 1000 that's both a perfect square and a perfect cube?"},
    # New Mathematical Logic problems
    {"role": "user", "content": "Using only the number 9 three times and any mathematical operations, create an expression equal to 123."}, 
    {"role": "user", "content": "What's the smallest positive integer that leaves remainder 1 when divided by 2, remainder 2 when divided by 3, and remainder 3 when divided by 4?"}, 
    {"role": "user", "content": "Find the product of all prime numbers less than 20."}, 
    {"role": "user", "content": "What's the sum of all three-digit numbers that are perfect squares?"}, 
    {"role": "user", "content": "Using the digits 0-9 exactly once, create a 10-digit number divisible by 11."}, 
    {"role": "user", "content": "Find the smallest positive integer that is both a perfect square and a perfect cube."}, 
    {"role": "user", "content": "What's the largest four-digit number that is divisible by 12, 15, and 18?"}, 
    {"role": "user", "content": "Find four consecutive integers whose product equals 120."}, 
    {"role": "user", "content": "Using the numbers 3, 5, 7, and 9 exactly once and any operations, create an expression equal to 24."}, 
    {"role": "user", "content": "What's the sum of all proper divisors of 36?"}, 
    {"role": "user", "content": "Find three different positive integers whose reciprocals sum to 1."}, 
    {"role": "user", "content": "What's the largest palindromic number less than 1000 that is divisible by 13?"}, 
    {"role": "user", "content": "Using only the number 8 exactly four times and any operations, make 100."}, 
    {"role": "user", "content": "Find the next number in the sequence: 1, 3, 6, 10, 15, ..."}, 
    {"role": "user", "content": "What's the product of the first 10 prime numbers?"}, 
    {"role": "user", "content": "Using all digits 1-9 exactly once, create a 9-digit number divisible by 9."}, 
    {"role": "user", "content": "What's the smallest positive number that is divisible by all integers from 1 to 10?"}, 
    {"role": "user", "content": "Find three different primes p, q, r where p + q = r and p × q = r + 100."}, 
    {"role": "user", "content": "Using the digits 1, 2, 3, and 4 exactly once and any operations, create an expression equal to 100."}, 
    {"role": "user", "content": "What's the sum of all two-digit numbers that are both perfect squares and perfect cubes?"},

    # Spatial Reasoning
    {"role": "user", "content": "If you fold a square piece of paper in half twice and then cut a triangle in one corner, what pattern will you see when you unfold it?"}, 
    {"role": "user", "content": "If you look at a cube from the front, top, and right side, how many faces of the cube can you see in total?"}, 
    {"role": "user", "content": "If you have a red cube inside a larger transparent cube, and rotate the larger cube 90 degrees forward and then 90 degrees right, where is the red cube now?"}, 
    {"role": "user", "content": "What shape would you get if you slice a cone with a plane parallel to its side?"}, 
    {"role": "user", "content": "If you stack 27 small cubes to make a larger 3x3x3 cube and remove all corner cubes, how many small cubes remain?"}, 
    {"role": "user", "content": "If you fold a rectangular paper in half lengthwise, then in half widthwise, then cut off one corner, how many holes will appear when you unfold it?"}, 
    {"role": "user", "content": "What shape would you see if you looked directly down on a pyramid from above?"}, 
    {"role": "user", "content": "If you connect the midpoints of all sides of any quadrilateral, what shape do you always get?"}, 
    {"role": "user", "content": "If you have a cylindrical can and cut it from top to bottom, then flatten it out, what shape do you get?"}, 
    {"role": "user", "content": "If you fold a paper into a triangle, then fold it in half three times, what fraction of the original triangle's area is showing?"}, 
    {"role": "user", "content": "If you slice a cube with a plane through three corners, what shape do you get?"}, 
    {"role": "user", "content": "What shape is formed when two cylinders of equal diameter intersect at right angles?"}, 
    {"role": "user", "content": "If you fold a square paper diagonally twice and cut off the tip, what shape appears when unfolded?"}, 
    {"role": "user", "content": "How many different rectangles can you see in a 3x4 grid of squares?"}, 
    {"role": "user", "content": "If you stack cubes in a pyramid with 4 cubes on each side at the base, how many cubes total?"}, 
    {"role": "user", "content": "What's the minimum number of straight cuts needed to divide a cube into 27 smaller cubes?"}, 
    {"role": "user", "content": "If you connect the centers of all faces of a cube, what 3D shape do you get?"}, 
    {"role": "user", "content": "What shape is formed by the intersection of a sphere and a plane?"}, 
    {"role": "user", "content": "If you rotate a right triangle 360° around one of its legs as an axis, what 3D shape is formed?"}, 
    {"role": "user", "content": "How many different triangles can you make by connecting any three corners of a cube?"}, 
    {"role": "user", "content": "What shape do you get when you slice a tetrahedron parallel to one of its faces?"}, 
    {"role": "user", "content": "If you fold a circular paper in half three times and unfold, how many creases are there?"}, 
    {"role": "user", "content": "What's the maximum number of regions you can divide a circle's surface into with four straight lines?"}, 
    {"role": "user", "content": "If you look at a regular octahedron from directly above a vertex, what shape do you see?"}, 
    {"role": "user", "content": "How many faces does a shape have if every vertex connects to exactly three edges?"}, 
    {"role": "user", "content": "What shape is formed when a cone intersects with a plane parallel to its axis?"}, 
    {"role": "user", "content": "If you connect the midpoints of adjacent edges of a cube, what shape do you get?"}, 
    {"role": "user", "content": "What's the shape of the shadow cast by a cylinder when light hits it at a 45° angle?"}, 
    {"role": "user", "content": "If you slice a donut (torus) through its center, what shape do you get?"}, 
    {"role": "user", "content": "How many different squares can you see in a 3x3 grid?"},
    # New Spatial Reasoning problems
    {"role": "user", "content": "If you fold a regular hexagon along all lines connecting opposite vertices, what 3D shape do you form?"}, 
    {"role": "user", "content": "What's the maximum number of regions you can divide a sphere into with 5 great circles?"}, 
    {"role": "user", "content": "If you slice a regular tetrahedron with a plane that intersects all edges, what shape can you get?"}, 
    {"role": "user", "content": "If you project a cube onto a flat surface with light coming from directly above one vertex, what shape is the shadow?"}, 
    {"role": "user", "content": "What 3D shape is created when you rotate an equilateral triangle around one of its sides as an axis?"}, 
    {"role": "user", "content": "If you fold a square paper in eighths using only valley folds and cut one symmetric pattern on the folded edge, what's the maximum number of distinct shapes you can create when unfolded?"}, 
    {"role": "user", "content": "If you connect the center of a cube to all of its vertices, how many identical pyramids do you form?"}, 
    {"role": "user", "content": "What shape is formed at the intersection of three identical cylinders placed perpendicular to each other?"}, 
    {"role": "user", "content": "If you slice a regular octahedron with a plane through its center parallel to two of its faces, what shape do you get?"}, 
    {"role": "user", "content": "How many distinct planes of symmetry does a regular dodecahedron have?"}, 
    {"role": "user", "content": "What 3D shape is formed when you rotate a semicircle around its diameter?"}, 
    {"role": "user", "content": "If you stack identical cubes to form a 4×4×4 large cube and remove all cubes that have at least one face exposed, how many cubes remain?"}, 
    {"role": "user", "content": "What's the shape of the intersection when a cone passes through the center of a sphere?"}, 
    {"role": "user", "content": "If you connect the midpoints of all edges of a regular tetrahedron, what polyhedron do you get?"}, 
    {"role": "user", "content": "When a 4D hypercube is projected into 3D space, what 3D shape can you see?"}, 
    {"role": "user", "content": "How many different planes can be defined by selecting three vertices of a cube?"}, 
    {"role": "user", "content": "What shape is formed when you slice an icosahedron through its center parallel to two opposite faces?"}, 
    {"role": "user", "content": "If you fold a paper into a triangle, then fold in all three corners to the center, what shape do you get?"}, 
    {"role": "user", "content": "What's the maximum number of regions you can divide a cube's surface into with 5 complete circular cuts?"}, 
    {"role": "user", "content": "When light passes through a triangular prism, what 3D shape does the refracted light form?"},

    # Verbal Logic
    {"role": "user", "content": "Complete the analogy and explain why: Ocean is to wave as earth is to _____"}, 
    {"role": "user", "content": "Complete: Book is to page as movie is to _____"}, 
    {"role": "user", "content": "Complete: Sheep is to flock as star is to _____"}, 
    {"role": "user", "content": "Complete: Paint is to artist as words are to _____"}, 
    {"role": "user", "content": "Complete: Winter is to summer as night is to _____"}, 
    {"role": "user", "content": "Complete: Keyboard is to type as pencil is to _____"}, 
    {"role": "user", "content": "Complete: Cloud is to rain as smile is to _____"}, 
    {"role": "user", "content": "Complete: Tree is to forest as brick is to _____"}, 
    {"role": "user", "content": "Complete: Shoe is to foot as glove is to _____"}, 
    {"role": "user", "content": "Complete: Fire is to ash as life is to _____"}, 
    {"role": "user", "content": "Complete: Mountain is to peak as ocean is to _____"}, 
    {"role": "user", "content": "Complete: Canvas is to painter as stage is to _____"}, 
    {"role": "user", "content": "Complete: Hunger is to eat as thirst is to _____"}, 
    {"role": "user", "content": "Complete: Clock is to time as ruler is to _____"}, 
    {"role": "user", "content": "Complete: Butterfly is to cocoon as frog is to _____"}, 
    {"role": "user", "content": "Complete: Camera is to photo as pen is to _____"}, 
    {"role": "user", "content": "Complete: Desert is to oasis as space is to _____"}, 
    {"role": "user", "content": "Complete: Dawn is to dusk as birth is to _____"}, 
    {"role": "user", "content": "Complete: Sail is to boat as wing is to _____"}, 
    {"role": "user", "content": "Complete: Seed is to plant as egg is to _____"}, 
    {"role": "user", "content": "Complete: Symphony is to composer as novel is to _____"}, 
    {"role": "user", "content": "Complete: Telescope is to stars as microscope is to _____"}, 
    {"role": "user", "content": "Complete: River is to delta as tree is to _____"}, 
    {"role": "user", "content": "Complete: Lightning is to thunder as cause is to _____"}, 
    {"role": "user", "content": "Complete: Needle is to thread as key is to _____"}, 
    {"role": "user", "content": "Complete: Wheel is to car as leg is to _____"}, 
    {"role": "user", "content": "Complete: Dictionary is to words as atlas is to _____"}, 
    {"role": "user", "content": "Complete: Umbrella is to rain as sunscreen is to _____"}, 
    {"role": "user", "content": "Complete: Oxygen is to lungs as fuel is to _____"}, 
    {"role": "user", "content": "Complete: Lighthouse is to ships as compass is to _____"},
    # New Verbal Logic problems
    {"role": "user", "content": "Complete: Anchor is to ship as foundation is to _____"}, 
    {"role": "user", "content": "Complete: Ink is to printer as lead is to _____"}, 
    {"role": "user", "content": "Complete: Spark is to fire as seed is to _____"}, 
    {"role": "user", "content": "Complete: Orchestra is to conductor as team is to _____"}, 
    {"role": "user", "content": "Complete: Pupil is to eye as aperture is to _____"}, 
    {"role": "user", "content": "Complete: Fossil is to paleontologist as star is to _____"}, 
    {"role": "user", "content": "Complete: Sunlight is to photosynthesis as ingredients are to _____"}, 
    {"role": "user", "content": "Complete: Island is to archipelago as mountain is to _____"}, 
    {"role": "user", "content": "Complete: Scales is to fish as bark is to _____"}, 
    {"role": "user", "content": "Complete: Sponge is to absorb as fan is to _____"}, 
    {"role": "user", "content": "Complete: Antidote is to poison as reconciliation is to _____"}, 
    {"role": "user", "content": "Complete: Palette is to colors as menu is to _____"}, 
    {"role": "user", "content": "Complete: Passport is to country as ticket is to _____"}, 
    {"role": "user", "content": "Complete: Lyrics are to song as dialogue is to _____"}, 
    {"role": "user", "content": "Complete: Retina is to image as cochlea is to _____"}, 
    {"role": "user", "content": "Complete: DNA is to organism as blueprint is to _____"}, 
    {"role": "user", "content": "Complete: Archive is to history as seed bank is to _____"}, 
    {"role": "user", "content": "Complete: Spine is to book as keel is to _____"}, 
    {"role": "user", "content": "Complete: Thermostat is to temperature as governor is to _____"}, 
    {"role": "user", "content": "Complete: Eclipse is to shadow as mirage is to _____"},

    # Pattern Recognition
    {"role": "user", "content": "What comes next in this sequence and why: Leaf, Branch, Tree, Forest, _____"}, 
    {"role": "user", "content": "What comes next: 2, 6, 12, 20, 30, _____"}, 
    {"role": "user", "content": "What comes next: Triangle, Square, Pentagon, Hexagon, _____"}, 
    {"role": "user", "content": "What comes next: AABABC, AABAB_, _____"}, 
    {"role": "user", "content": "What comes next: Mozart, Bach, Beethoven, Chopin, _____"}, 
    {"role": "user", "content": "What comes next: Spring 1, Summer 2, Fall 4, Winter 8, Spring _____"}, 
    {"role": "user", "content": "What comes next: O T T F F S S _____"}, 
    {"role": "user", "content": "What comes next: 1, 11, 21, 1211, 111221, _____"}, 
    {"role": "user", "content": "What comes next: RED, BLUE, YELLOW, GREEN, _____"}, 
    {"role": "user", "content": "What comes next: 3, 6, 12, 24, 48, _____"}, 
    {"role": "user", "content": "What comes next: 1, 4, 9, 16, 25, _____"}, 
    {"role": "user", "content": "What comes next: JAVA, PYTHON, RUBY, SWIFT, _____"}, 
    {"role": "user", "content": "What comes next: Circle, Sphere, Square, Cube, Triangle, _____"}, 
    {"role": "user", "content": "What comes next: 1, 3, 6, 10, 15, _____"}, 
    {"role": "user", "content": "What comes next: ROYGBIV, ROYGBI_, _____"}, 
    {"role": "user", "content": "What comes next: Mercury, Venus, Earth, Mars, _____"}, 
    {"role": "user", "content": "What comes next: Do, Re, Mi, Fa, _____"}, 
    {"role": "user", "content": "What comes next: WWW, What, When, Where, _____"}, 
    {"role": "user", "content": "What comes next: 2, 3, 5, 7, 11, _____"}, 
    {"role": "user", "content": "What comes next: 1, 1, 2, 3, 5, 8, _____"}, 
    {"role": "user", "content": "What comes next: A1, B2, C3, D4, _____"}, 
    {"role": "user", "content": "What comes next: 1, 4, 7, 10, 13, _____"}, 
    {"role": "user", "content": "What comes next: Monday, Wednesday, Friday, _____"}, 
    {"role": "user", "content": "What comes next: 100, 50, 25, 12.5, _____"}, 
    {"role": "user", "content": "What comes next: PEACH, PLUM, PEAR, PAPAYA, _____"}, 
    {"role": "user", "content": "What comes next: 1, 2, 4, 8, 16, _____"}, 
    {"role": "user", "content": "What comes next: AZ, BY, CX, DW, _____"}, 
    {"role": "user", "content": "What comes next: 7, 14, 28, 56, _____"}, 
    {"role": "user", "content": "What comes next: Triangle3, Square4, Pentagon5, _____"}, 
    {"role": "user", "content": "What comes next: 1A, 2B, 6C, 24D, _____"},
    # New Pattern Recognition problems
    {"role": "user", "content": "What comes next: 3, 1, 4, 1, 5, 9, _____"}, 
    {"role": "user", "content": "What comes next: OTTFFSSE, OTTFFSSE_, _____"}, 
    {"role": "user", "content": "What comes next: 1, 8, 27, 64, _____"}, 
    {"role": "user", "content": "What comes next: JFMAMJJASON_, _____"}, 
    {"role": "user", "content": "What comes next: 1, 4, 9, 16, 25, 36, _____"}, 
    {"role": "user", "content": "What comes next: Uni, Bi, Tri, Quad, _____"}, 
    {"role": "user", "content": "What comes next: 1, 10, 100, 1000, _____"}, 
    {"role": "user", "content": "What comes next: Penny, Nickel, Dime, Quarter, _____"}, 
    {"role": "user", "content": "What comes next: 31, 28, 31, 30, 31, _____"}, 
    {"role": "user", "content": "What comes next: FIRST, SECOND, THIRD, FOURTH, _____"}, 
    {"role": "user", "content": "What comes next: MTWT_, _____"}, 
    {"role": "user", "content": "What comes next: 0, 1, 1, 2, 3, 5, 8, _____"}, 
    {"role": "user", "content": "What comes next: 4, 9, 16, 25, 36, _____"}, 
    {"role": "user", "content": "What comes next: AB, CD, EF, GH, _____"}, 
    {"role": "user", "content": "What comes next: 26, 25, 22, 17, 10, _____"}, 
    {"role": "user", "content": "What comes next: Ra, Ac, Tu, Wd, Th, _____"}, 
    {"role": "user", "content": "What comes next: I, V, X, L, _____"}, 
    {"role": "user", "content": "What comes next: NOON, LEVEL, RADAR, _____"}, 
    {"role": "user", "content": "What comes next: Quark, Electron, Atom, Molecule, _____"}, 
    {"role": "user", "content": "What comes next: Alpha, Beta, Gamma, Delta, _____"},

    # Lateral Thinking
    {"role": "user", "content": "A woman shoots her husband, then holds him underwater for 5 minutes, finally hanging him. Yet the man later goes to work happy and healthy. How is this possible?"}, 
    {"role": "user", "content": "A man pushes his car until he reaches a hotel and realizes he's bankrupt. What's going on?"}, 
    {"role": "user", "content": "The more you take away, the larger it becomes. What is it?"}, 
    {"role": "user", "content": "A man lives on the 10th floor. Every morning he takes the elevator down to the ground floor. However, when returning home, he takes the elevator to the 7th floor and walks up the stairs to the 10th floor. Why?"}, 
    {"role": "user", "content": "Two mothers and two daughters went shopping. They bought three items in total, and each got exactly one item. How is this possible?"}, 
    {"role": "user", "content": "A man walks into a restaurant, orders albatross soup, takes one taste, and kills himself. Why?"}, 
    {"role": "user", "content": "What can run but never walks, has a mouth but never talks, has a head but never weeps, has a bed but never sleeps?"}, 
    {"role": "user", "content": "A father and son get in a car crash. The father dies, and the son is rushed to the hospital. The surgeon says 'I can't operate on him, he's my son.' How is this possible?"}, 
    {"role": "user", "content": "What has keys but no locks, space but no room, and you can enter but not go in?"}, 
    {"role": "user", "content": "A man is found dead in a circular mansion. Each resident claims they were in their room at the time of murder. Why doesn't the detective believe them?"}, 
    {"role": "user", "content": "A woman has two sons who were born on the same day of the same year, but they're not twins. How is this possible?"}, 
    {"role": "user", "content": "What can you hold in your right hand, but never in your left hand?"}, 
    {"role": "user", "content": "A man rode into town on Friday, stayed three days, and rode out on Friday. How?"}, 
    {"role": "user", "content": "What gets wetter and wetter the more it dries?"}, 
    {"role": "user", "content": "How can you throw a ball as hard as you can and have it come back to you, even if it doesn't bounce off anything?"}, 
    {"role": "user", "content": "What has keys that open no locks, space but no room, and you can enter but not go in?"}, 
    {"role": "user", "content": "What breaks when you say it?"}, 
    {"role": "user", "content": "A man is trapped in a room with only two exits. One exit leads to a room made of magnifying glass that will fry him instantly. The other exit leads to a room with a fire-breathing dragon. How does he escape?"}, 
    {"role": "user", "content": "What can travel around the world while staying in a corner?"}, 
    {"role": "user", "content": "The more you take, the more you leave behind. What are they?"}, 
    {"role": "user", "content": "What has a head and a tail but no body?"}, 
    {"role": "user", "content": "A man pushes his car to a hotel and tells the owner he's bankrupt. What's happening?"}, 
    {"role": "user", "content": "What has legs but doesn't walk?"}, 
    {"role": "user", "content": "What can you catch but not throw?"}, 
    {"role": "user", "content": "What has many keys but no locks, space but no room, and you can enter but not go in?"}, 
    {"role": "user", "content": "What gets bigger when you take away from it?"},
    # New Lateral Thinking problems
    {"role": "user", "content": "A woman gave birth to two children within minutes of each other. They were born on different days in different months of different years. How is this possible?"}, 
    {"role": "user", "content": "A man wearing black clothes and a black mask walked down a country lane. Suddenly, a large black car with its lights off came around the corner and screeched to a halt. How did the driver see the man?"}, 
    {"role": "user", "content": "What can fill a room but takes up no space?"}, 
    {"role": "user", "content": "You're in a room with no windows and no doors. All you have is a table and a mirror. How do you escape?"}, 
    {"role": "user", "content": "A girl went to bed at 8 pm and woke up at 7 am, yet she got only 1 hour of sleep. How is this possible?"}, 
    {"role": "user", "content": "Brothers and sisters I have none, but this man's father is my father's son. Who is this man to me?"}, 
    {"role": "user", "content": "What starts with 'e', ends with 'e', and contains only one letter?"}, 
    {"role": "user", "content": "What can be seen once in a minute, twice in a moment, and never in a thousand years?"}, 
    {"role": "user", "content": "A woman was attending her mother's funeral when she met a man she didn't know. She thought he was amazing and fell in love with him immediately, but never got his contact information. A few days later, she killed her sister. Why?"}, 
    {"role": "user", "content": "There is a house with four walls facing south. A bear walks by. What color is the bear?"}, 
    {"role": "user", "content": "A man goes into a restaurant, orders soup, takes one sip, and asks to use the phone. After returning from the phone, he finishes the soup and leaves. Why did he use the phone?"}, 
    {"role": "user", "content": "What disappears as soon as you say its name?"}, 
    {"role": "user", "content": "Two people are born at the exact same moment, die at the exact same moment, yet they lived for different amounts of time. How is this possible?"}, 
    {"role": "user", "content": "What five-letter word becomes shorter when you add two letters to it?"}, 
    {"role": "user", "content": "I'm light as a feather, but even the strongest person can't hold me for more than a few minutes. What am I?"}, 
    {"role": "user", "content": "A boy and a doctor go fishing. The boy is the doctor's son, but the doctor is not the boy's father. How is this possible?"}, 
    {"role": "user", "content": "You're running a race and you pass the person in second place. What position are you in now?"}, 
    {"role": "user", "content": "Forward I'm heavy, but backward I'm not. What am I?"}, 
    {"role": "user", "content": "A woman shoots her husband with a gun, then holds him underwater for five minutes. Finally, she hangs him. Five minutes later they enjoy a lovely dinner together. How is this possible?"}, 
    {"role": "user", "content": "I have branches but no fruit, trunk or leaves. What am I?"},

    # Causal Reasoning
    {"role": "user", "content": "If pushing domino A makes domino B fall, and domino B makes domino C fall, but domino C is already down, what happens when you push domino A?"}, 
    {"role": "user", "content": "If all roses in a garden die when they don't get water for a week, and it hasn't rained for two weeks, but the roses are alive, what can you conclude?"}, 
    {"role": "user", "content": "If every time it rains, the streets get wet, and the streets are wet now, can you conclude that it rained?"}, 
    {"role": "user", "content": "If all students who study hard pass the exam, and Jane passed the exam, can you conclude she studied hard?"}, 
    {"role": "user", "content": "If turning the key starts the car, and the key is turned but the car doesn't start, what can you deduce?"}, 
    {"role": "user", "content": "If eating ice cream always makes Tom happy, and Tom is happy now, must he have eaten ice cream?"}, 
    {"role": "user", "content": "If a plant grows taller when given fertilizer, and this plant is taller than last week, can you conclude it was given fertilizer?"}, 
    {"role": "user", "content": "If mixing blue and yellow always makes green, and you have green paint, must it have been made by mixing blue and yellow?"}, 
    {"role": "user", "content": "If all birds can fly, and you see a flying animal, must it be a bird?"}, 
    {"role": "user", "content": "If whenever it snows, schools close, and schools are closed today, can you conclude it snowed?"}, 
    {"role": "user", "content": "If a plant grows better in sunlight, and this plant isn't growing well despite being in sunlight, what can you conclude?"}, 
    {"role": "user", "content": "If all cats like fish, and Whiskers doesn't like fish, what can you deduce?"}, 
    {"role": "user", "content": "If exercise increases heart rate, and your heart rate is elevated, can you conclude you've exercised?"}, 
    {"role": "user", "content": "If all metals conduct electricity, and this material conducts electricity, must it be a metal?"}, 
    {"role": "user", "content": "If reading improves vocabulary, and Sam has an excellent vocabulary, must he be an avid reader?"}, 
    {"role": "user", "content": "If coffee keeps people awake, and Alex is awake at midnight, must they have had coffee?"}, 
    {"role": "user", "content": "If studying guarantees passing, and John failed, what can you conclude?"}, 
    {"role": "user", "content": "If wind makes leaves rustle, and leaves are rustling now, must it be windy?"}, 
    {"role": "user", "content": "If all dolphins are mammals, and this animal is a mammal, must it be a dolphin?"}, 
    {"role": "user", "content": "If sugar makes tea sweet, and this tea is sweet, must it contain sugar?"}, 
    {"role": "user", "content": "If all geniuses are creative, and Bob is creative, must he be a genius?"}, 
    {"role": "user", "content": "If running makes you sweat, and you're sweating, must you have been running?"}, 
    {"role": "user", "content": "If all squares are rectangles, and this shape is a rectangle, must it be a square?"}, 
    {"role": "user", "content": "If fever indicates infection, and you have an infection, must you have a fever?"}, 
    {"role": "user", "content": "If watering plants makes them grow, and these plants are growing, must someone be watering them?"}, 
    {"role": "user", "content": "If all birds lay eggs, and this animal lays eggs, must it be a bird?"}, 
    {"role": "user", "content": "If practice makes perfect, and Mary is perfect at piano, must she have practiced a lot?"}, 
    {"role": "user", "content": "If all A's are B's, and this is not a B, what can you conclude about it being an A?"}, 
    {"role": "user", "content": "If sunlight makes shadows, and there's a shadow here, must the sun be shining?"}, 
    {"role": "user", "content": "If all geniuses are creative, and Bob is creative, must he be a genius?"},
    # New Causal Reasoning problems
    {"role": "user", "content": "If eating peanuts causes allergic reactions in some people, and Lisa had an allergic reaction, can you conclude she ate peanuts?"}, 
    {"role": "user", "content": "If smoking increases the risk of lung cancer, and someone has lung cancer, must they have been a smoker?"}, 
    {"role": "user", "content": "If all successful entrepreneurs work hard, and Tina works hard, must she be a successful entrepreneur?"}, 
    {"role": "user", "content": "If sleep deprivation impairs cognitive function, and Michael's cognitive function is impaired, must he be sleep deprived?"}, 
    {"role": "user", "content": "If all computers made by company X crash frequently, and your computer crashes frequently, must it be made by company X?"}, 
    {"role": "user", "content": "If exposure to sunlight produces vitamin D in humans, and Emma has adequate vitamin D levels, must she get regular sun exposure?"}, 
    {"role": "user", "content": "If water boils at 100°C at sea level, and water is boiling, must the temperature be exactly 100°C?"}, 
    {"role": "user", "content": "If all professional basketball players are tall, and Jordan is tall, must he be a professional basketball player?"}, 
    {"role": "user", "content": "If a drought causes crop failure, and there's a crop failure, can you conclude there was a drought?"}, 
    {"role": "user", "content": "If all Olympic athletes train daily, and Marcus trains daily, must he be an Olympic athlete?"}, 
    {"role": "user", "content": "If regular maintenance prevents car breakdowns, and a car hasn't broken down, must it have received regular maintenance?"}, 
    {"role": "user", "content": "If all fish need water to survive, and this animal needs water to survive, must it be a fish?"}, 
    {"role": "user", "content": "If lightning is always followed by thunder, and you hear thunder, must there have been lightning?"}, 
    {"role": "user", "content": "If all apples from this tree are red, and this fruit is red, must it be an apple from this tree?"}, 
    {"role": "user", "content": "If taking this medication always reduces fever within an hour, and the patient's fever hasn't reduced after an hour, what can you conclude?"}, 
    {"role": "user", "content": "If all blood vessels carrying blood toward the heart are veins, and this blood vessel is carrying blood toward the heart, must it be a vein?"}, 
    {"role": "user", "content": "If calcium deficiency leads to weak bones, and someone has weak bones, must they have a calcium deficiency?"}, 
    {"role": "user", "content": "If regular exercise improves cardiovascular health, and David has excellent cardiovascular health, must he exercise regularly?"}, 
    {"role": "user", "content": "If a gene mutation causes this disease, and a person has this disease, must they have this gene mutation?"}, 
    {"role": "user", "content": "If all mammals have hair, and this animal has hair, must it be a mammal?"},

    # Probabilistic Thinking
    {"role": "user", "content": "In a drawer there are 10 red socks, 10 blue socks, and 10 black socks. If you reach in the dark and grab socks one at a time, how many socks minimum do you need to grab to guarantee a matching pair?"}, 
    {"role": "user", "content": "If you roll two dice, what's the probability of getting a sum greater than 7?"}, 
    {"role": "user", "content": "In a bag of 100 marbles, 70 are blue and 30 are red. If you draw 2 marbles without replacement, what's the probability they're different colors?"}, 
    {"role": "user", "content": "If you flip a coin 3 times, what's the probability of getting at least one heads?"}, 
    {"role": "user", "content": "In a standard deck of 52 cards, what's the probability of drawing two aces in a row without replacement?"}, 
    {"role": "user", "content": "If you have 3 red balls and 2 blue balls in a bag and draw 2 balls without replacement, what's the probability of getting matching colors?"}, 
    {"role": "user", "content": "What's the probability of getting exactly two 6s when rolling three dice?"}, 
    {"role": "user", "content": "If you randomly select 2 numbers from 1-10, what's the probability their sum is even?"}, 
    {"role": "user", "content": "In a room of 30 people, what's the probability that at least two people share the same birthday?"}, 
    {"role": "user", "content": "If you shuffle a deck of cards and deal 5 cards, what's the probability of getting a flush (all same suit)?"}, 
    {"role": "user", "content": "If you have 5 different keys and only one opens the door, what's the probability of opening it on the first try?"}, 
    {"role": "user", "content": "In a class of 20 students, what's the probability that none share a birthday month?"}, 
    {"role": "user", "content": "If you roll three dice, what's the probability of getting three different numbers?"}, 
    {"role": "user", "content": "With 4 red balls and 3 green balls, what's the probability of drawing 2 red balls with replacement?"}, 
    {"role": "user", "content": "What's the probability of drawing a royal flush in poker?"}, 
    {"role": "user", "content": "If you flip a coin until you get heads, what's the probability it takes exactly 3 flips?"}, 
    {"role": "user", "content": "In a bag with 3 black, 4 white, and 5 red marbles, what's the probability of drawing black then white?"}, 
    {"role": "user", "content": "What's the probability of rolling a sum of 7 with two dice?"}, 
    {"role": "user", "content": "If you randomly arrange the letters in 'MATH', what's the probability of spelling 'MATH'?"}, 
    {"role": "user", "content": "In a deck of cards, what's the probability of drawing three hearts in a row with replacement?"}, 
    {"role": "user", "content": "What's the probability of getting exactly one 6 when rolling five dice?"}, 
    {"role": "user", "content": "If you pick 2 numbers from 1-20, what's the probability their difference is 5?"}, 
    {"role": "user", "content": "What's the probability of drawing all face cards when dealing 3 cards?"}, 
    {"role": "user", "content": "If you flip 4 coins, what's the probability of getting exactly 2 heads?"}, 
    {"role": "user", "content": "In a group of 5 people, what's the probability at least 2 share the same birthday?"}, 
    {"role": "user", "content": "What's the probability of drawing 2 kings and 1 queen from a deck in that order?"}, 
    {"role": "user", "content": "If you roll two dice, what's the probability their product is even?"}, 
    {"role": "user", "content": "What's the probability of getting no pairs when dealt 5 cards?"}, 
    {"role": "user", "content": "If you pick 3 random letters, what's the probability of being able to spell 'CAT'?"}, 
    {"role": "user", "content": "What's the probability of rolling four dice and getting all different numbers?"},
    # New Probabilistic Thinking problems
    {"role": "user", "content": "In a lottery where you pick 6 numbers from 1-49, what's the probability of matching at least 3 numbers?"}, 
    {"role": "user", "content": "If 3 people randomly select seats in a 5-seat row, what's the probability they all sit next to each other?"}, 
    {"role": "user", "content": "In a box of 20 chocolates where 5 contain nuts, if you eat 3 chocolates randomly, what's the probability of not eating any with nuts?"}, 
    {"role": "user", "content": "What's the probability of getting a Yahtzee (all five dice showing the same number) in a single roll?"}, 
    {"role": "user", "content": "If 40% of a population has blood type A and 60% has blue eyes, what's the probability someone has both if the traits are independent?"}, 
    {"role": "user", "content": "In a bag with 4 green, 3 yellow, and 5 orange marbles, what's the probability of drawing 3 marbles of different colors without replacement?"}, 
    {"role": "user", "content": "If you shuffle a standard deck of cards and deal the top 13 cards, what's the probability of getting all spades?"}, 
    {"role": "user", "content": "When rolling a pair of dice, what's the probability of getting a sum that's a prime number?"}, 
    {"role": "user", "content": "In a group of 7 people, what's the probability that at least 3 share the same birthday month?"}, 
    {"role": "user", "content": "If you randomly select 5 people from a group where 30% are left-handed, what's the probability exactly 2 are left-handed?"}, 
    {"role": "user", "content": "What's the probability of getting at least one double-six when rolling a pair of dice 24 times?"}, 
    {"role": "user", "content": "If you randomly arrange 8 people in a line, what's the probability that Alice and Bob stand next to each other?"}, 
    {"role": "user", "content": "In a drawer with 6 black socks, 8 white socks, and 4 blue socks, what's the probability of picking 2 socks of the same color without looking?"}, 
    {"role": "user", "content": "If 5 cards are dealt from a standard deck, what's the probability of getting exactly one pair (and no three of a kind)?"}, 
    {"role": "user", "content": "When flipping a fair coin 10 times, what's the probability of getting exactly 5 heads followed by 5 tails?"}, 
    {"role": "user", "content": "If 10 people are randomly assigned to 3 different teams, what's the probability that Alice and Bob are on the same team?"}, 
    {"role": "user", "content": "In a jar with 10 red, 15 green, and 5 blue jelly beans, what's the probability of drawing exactly 2 of each color when picking 6 without replacement?"}, 
    {"role": "user", "content": "What's the probability that a 5-card poker hand contains cards of exactly 3 different suits?"}, 
    {"role": "user", "content": "If 20% of components are defective, what's the probability that exactly 2 are defective in a random sample of 5?"}, 
    {"role": "user", "content": "In a game where you roll 3 dice, what's the probability that the sum of the results is divisible by 3?"},

    # Systems Thinking
    {"role": "user", "content": "In a small town, the birth rate increases but the population decreases. Give three possible explanations for this phenomenon."}, 
    {"role": "user", "content": "How might introducing wolves into an ecosystem affect the population of plants?"}, 
    {"role": "user", "content": "What could be the long-term effects of making all public transportation free in a city?"}, 
    {"role": "user", "content": "How might increasing minimum wage affect small businesses, employment rates, and consumer spending?"}, 
    {"role": "user", "content": "What might be the cascading effects of removing all social media platforms for one month globally?"}, 
    {"role": "user", "content": "How might widespread adoption of electric cars affect oil prices, air quality, and electricity demand?"}, 
    {"role": "user", "content": "What could be the systemic effects of making college education free?"}, 
    {"role": "user", "content": "How might a significant reduction in bee populations affect agriculture and the broader ecosystem?"}, 
    {"role": "user", "content": "What might be the ripple effects of implementing a four-day work week nationwide?"}, 
    {"role": "user", "content": "How might widespread adoption of remote work affect urban development, transportation, and housing markets?"}, 
    {"role": "user", "content": "How would introducing a new predator species affect an ecosystem's food chain?"}, 
    {"role": "user", "content": "What would be the ripple effects of removing all bees from Earth's ecosystem?"}, 
    {"role": "user", "content": "How might increasing minimum wage affect different aspects of the economy?"}, 
    {"role": "user", "content": "What would be the systemic effects of making all public transportation free?"}, 
    {"role": "user", "content": "How would widespread adoption of remote work affect urban development?"}, 
    {"role": "user", "content": "What would be the cascading effects of removing all social media platforms?"}, 
    {"role": "user", "content": "How might implementing a four-day work week affect society and the economy?"}, 
    {"role": "user", "content": "What would be the systemic impact of making all vehicles electric by 2030?"}, 
    {"role": "user", "content": "How would universal basic income affect various aspects of society?"}, 
    {"role": "user", "content": "What would be the ripple effects of making college education free?"}, 
    {"role": "user", "content": "How might eliminating plastic packaging affect global supply chains?"}, 
    {"role": "user", "content": "What would be the systemic effects of implementing global carbon pricing?"}, 
    {"role": "user", "content": "How would universal healthcare affect various aspects of society?"}, 
    {"role": "user", "content": "What would be the cascading effects of switching to 100% renewable energy?"}, 
    {"role": "user", "content": "How might automated vehicles affect transportation and urban planning?"}, 
    {"role": "user", "content": "What would be the systemic impact of vertical farming in major cities?"}, 
    {"role": "user", "content": "How would universal internet access affect global education and commerce?"}, 
    {"role": "user", "content": "What would be the ripple effects of implementing a global currency?"}, 
    {"role": "user", "content": "How might artificial intelligence affect different sectors of the economy?"}, 
    {"role": "user", "content": "What would be the systemic effects of colonizing Mars?"},
    # New Systems Thinking problems
    {"role": "user", "content": "What would be the cascading effects of implementing a worldwide 3-day weekend?"}, 
    {"role": "user", "content": "How might providing universal access to clean drinking water affect global development, health systems, and economic productivity?"}, 
    {"role": "user", "content": "What would be the systemic consequences of reducing global meat consumption by 50%?"}, 
    {"role": "user", "content": "How might implementing a global language affect cultural diversity, international relations, and educational systems?"}, 
    {"role": "user", "content": "What could be the ripple effects of making all software open source?"}, 
    {"role": "user", "content": "How might the disappearance of coral reefs affect marine ecosystems, tourism, and coastal communities?"}, 
    {"role": "user", "content": "What would be the systemic impacts of doubling urban green spaces in major cities worldwide?"}, 
    {"role": "user", "content": "How might implementing a 20-hour workweek affect productivity, mental health, and family structures?"}, 
    {"role": "user", "content": "What could be the cascading effects of eliminating all agricultural pesticides globally?"}, 
    {"role": "user", "content": "How might universal childcare affect workforce participation, early childhood development, and gender equality?"}, 
    {"role": "user", "content": "What would be the ripple effects of a significant asteroid mining industry?"}, 
    {"role": "user", "content": "How might implementing a global carbon tax affect energy production, international trade, and technological innovation?"}, 
    {"role": "user", "content": "What could be the systemic effects of drastically reducing military spending worldwide?"}, 
    {"role": "user", "content": "How might extending human lifespan to 150 years affect social security systems, family structures, and career patterns?"}, 
    {"role": "user", "content": "What would be the cascading impacts of a significant sea level rise on coastal cities, migration patterns, and global economics?"}, 
    {"role": "user", "content": "How might widespread adoption of lab-grown meat affect traditional agriculture, land use, and cultural food practices?"}, 
    {"role": "user", "content": "What could be the ripple effects of implementing a global cap on personal wealth?"}, 
    {"role": "user", "content": "How might universal access to brain-computer interfaces affect education, communication, and cognitive development?"}, 
    {"role": "user", "content": "What would be the systemic impacts of transitioning to a circular economy where all products are designed for reuse?"}, 
    {"role": "user", "content": "How might the reintroduction of keystone species to damaged ecosystems affect biodiversity, land management, and local economies?"},

    # Creative Problem Solving
    {"role": "user", "content": "How can you cut a cake into 8 equal pieces using only 3 straight cuts?"}, 
    {"role": "user", "content": "How can you measure exactly 4 liters using only a 3-liter and a 5-liter container?"}, 
    {"role": "user", "content": "How can you cross a river with a wolf, a goat, and a cabbage if you can only take one at a time and can't leave the wolf alone with the goat or the goat alone with the cabbage?"}, 
    {"role": "user", "content": "Without using a scale, how can you find the one heavier ball among eight identical-looking balls using just two weighings on a balance scale?"}, 
    {"role": "user", "content": "How can you arrange 10 coins in a triangle shape, then move only 3 coins to form 5 perfect triangles?"}, 
    {"role": "user", "content": "How can you divide 11 apples among 3 people equally without cutting any apples?"}, 
    {"role": "user", "content": "How can you measure 45 seconds using two ropes that each take 1 minute to burn completely (and burn non-uniformly)?"}, 
    {"role": "user", "content": "How can you write the number 100 using only four 9s and basic mathematical operations?"}, 
    {"role": "user", "content": "How can you make four triangles using six matchsticks?"}, 
    {"role": "user", "content": "How can you share 1000 dollars among three people so that each person gets 1 dollar more than the person before them?"}, 
    {"role": "user", "content": "How can you keep plants watered while away for a month without technology?"}, 
    {"role": "user", "content": "Design a system to sort recycling without using electricity."}, 
    {"role": "user", "content": "How can you cool a room without air conditioning?"}, 
    {"role": "user", "content": "Create a way to teach multiplication without using numbers."}, 
    {"role": "user", "content": "Design a transportation system for a city without cars."}, 
    {"role": "user", "content": "How can you measure time without a clock or watch?"}, 
    {"role": "user", "content": "Create a communication system without using words or pictures."}, 
    {"role": "user", "content": "Design a way to keep food fresh without refrigeration."}, 
    {"role": "user", "content": "How can you clean clothes without using water?"}, 
    {"role": "user", "content": "Create a system for organizing books without using alphabetical order."}, 
    {"role": "user", "content": "How can you make music without traditional instruments?"}, 
    {"role": "user", "content": "Design a way to share resources in a community without using money."}, 
    {"role": "user", "content": "How can you teach a language without using translations?"}, 
    {"role": "user", "content": "Create a navigation system without using maps or GPS."}, 
    {"role": "user", "content": "How can you build shelter using only natural materials?"}, 
    {"role": "user", "content": "Design a system to reduce food waste in restaurants."}, 
    {"role": "user", "content": "How can you create art without using traditional art supplies?"}, 
    {"role": "user", "content": "Create a security system without using technology."}, 
    {"role": "user", "content": "How can you teach children about sustainability through play?"}, 
    {"role": "user", "content": "Design a system for urban farming in limited space."},
    # New Creative Problem Solving problems
    {"role": "user", "content": "How can you determine which of two identical-looking eggs is hard-boiled and which is raw without cracking them open?"}, 
    {"role": "user", "content": "Design a way to move a 500-pound stone block up a hill without modern machinery."}, 
    {"role": "user", "content": "How can you cook a meal without using any heat source?"}, 
    {"role": "user", "content": "Create a method to memorize a deck of 52 cards in order within 5 minutes."}, 
    {"role": "user", "content": "How can you send a message across a town without using technology or leaving your location?"}, 
    {"role": "user", "content": "Design a water collection system for a desert region using only local materials."}, 
    {"role": "user", "content": "How can you determine north without a compass or celestial references?"}, 
    {"role": "user", "content": "Create a soundproofing solution for a room using only household items."}, 
    {"role": "user", "content": "How can you solve the problem of plastic waste in oceans using biomimicry (solutions inspired by nature)?"}, 
    {"role": "user", "content": "Design a communication system for people who have lost both hearing and sight."}, 
    {"role": "user", "content": "How can you create a lighting system for a home during extended power outages without batteries or generators?"}, 
    {"role": "user", "content": "Create a method to measure the height of a tall building from the ground without specialized equipment."}, 
    {"role": "user", "content": "How can you divide a piece of land fairly among three people when each values different parts differently?"}, 
    {"role": "user", "content": "Design a system to teach critical thinking skills to children under 10."}, 
    {"role": "user", "content": "How can you protect crops from pests without using chemicals or electricity?"}, 
    {"role": "user", "content": "Create a barter system for a community of people with highly diverse skills and needs."}, 
    {"role": "user", "content": "How can you detect if food has spoiled without tasting, smelling, or using technology?"}, 
    {"role": "user", "content": "Design a reliable voting system that ensures anonymity without using computers."}, 
    {"role": "user", "content": "How can you predict weather patterns using only observations of your local environment?"}, 
    {"role": "user", "content": "Create a system for reaching group consensus that gives everyone equal input regardless of personality type."},

    # Scientific Reasoning
    {"role": "user", "content": "If a tree falls in a forest and no one is around to hear it, does it create sound waves? Explain your reasoning."}, 
    {"role": "user", "content": "Why does a hot cup of coffee cool down but a cold glass of water warms up when both are left in room temperature?"}, 
    {"role": "user", "content": "Why do wheels appear to spin backward in videos sometimes?"}, 
    {"role": "user", "content": "Why does a metal spoon feel colder than a wooden spoon when both are at the same temperature?"}, 
    {"role": "user", "content": "Why does salt make ice melt but also help make ice cream freeze?"}, 
    {"role": "user", "content": "Why do stars appear to twinkle but planets don't?"}, 
    {"role": "user", "content": "Why does a balloon inflate when heated and deflate when cooled?"}, 
    {"role": "user", "content": "Why does a spinning ice skater speed up when pulling in their arms?"}, 
    {"role": "user", "content": "Why does a straw appear to bend when placed in a glass of water?"}, 
    {"role": "user", "content": "Why does honey flow more slowly than water?"}, 
    {"role": "user", "content": "Why do some materials conduct electricity better than others?"}, 
    {"role": "user", "content": "How does the moon affect ocean tides?"}, 
    {"role": "user", "content": "Why do some substances dissolve in water while others don't?"}, 
    {"role": "user", "content": "How does pressure affect the boiling point of water?"}, 
    {"role": "user", "content": "Why do leaves change color in autumn?"}, 
    {"role": "user", "content": "How does temperature affect the speed of chemical reactions?"}, 
    {"role": "user", "content": "Why do some objects float while others sink?"}, 
    {"role": "user", "content": "How does light behave differently in water versus air?"}, 
    {"role": "user", "content": "Why do some materials reflect light while others absorb it?"}, 
    {"role": "user", "content": "How does sound travel through different mediums?"}, 
    {"role": "user", "content": "Why do some materials insulate better than others?"}, 
    {"role": "user", "content": "How does air pressure affect weather patterns?"}, 
    {"role": "user", "content": "Why do some materials become magnetic when others don't?"}, 
    {"role": "user", "content": "How does gravity affect objects of different masses?"}, 
    {"role": "user", "content": "Why do some substances change state at different temperatures?"}, 
    {"role": "user", "content": "How does cellular respiration produce energy?"}, 
    {"role": "user", "content": "Why do some chemical reactions release heat while others absorb it?"}, 
    {"role": "user", "content": "How does natural selection lead to evolution?"}, 
    {"role": "user", "content": "Why do some materials decompose faster than others?"}, 
    {"role": "user", "content": "How does the greenhouse effect influence Earth's temperature?"},
    # New Scientific Reasoning problems
    {"role": "user", "content": "Why does a gyroscope maintain its orientation when spinning?"}, 
    {"role": "user", "content": "How do birds navigate during migration without maps or GPS?"}, 
    {"role": "user", "content": "Why does microwaved food cool from the outside in, while oven-heated food cools from the inside out?"}, 
    {"role": "user", "content": "How do vaccines create immunity without causing the disease they protect against?"}, 
    {"role": "user", "content": "Why does a cup of hot tea cool faster in a cold room than a lukewarm room, but not proportionally to the temperature difference?"}, 
    {"role": "user", "content": "How do noise-cancelling headphones eliminate background noise while allowing you to hear music?"}, 
    {"role": "user", "content": "Why does stretching a rubber band make it feel warmer, while allowing it to contract makes it feel cooler?"}, 
    {"role": "user", "content": "How do plants sense and grow toward light?"}, 
    {"role": "user", "content": "Why does ice float on water when most solids sink in their liquid form?"}, 
    {"role": "user", "content": "How do chameleons change their color to match their surroundings?"}, 
    {"role": "user", "content": "Why does time seem to slow down during intense experiences or emergencies?"}, 
    {"role": "user", "content": "How do antibiotics kill bacteria without harming human cells?"}, 
    {"role": "user", "content": "Why does a refrigerator's back feel warm when its inside is cold?"}, 
    {"role": "user", "content": "How do fireflies produce light without generating heat?"}, 
    {"role": "user", "content": "Why does a pendulum swing with the same period regardless of the weight at the end?"}, 
    {"role": "user", "content": "How do dolphins use echolocation to navigate and hunt in murky waters?"}, 
    {"role": "user", "content": "Why does stretching a rubber band make it feel warmer, while allowing it to contract makes it feel cooler?"}, 
    {"role": "user", "content": "How do plants communicate with each other when under attack by pests?"}, 
    {"role": "user", "content": "Why does adding salt to water increase the time it takes to boil, but decrease the time needed to cook food?"}, 
    {"role": "user", "content": "How do migratory animals know when it's time to begin their seasonal journeys?"}
]

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

    # Normalize date formats (e.g., "October 30" vs "October\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized


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

    Args:
        solution_text: The full solution text

    Returns:
        List of chunks
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
