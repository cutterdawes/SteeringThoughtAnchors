import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
import re
import argparse

# Add the steering-thinking-llms directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (
    load_model_and_vectors,
    extract_thinking_process_and_answer,
    load_math_problems,
)

def generate_data(
    model_name: str,
    output_dir: str,
    num_prompts_to_test: int = 1,
    math_split: str = "test",
    math_count: int | None = None,
    math_level: str | None = None,
    math_type: str | None = None,
    save_gt: bool = True,
):
    """
    Generates (prompt, CoT, answer, activations) tuples for a given model.
    """
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading model and tokenizer for {model_name}...")
    model, tokenizer, _ = load_model_and_vectors(model_name=model_name, compute_features=False, device=device) # Pass the determined device
    print("Model and tokenizer loaded.")

    all_data = []
    gt_map = {}

    # Build prompt source exclusively from MATH dataset
    problems = load_math_problems(
        problem_type=math_type,
        level=math_level,
        num_problems=math_count,
        split=math_split,
    )
    if not problems:
        raise RuntimeError("Failed to load MATH problems. Ensure network access or provide a local cache.")
    qa_iterable = [
        {"role": "user", "content": p[1]["problem"], "_gt": p[1].get("gt_answer", ""), "_gt_solution": p[1].get("gt_solution", "")}
        for p in problems
    ]
    total_n = len(qa_iterable)

    for i, pair in enumerate(qa_iterable):
        if i >= num_prompts_to_test: # Limit for testing
            break

        original_question = pair["content"]
        gt_answer = pair.get("_gt")
        gt_solution = pair.get("_gt_solution")
        
        # Apply chat template to get the formatted prompt for the model
        message_for_template = {"role": "user", "content": f"""Solve the following problem step by step. You MUST put your final answer in \\boxed{{}}.

Problem: {original_question}

Think carefully and show your reasoning. At the end, provide the final answer enclosed in \\boxed{{}}."""}
        
        inputs_tensor = tokenizer.apply_chat_template([message_for_template], add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        prompt_string_for_model = tokenizer.decode(inputs_tensor[0], skip_special_tokens=True)
        # Calculate prompt_len by excluding the "<think>\n" part that is added by add_generation_prompt=True
        # This ensures that extract_thinking_process_and_answer can correctly find the <think> tag.
        prompt_len = len(prompt_string_for_model) - len("<think>\n")

        print(f"Processing prompt {i+1}/{total_n}: {original_question[:50]}...")
        # adjust length reference for math iterable
        

        # Generate response (CoT + answer) using the underlying transformers model's generate method
        outputs = model.generate(
            input_ids=inputs_tensor,
            attention_mask=(inputs_tensor != tokenizer.pad_token_id).long(), # Manually create attention mask
            max_new_tokens=1000,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n--- Raw Response for Prompt {i+1} ---")
        print(response)
        print(f"--- End Raw Response ---")

        # Extract CoT and answer (favor \boxed{...} extraction for the answer)
        cot, answer = extract_thinking_process_and_answer(response, prompt_len)

        # If no boxed answer was found, attempt a forced-answer completion pass
        if not answer:
            forced_prefix = (
                f"Solve the following problem step by step. You MUST put your final answer in \\boxed{{}}.\n\n"
                f"Problem: {original_question}\n\n"
                f"Solution:\n<think>\n{cot}\n</think>\n\nTherefore, the final answer is \\boxed{{"
            )
            forced_inputs = tokenizer(forced_prefix, return_tensors="pt").to(model.device)
            forced_outputs = model.generate(
                input_ids=forced_inputs["input_ids"],
                attention_mask=(forced_inputs["input_ids"] != tokenizer.pad_token_id).long(),
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            forced_text = tokenizer.decode(forced_outputs[0], skip_special_tokens=True)
            # Attempt boxed extraction from the forced output
            _, forced_answer = extract_thinking_process_and_answer(forced_text, prompt_len=0)
            if forced_answer:
                answer = forced_answer

        record = {
            "prompt": original_question,
            "model_input_prompt": prompt_string_for_model,
            "raw_response": response,
            "cot": cot,
            "answer": answer,
        }
        record["gt_answer"] = gt_answer or ""
        record["gt_solution"] = gt_solution or ""
        if save_gt and gt_answer:
            gt_map[original_question] = gt_answer
        all_data.append(record)
        
        if i % 10 == 0:
            print(f"Generated data for {i+1} prompts.")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generated_data_{model_name.replace('/', '-')}.json") 
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Data generation complete. Saved to {output_file}")
    if save_gt:
        gt_file = os.path.join(output_dir, "ground_truth_math.json")
        with open(gt_file, 'w') as f:
            json.dump(gt_map, f, indent=2)
        print(f"Saved ground-truth answers to {gt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset of (prompt, CoT, answer) tuples from the MATH dataset. "
            "Also writes a prompt→ground‑truth mapping to ground_truth_math.json."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help=(
            "Hugging Face model id to use for generation (thinking model recommended). "
            "Examples: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_data",
        help=(
            "Directory to write outputs. Saves generated_data_{model}.json and "
            "ground_truth_math.json here."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help=(
            "Number of problems to generate in this run (upper bound over the loaded MATH problems)."
        ),
    )
    parser.add_argument(
        "--math_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split from MATH to sample problems from (train or test).",
    )
    parser.add_argument(
        "--math_count",
        type=int,
        default=None,
        help=(
            "How many MATH problems to fetch before generation (random sample). "
            "If omitted, uses all available in the chosen split."
        ),
    )
    parser.add_argument(
        "--math_level",
        type=str,
        default=None,
        help=(
            "Optional difficulty filter for MATH (e.g., level1…level5). If set, only problems "
            "with this level are considered."
        ),
    )
    parser.add_argument(
        "--math_type",
        type=str,
        default=None,
        help=(
            "Optional topic/category filter for MATH (e.g., algebra, number_theory). If set, only "
            "problems with this type are considered."
        ),
    )
    args = parser.parse_args()

    generate_data(
        model_name=args.model,
        output_dir=args.output,
        num_prompts_to_test=args.count,
        math_split=args.math_split,
        math_count=args.math_count,
        math_level=args.math_level,
        math_type=args.math_type,
    )
