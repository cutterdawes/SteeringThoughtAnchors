import torch
from src.config import device, MODELS, prompt_A, prompt_B, cot_A, cot_B
from src.models import load_model_and_tokenizer
from src.generation_utils import print_decoded, generate_from_prompt, generate_guided_two_prompts, amplify_sentence_difference

# Load models and tokenizers
model_A, tok_A = load_model_and_tokenizer(MODELS["A"])
model_B, tok_B = load_model_and_tokenizer(MODELS["B"])
model = model_B

# Tokenize prompts
input_ids_A = tok_A(prompt_A, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)
input_ids_B = tok_B(prompt_B, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)

input_ids_P = tok_B(cot_A, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)
input_ids_Q = tok_B(cot_B, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)

# Problem 1: amplifying new behavior

baseline_new = generate_from_prompt(model_B, input_ids_B,
                                    steps=256, temperature=0.7, top_p=0.7)
print_decoded(baseline_new, tok_B, "Baseline B")

for alpha in [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.5, 2.0]:
    new_tokens = generate_guided_two_prompts(model_A,model_B, input_ids_A, input_ids_B,
                                             alpha=alpha, steps=256, temperature=0.7, top_p=0.7)
    print_decoded(new_tokens, tok_B, f"Amplified difference tokens (alpha={alpha})")


# Problem 1: amplifying new behavior, sentence level

baseline_new = generate_from_prompt(model_B, input_ids_Q,
                                    steps=256, temperature=0.7, top_p=0.7)
print_decoded(baseline_new, tok_B, "Baseline B")

for alpha in [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.5, 2.0]:
    new_tokens = amplify_sentence_difference(model, input_ids_P, input_ids_Q,
                                             alpha=alpha, steps=256, temperature=0.7, top_p=0.7)
    print_decoded(new_tokens, tok_B, f"Amplified difference tokens (alpha={alpha})")



