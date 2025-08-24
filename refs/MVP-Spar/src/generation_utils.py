import torch


@torch.no_grad()
def top_p_decoding(probs, top_p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    next_id = torch.multinomial(sorted_probs, num_samples=1)
    next_id = sorted_idx.gather(-1, next_id)
    return next_id

def print_decoded(tokens, tokenizer, title="Output"):
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f"\n=== {title} ===")
    print(text)
    print("-----------------------------------------")

@torch.no_grad()
def generate_from_prompt(model, input_ids,
                         steps=64, temperature=1.0, top_p=1.0, device='cuda'):
    input_ids = input_ids.to(device)
    past = None
    generated_ids = input_ids.clone()
    for step in range(steps):
        if past is None:
            out = model(input_ids=generated_ids, use_cache=True)
        else:
            out = model(input_ids=generated_ids[:, -1:], use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            next_id = top_p_decoding(probs, top_p)
        else:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_id], dim=-1)
    return generated_ids

# Problem 1: amplifying new behavior


@torch.no_grad()
def generate_guided_two_prompts(model_A, model_B, input_ids_A, input_ids_B,
                                alpha=0.8, steps=64, temperature=1.0, top_p=0.9):
    ids_A, ids_B = input_ids_A, input_ids_B
    past_A, past_B = None, None
    for step in range(steps):
        if past_A is None:
            out_A = model_A(input_ids=ids_A, use_cache=True)
            out_B = model_B(input_ids=ids_B, use_cache=True)
        else:
            out_A = model_A(input_ids=ids_A[:, -1:], use_cache=True, past_key_values=past_A)
            out_B = model_B(input_ids=ids_B[:, -1:], use_cache=True, past_key_values=past_B)
        past_A, past_B = out_A.past_key_values, out_B.past_key_values
        logits_A = out_A.logits[:, -1, :]
        logits_B = out_B.logits[:, -1, :]
        steered = (1 + alpha) * logits_B - alpha * logits_A
        if temperature != 1.0:
            steered = steered / temperature
        steered = steered - steered.max(dim=-1, keepdim=True).values
        probs = torch.softmax(steered, dim=-1)
        if top_p < 1.0:
            next_id = top_p_decoding(probs, top_p)
        else:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        ids_B = torch.cat([ids_B, next_id], dim=-1)
    return ids_B[:, input_ids_B.shape[1]:]

# Problem 1: amplifying new behavior, sentence level

@torch.no_grad()
def amplify_sentence_difference(model, input_ids_P, input_ids_Q,
                                alpha=0.8, steps=64, temperature=1.0, top_p=0.9):
    ids_P, ids_Q = input_ids_P, input_ids_Q
    past_P, past_Q = None, None
    for step in range(steps):
        if past_P is None:
            out_P = model(input_ids=ids_P, use_cache=True)
            out_Q = model(input_ids=ids_Q, use_cache=True)
        else:
            out_P = model(input_ids=ids_P[:, -1:], use_cache=True, past_key_values=past_P)
            out_Q = model(input_ids=ids_Q[:, -1:], use_cache=True, past_key_values=past_Q)
        
        past_P, past_Q = out_P.past_key_values, out_Q.past_key_values
        # Extracting logits
        logits_P = out_P.logits[:, -1, :]
        logits_Q = out_Q.logits[:, -1, :]
        steered = (1 + alpha) * logits_Q - alpha * logits_P # Steering logits
        
        if temperature != 1.0:
            steered = steered / temperature
        steered = steered - steered.max(dim=-1, keepdim=True).values
        probs = torch.softmax(steered, dim=-1)
        if top_p < 1.0:
            next_id = top_p_decoding(probs, top_p)
        else:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        ids_Q = torch.cat([ids_Q, next_id], dim=-1)
    return ids_Q[:, input_ids_Q.shape[1]:]