import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import device

def load_model_and_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return model, tok