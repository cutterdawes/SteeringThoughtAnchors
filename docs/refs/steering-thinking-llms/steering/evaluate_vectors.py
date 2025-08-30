# %%
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
from utils import chat, steering_config
import re
import numpy as np
from messages import eval_messages
from tqdm import tqdm
import gc
import os
import utils
from utils import steering_config

os.system('')  # Enable ANSI support on Windows

# %% Evaluation examples - 3 from each category
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Get activations and response
data_idx = 2

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_steering(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=250,
    label="none", 
    feature_vectors=None,
    steering_config=steering_config[model_name],
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")


# %%
for t in ["positive", "negative"]:

    input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
    output_ids = utils.custom_generate_steering(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=250,
        label="example-testing",
        feature_vectors=feature_vectors,
        steering_config=steering_config[model_name],
        steer_positive=True if t == "positive" else False,
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(response)
    print("\n================\n")

# %%
