import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from transformers import StoppingCriteria, StoppingCriteriaList


SYSTEM_PROMPT = (
    "You convert natural language into structured data.\n"
    "STRICT RULES:\n"
    "- Output ONLY valid {format}\n"
    "- No explanations\n"
    "- No extra text\n"
)

STOP_STRINGS = {
    "json": ["}", "<|user|>", "<|assistant|>"],
    "xml": ["</record>", "<|user|>", "<|assistant|>"],
    "yaml": ["\n<|", "<|user|>", "<|assistant|>"],
    "csv": ["\n<|", "<|user|>", "<|assistant|>"],
    "toml": ["\n<|", "<|user|>", "<|assistant|>"],
}


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids  # list of lists

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if len(input_ids[0]) >= len(stop_ids):
                if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


def build_stopping_criteria(tokenizer, format_name):
    stop_strings = STOP_STRINGS.get(format_name, [])

    stop_token_ids = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) > 0:
            stop_token_ids.append(ids)

    if not stop_token_ids:
        return None

    return StoppingCriteriaList([StopOnTokens(stop_token_ids)])

def apply_stop(text, format_name):
    stops = STOP_STRINGS.get(format_name, [])
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            return text[: idx + len(s)]
    return text


def generate(model, tokenizer, prompt: str, format_name: str) -> str:
    system_prompt = SYSTEM_PROMPT.format(format=format_name.upper())

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    stopping_criteria = build_stopping_criteria(tokenizer, format_name)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # 🔥 ADD THIS
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    response = apply_stop(response, format_name)

    for bad in ["user", "assistant"]:
        idx = response.find(bad)
        if idx != -1:
            response = response[:idx]

    # optional safety trim (keep it as backup)
    for s in STOP_STRINGS.get(format_name, []):
        idx = response.find(s)
        if idx != -1:
            response = response[: idx + len(s)]
            break

    del inputs, output_ids, generated
    torch.cuda.empty_cache()

    return response.strip()