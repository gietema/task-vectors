from typing import Literal

import click
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LlamaModelSize = Literal["1B", "3B", "11B", "90B"]


def validate_model_size(ctx, param, value):
    valid_sizes = {"1B", "3B", "11B", "90B"}
    if value not in valid_sizes:
        raise click.BadParameter(
            f"Invalid model size: {value}. Must be one of {', '.join(valid_sizes)}."
        )
    return value


def load_model(model_size: LlamaModelSize, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        f"meta-llama/Llama-3.2-{model_size}", paddin_size="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        f"meta-llama/Llama-3.2-{model_size}",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.top_p = 1.0
    model.generation_config.temperature = 1.0
    return model, tokenizer
