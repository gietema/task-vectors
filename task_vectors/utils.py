import random
from pathlib import Path

import pandas as pd
import torch

from task_vectors.constants import SEPARATOR


def save_results(results_per_layer: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_per_layer.items(), columns=["layer", "accuracy"])
    df.to_csv(filepath, index=False)


def get_icl_prompt(data: dict, nb_shots: int) -> tuple[list[str], dict]:
    # +1 for including the dummy query
    sampled_items = random.sample(list(data.items()), nb_shots + 1)
    prompts = [
        f"{task_input}{SEPARATOR}{target_output}"
        for task_input, target_output in sampled_items[:-1]
    ]
    # Last item is the dummy query, we just add the task input plus the separator
    prompts += [sampled_items[-1][0] + SEPARATOR]
    return prompts, dict(sampled_items)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    return device
