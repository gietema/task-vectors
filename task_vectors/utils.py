import random

import pandas as pd

from task_vectors.constants import SEPARATOR


def save_results(results_per_layer: dict, filename: str):
    df = pd.DataFrame(results_per_layer.items(), columns=["layer", "accuracy"])
    df.to_csv(filename, index=False)


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


def infer(model, tokenizer, prompt, device):
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[-1][-1], skip_special_tokens=True)
