import json
from typing import Iterator

from task_vectors.constants import DATA_DIR, SEPARATOR


def load_data(task_name: str, tokenizer) -> dict:
    with open(DATA_DIR / f"{task_name}.json") as f:
        data = json.load(f)
    return preprocess_data(data, tokenizer)


def preprocess_data(data: dict, tokenizer) -> dict:
    # Filter multiple-token outputs to simplify evaluation
    data_filtered = {}
    for input_item, target_output in filter(
        lambda x: (x[0].strip(), x[1].strip()), data.items()
    ):
        # FIXME: will not work for all tokenizers
        if not len(tokenizer.tokenize(f"!{target_output}")) != 2:
            continue
        data_filtered[input_item] = target_output
    return data_filtered


def data_iterator(data: dict, exclude_keys: set) -> Iterator[tuple[str, str]]:
    for input_item, target_output in data.items():
        if input_item in exclude_keys:
            continue
        yield f"{input_item}{SEPARATOR}", target_output
