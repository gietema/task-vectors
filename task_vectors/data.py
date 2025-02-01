import json
from dataclasses import dataclass
from pathlib import Path

from transformers import PreTrainedTokenizer

from task_vectors.constants import TEXT_DATA_DIR, VISION_DATA_DIR, IMAGE_DIR


@dataclass
class Sample:
    input_text: str
    target: str
    img: str | None = None


def task_factory(task_name: str, tokenizer: PreTrainedTokenizer) -> list[Sample]:
    if task_name.startswith("text"):
        return load_text_data(task_name.replace("text/", ""), tokenizer)
    if task_name.startswith("vision"):
        return load_vision_data(task_name.replace("vision_", ""))
    raise ValueError(f"Task {task_name} not found.")


def load_text_data(task_name: str, tokenizer) -> list[Sample]:
    with open(TEXT_DATA_DIR / f"{task_name}.json") as f:
        data = json.load(f)
    return preprocess_data(data, tokenizer)


def preprocess_data(data: dict, tokenizer) -> list[Sample]:
    # Filter multiple-token outputs to simplify evaluation
    data_filtered = []
    for input_item, target_output in filter(
        lambda x: (x[0].strip(), x[1].strip()), data.items()
    ):
        # FIXME: will not work for all tokenizers
        if not len(tokenizer.tokenize(f"!{target_output}")) != 2:
            continue
        data_filtered.append(Sample(input_item, target_output))
    return data_filtered


def load_vision_data(task_name: str, img_dir: Path = IMAGE_DIR) -> list[Sample]:
    data = []
    for split in ["test", "val"]:
        with open(VISION_DATA_DIR / f"{task_name}_{split}.json") as f:
            raw_data = json.load(f)
            data_processed = [
                Sample(
                    item["input"],
                    item["output"],
                    str(img_dir / Path(item["image"]).name),
                )
                for item in raw_data
            ]
            data.extend(data_processed)
    return data
