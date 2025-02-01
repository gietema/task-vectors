from abc import abstractmethod, ABC
from typing import Literal

import click
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoProcessor

from task_vectors.data import Sample

LlamaModel = Literal["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-11B", "Llama-3.2-90B"]
QwenModel = Literal["Qwen2-VL-2B", "Qwen2-VL-7B"]

ModelName = Literal[LlamaModel, QwenModel]


class ModelProcessor(ABC):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        device: str,
        processor: AutoProcessor | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.processor: AutoProcessor | None = processor

    @classmethod
    def from_name(cls, model_name: str, device):
        pass

    @abstractmethod
    def __call__(
        self, item: Sample, context: list[Sample] | None = None, *args, **kwargs
    ):
        pass

    @abstractmethod
    def preprocess(self, item: Sample, context: list[Sample] | None = None):
        pass


class VisionModelProcessor(ModelProcessor):
    @abstractmethod
    def __call__(
        self,
        item: Sample,
        context: list[Sample] | None = None,
        use_img: bool = False,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def preprocess(
        self, item: Sample, context: list[Sample] | None = None, use_img: bool = False
    ):
        pass


def model_factory(model_name: LlamaModel | QwenModel, device: str):
    if model_name.startswith("Llama"):
        from task_vectors.models.llama import LlamaModelProcessor

        return LlamaModelProcessor.from_name(model_name, device)  # type: ignore
    if model_name.startswith("Qwen"):
        from task_vectors.models.qwen import QwenModelProcessor

        return QwenModelProcessor.from_name(model_name, device)  # type: ignore
    raise ValueError(f"Invalid model name: {model_name}")


def validate_model_size(ctx, param, value):
    valid_sizes = {"Llama-1B", "Llama-3B", "Llama-11B", "Llama-90B"}
    if value not in valid_sizes:
        raise click.BadParameter(
            f"Invalid model size: {value}. Must be one of {', '.join(valid_sizes)}."
        )
    return value
