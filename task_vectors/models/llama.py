import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_vectors.constants import SEPARATOR
from task_vectors.data import Sample
from task_vectors.models.utils import ModelProcessor, LlamaModel


def _format_sample(sample: Sample) -> str:
    return f"{sample.input_text}{SEPARATOR}{sample.target}"


class LlamaModelProcessor(ModelProcessor):
    def __call__(
        self, item: Sample, context: list[Sample] | None = None, *args, **kwargs
    ):
        prompt = self.preprocess(item, context)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_token_type_ids=False
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(output_ids[-1][-1], skip_special_tokens=True)

    def preprocess(self, item: Sample, context: list[Sample] | None = None) -> str:
        prompt = ""
        if context is not None:
            prompt = "\n".join(
                _format_sample(sample) for sample in context
            ) + "\n"
        prompt += f"{item.input_text}{SEPARATOR}"
        return prompt

    @classmethod
    def from_name(cls, model_name: LlamaModel, device: str):
        model = AutoModelForCausalLM.from_pretrained(
            f"meta-llama/{model_name}",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            f"meta-llama/{model_name}", padding_size="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.top_p = 1.0
        model.generation_config.temperature = 1.0
        return cls(model, tokenizer, device=device)
