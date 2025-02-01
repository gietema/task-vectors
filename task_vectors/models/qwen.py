from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2TokenizerFast,
)

from task_vectors.data import Sample
from task_vectors.models.qwen_utils import process_vision_info
from task_vectors.models.utils import VisionModelProcessor, QwenModel


import random


def create_context(
    data: list[Sample], nb_demonstrations: int = 5, use_img: bool = False
) -> list[dict[str, str]]:
    messages = []
    demonstrations = random.sample(data, nb_demonstrations)
    for sample in demonstrations:
        if use_img:
            content = {"type": "image", "image": str(sample.img)}
        else:
            content = {"type": "text", "text": sample.input_text}
        prompt = [
            {"role": "user", "content": [content]},
            {"role": "assistant", "content": [{"type": "text", "text": sample.target}]},
        ]
        messages.extend(prompt)
    return messages


def create_prompt(
    sample: Sample, context: list[Sample] | None = None, use_img: bool = False
):
    messages = [] if not context else create_context(context, use_img=use_img)
    if use_img:
        content = {"type": "image", "image": str(sample.img)}
    else:
        content = {"type": "text", "text": sample.input_text}
    messages.extend([{"role": "user", "content": [content]}])
    return messages


class QwenModelProcessor(VisionModelProcessor):
    @classmethod
    def from_name(cls, model_name: QwenModel, device: str):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            f"Qwen/{model_name}", torch_dtype="auto", device_map="auto"
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(f"Qwen/{model_name}")
        # chat template of base processor does not work, so we need to load the Instruct processor
        processor: AutoProcessor = AutoProcessor.from_pretrained(
            f"Qwen/{model_name}-Instruct"
        )
        return cls(model, tokenizer, device=device, processor=processor)

    def preprocess(
        self, sample: Sample, context: list[Sample] | None = None, use_img: bool = False
    ):
        messages = create_prompt(sample, context, use_img)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        return inputs

    def __call__(
        self, sample: Sample, context: list[Sample] | None = None, use_img: bool = False, *args,
        **kwargs,
    ):
        inputs = self.preprocess(sample, context, use_img)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
