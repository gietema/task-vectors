import logging
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import click
import torch
from tqdm import tqdm

from task_vectors.data import Sample, task_factory

from task_vectors.models.utils import model_factory, ModelProcessor, ModelName
from task_vectors.utils import save_results, get_device


def get_task_accuracy(
    model: ModelProcessor,
    task_data: list[Sample],
    use_context: bool,
    context: list[Sample] | None = None,
    use_img: bool = False,
):
    exclude_input_texts = set([sample.input_text for sample in context]) if context else set()
    total_count = match_count = 0
    for sample in tqdm(
        filter(lambda x: x.input_text not in exclude_input_texts, task_data),
        total=len(task_data) - len(exclude_input_texts),
    ):
        answer = model(sample, context=context if use_context else None, use_img=use_img)
        match = sample.target.startswith(answer) or answer.startswith(sample.target)
        match_count += match
        total_count += 1
    return round((match_count / total_count) * 100, 4)


@click.command()
@click.option(
    "--model-size",
    type=click.Choice(["Llama-3.2-1B", "Llama-3.2-3B", "Qwen2-VL-2B", "Qwen2-VL-7B"]),
    help="Model size must be one of: Llama-3.2-1B, Llama-3.2-3B, Qwen2-VL-2B, Qwen2-VL-7B",
)
@click.option("--task", type=click.STRING)
@click.option("--nb-demonstrations", type=click.INT, required=False, default=5)
@click.option("--nb-runs", type=click.INT, required=False, default=1)
@click.option("--use-img", is_flag=True, default=False)
def main(model_size: ModelName, task: str, nb_demonstrations: int = 5, nb_runs: int = 1, use_img: bool = False):
    device = get_device()

    model = model_factory(model_size, device)
    task_data = task_factory(task, model.tokenizer)

    def hook_extract(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor],
        outputs: tuple[torch.Tensor],
    ):
        task_vectors.append(outputs[0][:, -1, :].detach().cpu())

    def hook_inject(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor],
        outputs: tuple[torch.Tensor],
        task_vectors: torch.Tensor,
    ):
        outputs[0][:, -1, :] = task_vectors.to(outputs[0].device)
        return outputs

    results_per_layer = defaultdict(list)
    output_filepath = f"results/{model_size}_{task}_{nb_demonstrations}_demos_{nb_runs}_runs.csv"
    if Path(output_filepath).exists():
        logging.info(f"Results already exist for {model_size}, {task}, {nb_demonstrations}, {nb_runs}. Exiting.")
        return

    for _ in tqdm(range(nb_runs), desc=f"Runs for {task}"):
        # Use different in-context data for each run
        # create context with additional dummy query
        context_data: list[Sample] = random.sample(task_data, nb_demonstrations + 1)
        # get baseline standard ICL accuracy
        acc_icl = get_task_accuracy(
            model=model,
            task_data=task_data,
            context=context_data[:-1],  # exclude the dummy query
            use_img=use_img,
            use_context=True,
        )
        results_per_layer["icl"].append(acc_icl)
        tqdm.write(f"ICL Accuracy: {acc_icl}%")

        # get accuracy per layer with task vectors
        for layer_index, layer in tqdm(
            enumerate(model.model.model.layers),
            desc="Layers",
            total=len(model.model.model.layers),
        ):
            task_vectors = []  # Reset task vectors for each layer
            extract_hook = layer.register_forward_hook(hook_extract)
            model(context_data[-1], context=context_data[:-1], use_img=use_img)
            extract_hook.remove()
            inject_hook = layer.register_forward_hook(
                partial(hook_inject, task_vectors=task_vectors[0])
            )
            acc = get_task_accuracy(
                model=model,
                task_data=task_data,
                context=context_data,
                use_context=False,
            )
            tqdm.write(f"Layer {layer_index} Accuracy: {acc}%")
            results_per_layer[layer_index].append(acc)
            extract_hook.remove()
            inject_hook.remove()

    if not Path(output_filepath).exists():
        save_results(
            results_per_layer,
            filepath=output_filepath,
        )


if __name__ == "__main__":
    main()
