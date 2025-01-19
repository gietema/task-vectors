from collections import defaultdict
from functools import partial
import click
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_vectors.data import load_data, data_iterator
from task_vectors.model import load_model, validate_model_size, LlamaModelSize
from task_vectors.utils import save_results, get_icl_prompt, infer


def get_task_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_data: dict,
    exclude_keys: set[str],
    device: str,
    context: str | None = None,
):
    total_count = match_count = 0
    for input_item, target_output in tqdm(
        data_iterator(task_data, exclude_keys),
        total=len(task_data) - len(exclude_keys),
    ):
        prompt = f"\n{context}\n{input_item}" if context else f"\n{input_item}"
        answer = infer(model, tokenizer, prompt, device)
        match = target_output.startswith(answer) or answer.startswith(target_output)
        match_count += match
        total_count += 1
    return round((match_count / total_count) * 100, 4)


@click.command()
@click.option(
    "--model-size",
    type=click.STRING,
    callback=validate_model_size,
    help="Model size must be one of: 1B, 3B, 11B, 90B",
)
@click.option("--task", type=click.STRING)
@click.option("--nb-demonstrations", type=click.INT, required=False, default=5)
@click.option("--nb-runs", type=click.INT, required=False, default=1)
def main(model_size: LlamaModelSize, task: str, nb_demonstrations: int = 5, nb_runs: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    model, tokenizer = load_model(model_size, device)
    task_data = load_data(task, tokenizer)

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
    for _ in tqdm(range(nb_runs), desc="Runs"):
        # Use different few-shot data for each run
        few_shot_prompts, few_shot_data = get_icl_prompt(task_data, nb_demonstrations)
        exclude_keys = set(few_shot_data.keys())

        # get baseline standard ICL accuracy
        acc_icl = get_task_accuracy(
            model=model,
            tokenizer=tokenizer,
            task_data=task_data,
            exclude_keys=exclude_keys,
            device=device,
            context="\n".join(few_shot_prompts),
        )
        results_per_layer["icl"].append(acc_icl)
        tqdm.write(f"ICL Accuracy: {acc_icl}%")

        # get accuracy per layer with task vectors
        for layer_index, layer in tqdm(
            enumerate(model.model.layers),
            desc="Layers",
            total=len(model.model.layers),
        ):
            task_vectors = []  # Reset task vectors for each layer
            extract_hook = layer.register_forward_hook(hook_extract)
            infer(model, tokenizer, prompt="\n".join(few_shot_prompts), device=device)
            extract_hook.remove()
            inject_hook = layer.register_forward_hook(
                partial(hook_inject, task_vectors=task_vectors[0])
            )
            acc = get_task_accuracy(
                model=model,
                tokenizer=tokenizer,
                task_data=task_data,
                exclude_keys=exclude_keys,
                device=device,
            )
            tqdm.write(f"Layer {layer_index} Accuracy: {acc}%")
            results_per_layer[layer_index].append(acc)
            extract_hook.remove()
            inject_hook.remove()

    save_results(
        results_per_layer,
        filename=f"{model_size}_{task}_{nb_demonstrations}_demos_{nb_runs}_runs.csv",
    )


if __name__ == "__main__":
    main()
