# Task Vectors for In-Context-Learning

Reproducing [In-Context Learning Creates Task Vectors](https://arxiv.org/abs/2310.15916) by
Roee Hendel, Mor Geva and Amir Globerson (2023). 

# Reproduce results
```commandline
poetry install
```

Before running the script, get access to the `huggingface/llama` models.
You'll have to request access [here](https://huggingface.co/meta-llama/Llama-3.2-1B).
Make sure to add an affiliation to your request as your request might be denied without it.
You can then download the models after `huggingface-cli login` and providing a *read* [access token](https://huggingface.co/settings/tokens) (not a fine-grained one).


Then, reproduce the results with this command:
```commandline
bash bash_scripts/main.sh
```
