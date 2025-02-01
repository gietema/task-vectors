#!/bin/bash

# Define categories and tasks as separate arrays
categories=("knowledge")
translation_tasks=("en_es" "en_fr" "es_en" "fr_en")
linguistic_tasks=("antonyms" "singular_plural" "present_simple_gerund" "present_simple_past_simple")
knowledge_tasks=("country_capital" "location_continent" "location_religion" "person_language")

demonstrations=(2 5 10 20)

# Loop through each category
for category in "${categories[@]}"; do
  # Dynamically get the tasks array for the current category
  tasks_var="${category}_tasks[@]"
  tasks=("${!tasks_var}")

  for task in "${tasks[@]}"; do
    for nb_demonstrations in "${demonstrations[@]}"; do
      poetry run python scripts/main.py --model-size Llama-3.2-1B --task "text/$category/$task" --nb-runs 5 --nb-demonstrations "$nb_demonstrations"
    done
  done
done