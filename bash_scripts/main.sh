#!/bin/bash

declare -A tasks=(
  ["translation"]="en_es en_fr es_en fr_en"
  ["linguistic"]="antonyms singular_plural present_simple_gerund present_simple_past_simple"
  ["knowledge"]="country_capital location_continent location_religion person_language"
)

demonstrations=(2 5 10 20)

for category in "${!tasks[@]}"; do
  for task in ${tasks[$category]}; do
    for nb_demonstrations in "${demonstrations[@]}"; do
      poetry run python scripts/main.py --model-size 1B --task "$category/$task" --nb-runs 5 --nb-demonstrations "$nb_demonstrations"
    done
  done
done
