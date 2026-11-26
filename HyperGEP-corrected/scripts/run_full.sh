#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
data="${DATA_DIR:-data}"
out="${OUTPUT_DIR:-artifacts/corrected}"
device="${DEVICE:-cpu}"
status=0
for dataset in maizegep g2f; do
  for experiment in main baselines ablation negative_transfer sensitivity interpretability; do
    python -m "experiments.run_$experiment" --dataset "$dataset" --config configs/paper.yaml \
      --data-dir "$data" --output-dir "$out" --device "$device" --resume || status=2
  done
done
python -m experiments.run_genomic_comparators --dataset g2f --config configs/paper.yaml \
  --data-dir "$data" --output-dir "$out" --device "$device" --resume || status=2
python -m experiments.run_loeo --dataset maizegep --config configs/paper.yaml \
  --data-dir "$data" --output-dir "$out" --device "$device" --resume || status=2
python -m scripts.aggregate_results "$out" --output-dir "$out/summary" || status=2
python -m scripts.meta_dependencies "$out" --output-dir "$out/trait_dependencies" || status=2
for dataset in maizegep g2f; do
  python -m scripts.attribution_panels "$out/corrected-v1/$dataset/interpretability" \
    --markers "$data/$dataset/markers.csv" --output-dir "$out/interpretability/$dataset" || status=2
done
exit "$status"
