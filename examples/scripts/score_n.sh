#!/bin/bash

# Get current timestamp
current_time=$(date +"%Y%m%d%H%M%S")

# Set num_sequences_param from the first script argument, default to 16 if not provided
num_sequences_param=${1:-16}
# same for batch size
batch_size_param=${2:-128}

# Construct the output file name
output_file_name="/lfs/skampere1/0/ahmedah/logs/bon_opt_alp/output_7b${current_time}_${num_sequences_param}.json"


# Run the script with the dynamically generated output path
echo "Running score_n.py with output path: $output_file_name"

python examples/score_n.py \
  --task "score_n" \
  --decoder_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-sft/" \
  --scorer_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-rwl/" \
  --input_path "/lfs/skampere1/0/ahmedah/logs/bon_opt_alp/final_1200.json" \
  --num_return_sequences $num_sequences_param \
  --per_device_batch_size $batch_size_param \
  --split "eval" \
  --mixed_precision "bf16" \
  --tf32 True \
  --flash_attn True \
  --output_path $output_file_name \
  --rerank_top_k 1 \
  --dump_all False \
  --singleton True