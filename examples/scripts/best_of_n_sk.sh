#!/bin/bash

# Get current timestamp
current_time=$(date +"%Y%m%d%H%M%S")

# Set num_sequences_param from the first script argument, default to 16 if not provided
num_sequences_param=${1:-16}

# Construct the output file name
output_file_name="/lfs/skampere1/0/ahmedah/logs/bon_opt_alp/output_${current_time}_${num_sequences_param}.json"

# Run the script with the dynamically generated output path
python examples/best_of_n.py \
  --task "run_best_of_n" \
  --decoder_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-sft/" \
  --scorer_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-rwl/" \
  --num_return_sequences $num_sequences_param \
  --per_device_batch_size 32 \
  --split "eval" \
  --mixed_precision "bf16" \
  --tf32 True \
  --flash_attn True \
  --output_path $output_file_name \
  --rerank_top_k 4 \
  --dump_all True
