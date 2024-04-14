#!/bin/bash

# Set default values


# Override defaults with command-line arguments if provided
run_name=${1:-"opt-6.7b-sft-seed2"}
output_dir=${2:-/iris/u/ahmedah/opt7b_sft_alp_seed2/}
dataset_name=${3:-"alpaca_instructions"}
model_name_or_path=${4:-"facebook/opt-6.7b"}
n_proc=${5:-4}

# Rest of the script with the specified or default parameters

torchrun --nproc_per_node=$n_proc --master_port=1242 examples/supervised.py \
  --model_name_or_path "$model_name_or_path" \
  --fp16 False \
  --bf16 True \
  --seed 2 \
  --output_dir "$output_dir" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_steps 10 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 1 \
  --wandb_project "alpaca_farm" \
  --run_name "$run_name" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  --dataset_name "$dataset_name" 
