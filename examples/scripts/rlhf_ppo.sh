#!/bin/bash


run_name=$1
kl_coef=${2:-0.3}
reward_model_name_or_path=$3
policy_model_name_or_path=$4

config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_opt_8gpu.yaml"

accelerate launch --config_file "${config_file}" examples/rlhf_ppo.py \
  --run_name "${run_name}" \
  --step_per_device_batch_size 4 \
  --rollout_per_device_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --output_dir "/iris/u/ahmedah/opt1bppoalp/" \
  --reward_model_name_or_path "/iris/u/ahmedah/opt1bmultirwlalp/" \
  --gold_reward_model_name_or_path "/self/scr-sync/ahmedah/opt7brwl/" \
  --policy_model_name_or_path "/iris/u/ahmedah/opt1bsftalp/" \
  --init_value_with_reward True \
  --rollout_batch_size 512 \
  --step_batch_size 256 \
  --learning_rate 6e-6 \
  --warmup_steps 10 \
  --kl_coef "${kl_coef}" \
  --total_epochs 100 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 100