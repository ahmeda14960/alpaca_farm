#!/bin/bash


run_name=$1
kl_coef=${2:-0.01}
reward_model_name_or_path=$3
policy_model_name_or_path=$4

config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_opt_8gpu.yaml"

accelerate launch --config_file "${config_file}" examples/rlhf_ppo.py \
  --seed 0 \
  --run_name "${run_name}" \
  --dataset_name "alpaca_instructions" \
  --step_per_device_batch_size 4 \
  --rollout_per_device_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --output_dir "/iris/u/ahmedah/ppo_alp_ensemble_seed0/" \
  --reward_model_name_or_path "/self/scr-sync/ahmedah/opt1brwlalp/" \
  --gold_reward_model_name_or_path "/self/scr-sync/ahmedah/opt7brwlalp/" \
  --policy_model_name_or_path "/self/scr-sync/ahmedah/opt1bsftalp/" \
  --init_value_with_reward True \
  --rollout_batch_size 512 \
  --step_batch_size 256 \
  --learning_rate 6e-6 \
  --warmup_steps 10 \
  --kl_coef "${kl_coef}" \
  --total_epochs 8 \
  --flash_attn True \
  --ensemble True \
  --multi False \
  --varnorm False \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 20
