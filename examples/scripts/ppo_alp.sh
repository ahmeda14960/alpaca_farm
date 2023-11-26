run_name=$2
reward_model_name_or_path=$3
policy_model_name_or_path=$4
<<<<<<< Updated upstream
kl_coef=${5:-0.01}
=======
kl_coef=${5:-0.03}
>>>>>>> Stashed changes

config_file="./examples/accelerate_configs/rlhf_ppo_fsdp_opt_8gpu.yaml"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file "${config_file}" examples/rlhf_ppo.py \
  --run_name "alp-ppo-opt1b_32bsz_lr2e-5" \
  --step_per_device_batch_size 4 \
  --rollout_per_device_batch_size 32 \
  --per_device_eval_batch_size 32 \
<<<<<<< Updated upstream
  --output_dir "/home/azureuser/out/alp_ppo_opt" \
  --reward_model_name_or_path "/home/azureuser/out/alp_rwl/" \
  --policy_model_name_or_path "/home/azureuser/out/alp_opt_sft/" \
  --init_value_with_reward True \
  --rollout_batch_size 512 \
  --step_batch_size 512 \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --kl_coef "${kl_coef}" \
  --total_epochs 30 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --save_steps 20 \
  --dataset_name alpaca_instructions
=======
  --output_dir "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-ppo/" \
  --reward_model_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-rwl/" \
  --policy_model_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-sft/" \
  --init_value_with_reward True \
  --rollout_batch_size 512 \
  --step_batch_size 256 \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --kl_coef "${kl_coef}" \
  --total_epochs 10 \
  --flash_attn True \
  --prompt_dict_path "./examples/prompts/v0_inputs_noinputs.json" \
  --dataset_name "alpaca_instructions" \
  --save_steps 20
>>>>>>> Stashed changes
