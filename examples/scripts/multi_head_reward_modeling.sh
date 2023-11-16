run_number=$1
# add line that makes the output dir
# mkdir -p "/scr-ssd/ahmedah/debug-shp-rwl1b-${run_number}/"

current_datetime=$(date +"%Y%m%d%H%M%S")
output_dir=/home/azureuser/out/alp_rw_opt_${current_datetime}
mkdir -p $output_dir

GPUS=4
GA=8

torchrun --nproc_per_node=4 --master_port=1343 examples/multi_reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed 0 \
  --model_name_or_path "//home/azureuser/out/opt_1b_alpsft_20231116213715" \
  --dataset_name "alpaca_human_preference" \
  --output_dir "$output_dir" \
  --model_max_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --eval_steps 10 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm_debug" \
  --run_name "alp-rwl-opt1b-multi-part2" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800 \
  --initialize_model_on_cpu True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" 
