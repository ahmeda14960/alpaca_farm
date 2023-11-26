run_name=$1
model_name_or_path=$3

current_datetime=$(date +"%Y%m%d%H%M%S")
output_dir=/lfs/skampere1/0/ahmedah/logs/opt125m_alprwl_${current_datetime}

mkdir -p $output_dir

echo "Saving to $output_dir"

#CUDA_VISIBLE_DEVICES=$run_number 
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1343 examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --model_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt125m_alpsft_20231117135643/" \
  --dataset_name "alpaca_human_preference" \
  --output_dir $output_dir \
  --model_max_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_steps 10 \
  --save_strategy "epoch" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm" \
  --run_name $1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800 \
  --initialize_model_on_cpu True
