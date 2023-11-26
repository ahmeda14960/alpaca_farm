run_number=$1
# add line that makes the output dir
current_datetime=$(date +"%H%M%S")



seed=${1:-0}
num_heads=${2:-3}

output_dir=/lfs/skampere1/0/ahmedah/logs/alp_multi_rw_opt_{$num_heads}_heads${current_datetime}
mkdir -p $output_dir
GPUS=4
GA=8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1343 examples/multi_reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed $seed \
  --model_name_or_path "/lfs/skampere1/0/ahmedah/logs/opt1b-alp-sft/" \
  --dataset_name "alpaca_human_preference" \
  --output_dir "$output_dir" \
  --model_max_length 512 \
  --num_train_epochs 1 \
  --num_heads $num_heads \
  --per_device_train_batch_size 2 \
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
  --wandb_project "alpaca_farm" \
  --run_name "alp-rwl-opt1b-multi-{$num_heads}heads_{$seed}" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800 \
  --initialize_model_on_cpu True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer"