#output_dir=$1
run_name=$1
model_name_or_path=$3

torchrun --nproc_per_node=1 --master_port=1242 examples/supervised.py \
  --model_name_or_path "facebook/opt-1.3b" \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "/scr-ssd/ahmedah/alp/" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 2048 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  --train_splits "sft"
