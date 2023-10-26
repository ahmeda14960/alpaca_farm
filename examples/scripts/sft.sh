#output_dir=$1
run_number=$1
run_name=$2
model_name_or_path=$3

#CUDA_VISIBLE_DEVICES=$run_number 
torchrun --nproc_per_node=4 --master_port=1242 examples/supervised.py \
  --model_name_or_path "facebook/opt-1.3b" \
  --dataset_name "stanfordnlp/SHP" \
  --fp16 False \
  --bf16 True \
  --seed $run_number \
  --output_dir "/data/ahmed_mohamed_ahmed/code/workstream1_code/output_results" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm_debug" \
  --run_name "${run_name}-debug" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  --train_splits "sft"
