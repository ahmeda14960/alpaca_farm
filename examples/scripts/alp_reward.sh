run_number=$1
# add line that makes the output dir

GPUS=4
GA=8
#CUDA_VISIBLE_DEVICES=$run_number 
torchrun --nproc_per_node=1 --master_port=1343 examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --model_name_or_path "/home/azureuser/out/alp_opt_sft/" \
  --dataset_name "alpaca_human_preference" \
  --output_dir "/home/azureuser/out/alp_rwl/" \
  --model_max_length 512 \
  --num_train_epochs 0.1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_steps 100 \
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
  --run_name "alp-rwl-opt1b" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800 \
  --initialize_model_on_cpu True
  #--fsdp "full_shard auto_wrap" \
  #--fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
 
