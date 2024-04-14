

#current_datetime=$(date +"%Y%m%d%H%M%S")
#output_dir=/home/azureuser/out/alp_rw_opt_${current_datetime}
#mkdir -p $output_dir

# terrible hack where batch is 511 since 7 gpus * 73 = 511.. close enough to 512
run_name=$1
model_name_or_path=${2:-"/self/scr-sync/ahmedah/opt7brwlshp/"}
dataset_name=${3:-"stanfordnlp/SHP"}
torchrun --nproc_per_node=4 --master_port=1343 examples/reward_modeling.py \
  --fp16 False \
  --bf16 True \
  --seed 0 \
  --model_name_or_path "/iris/u/ahmedah/opt7bsftalp/" \
  --dataset_name "alpaca_noisy_multi_preference" \
  --output_dir "/iris/u/ahmedah/opt125mrwlalp/" \
  --model_max_length 512 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_steps 10 \
  --save_strategy "steps" \
  --save_steps 1000000 \
  --save_total_limit 1 \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm" \
  --run_name "${run_name}" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  --tf32 True \
  --flash_attn True \
  --ddp_timeout 1800 \
  --initialize_model_on_cpu True
