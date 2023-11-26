<<<<<<< Updated upstream
#output_dir=$1
run_number=$1
run_name=azure_sft
=======
run_name=$1
>>>>>>> Stashed changes
model_name_or_path=$3

<<<<<<< Updated upstream
#CUDA_VISIBLE_DEVICES=$run_number 
torchrun --nproc_per_node=1 --master_port=1242 examples/supervised.py \
=======
current_datetime=$(date +"%Y%m%d%H%M%S")
<<<<<<< Updated upstream
output_dir=/home/azureuser/out/1b_alpsft_${current_datetime}
=======
output_dir=/lfs/skampere1/0/ahmedah/logs/opt125m_alpsft_${current_datetime}
>>>>>>> Stashed changes

mkdir -p $output_dir

echo "Saving to $output_dir"

#CUDA_VISIBLE_DEVICES=$run_number 
<<<<<<< Updated upstream
torchrun --nproc_per_node=4 --master_port=1242 examples/supervised.py \
>>>>>>> Stashed changes
  --model_name_or_path "facebook/opt-1.3b" \
=======
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1234 examples/supervised.py \
  --model_name_or_path "facebook/opt-125m" \
>>>>>>> Stashed changes
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --dataset_name "alpaca_instructions" \
<<<<<<< Updated upstream
  --output_dir "/home/azureuser/out/alp_opt_sft" \
  --num_train_epochs 0.1 \
=======
  --output_dir $output_dir \
  --num_train_epochs 3 \
>>>>>>> Stashed changes
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "alpaca_farm_debug" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --train_splits "sft" \
  #--fsdp "full_shard auto_wrap" \
  #--fsdp_transformer_layer_cls_to_wrap "OPTDecoderLayer" \
  
