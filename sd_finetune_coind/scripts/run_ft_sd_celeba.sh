export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_DIR="/research/hal-datastore/datasets/original/celeba/"
export DATASET_NAME="celeba"
export CACHE_DIR="/research/hal-gaudisac/Diffusion/CoInD-celeba/cache/"
export OUTPUT_DIR="/research/hal-gaudisac/Diffusion/CoInD-celeba/outputs/celeba_sd3"

CUDA_VISIBLE_DEVICES=2,3 /research/hal-gaudisac/miniconda3/bin/accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port 29500 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_data_dir=$DATASET_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --cache_dir=$CACHE_DIR \
  --output_dir=${OUTPUT_DIR}
