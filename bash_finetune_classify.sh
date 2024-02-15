#!/usr/bin/bash
echo "Finetuning MAE"

# Set the parameters

BATCH_SIZE=32
EPOCHS=100
WARMUP_EPOCHS=10
ACCUM_ITER=1
# Model Parameters
MODEL='vit_base_patch16'
INPUT_SIZE=96
IN_CHANS=4
# Optimizer parameters
LR=1e-3
BLR=1e-3
MIN_LR=1e-3
# Fine tuning parameters
# Dataset parameters
FINETUNE_PATH="/home/jazib/projects/mae/pretrain_models/vit_base_240209.ckpt"
DATA_PATH="/home/jazib/projects/SelfSupervisedLearning/species_classification/"
TRAIN_RATIO=0.8
# NB_CLASSES=10
OUTPUT_DIR="/home/jazib/projects/mae/finetune_class_240215a/"
LOG_DIR="/home/jazib/projects/mae/finetune_class_240215a/"
# RESUME="/home/jazib/projects/mae/finetune_class_240213a/checkpoint-49.pth"
START_EPOCH=0
DEVICE="cuda"



# Run the python script
python main_finetune.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --accum_iter $ACCUM_ITER \
  --model $MODEL \
  --input_size $INPUT_SIZE \
  --in_chans $IN_CHANS \
  --lr $LR \
  --blr $BLR \
  --min_lr $MIN_LR \
  --warmup_epochs $WARMUP_EPOCHS \
  --finetune $FINETUNE_PATH \
  --data_path $DATA_PATH \
  --train_ratio $TRAIN_RATIO \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --device $DEVICE \
  --start_epoch $START_EPOCH

# END SCRIPT


