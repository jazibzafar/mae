#!/usr/bin/bash
echo "Linear Probing MAE"

# Set the parameters

BATCH_SIZE=32
EPOCHS=50
WARMUP_EPOCHS=10
ACCUM_ITER=1
# Model Parameters
MODEL='vit_base_patch16'
INPUT_SIZE=96
IN_CHANS=4
# Optimizer parameters
LR=10e-4
BLR=10e-4
MIN_LR=10e-4
# Fine tuning parameters
# Dataset parameters
FINETUNE_PATH="/home/jazib/projects/mae/experiment_baseline_1309/checkpoint-99.pth"
DATA_PATH="/home/jazib/projects/SelfSupervisedLearning/species_classification/"
TRAIN_RATIO=0.8
# NB_CLASSES=10
OUTPUT_DIR="/home/jazib/projects/mae/lineval_class_2011a/"
LOG_DIR="/home/jazib/projects/mae/lineval_class_2011a/"
# RESUME="/home/jazib/projects/mae/finetune_class_2410d/checkpoint-49.pth"
START_EPOCH=0
DEVICE="cuda"



# Run the python script
python main_linprobe.py \
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


