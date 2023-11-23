#!/usr/bin/bash
echo "Finetuning MAE"

# Set the parameters

BATCH_SIZE=32
EPOCHS=50
WARMUP_EPOCHS=10
ACCUM_ITER=1
# Model Parameters
MODEL='vit_base_patch16'
INPUT_SIZE=96
# IN_CHANS=4
# Optimizer parameters
LR=10e-4
BLR=10e-4
MIN_LR=10e-4
# Fine tuning parameters
# Dataset parameters
# FINETUNE_PATH="/home/jazib/projects/mae/experiment_baseline_1309/checkpoint-99.pth"
DATA_PATH="/home/jazib/projects/SelfSupervisedLearning/species_classification/"
TRAIN_RATIO=0.1
NB_CLASSES=10
OUTPUT_DIR="/home/jazib/projects/mae/finetune_image_2111b/"
LOG_DIR="/home/jazib/projects/mae/finetune_image_2111b/"
# RESUME="/home/jazib/projects/mae/finetune_class_2410d/checkpoint-49.pth"
START_EPOCH=0
DEVICE="cuda"



# Run the python script
python main_finetune_imagenet.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --accum_iter $ACCUM_ITER \
  --input_size $INPUT_SIZE \
  --lr $LR \
  --blr $BLR \
  --min_lr $MIN_LR \
  --warmup_epochs $WARMUP_EPOCHS \
  --data_path $DATA_PATH \
  --train_ratio $TRAIN_RATIO \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --nb_classes $NB_CLASSES \
  --device $DEVICE \
  --start_epoch $START_EPOCH

# END SCRIPT


