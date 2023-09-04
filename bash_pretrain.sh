#!/usr/bin/bash
echo "Pretraining MAE"

# Change to the directory containing the python script
# example: cd /path/to/python.script
# Not needed in the current script - it's already in current dir

# Set the variables
# Training Parameters
BATCH_SIZE=32
EPOCHS=60
ACCUM_ITER=1 # default
# Model Parameters
MODEL='mae_vit_base_patch16' # default
INPUT_SIZE=224
MASK_RATIO=0.3 # default 0.75
IN_CHANS=4
# Optimizer Parameters
WEIGHT_DECAY=0.05 # default - also, w/o weight decay there's nans - network collapse probs.
LR=10e-4
BLR=10e-3
MIN_LR=10e-6
WARMUP_EPOCHS=10
# Dataset Parameters
DATA_PATH='/home/jazib/projects/SelfSupervisedLearning/'
DATA_LIST='/home/jazib/projects/SelfSupervisedLearning/NRW5k_filelist.txt'
OUTPUT_DIR='./logs_outputs_aug30/'
LOG_DIR='./logs_outputs_aug30/'
DEVICE='cuda' # default
RESUME="/home/jazib/projects/mae/logs_outputs_aug30/checkpoint-39.pth"
START_EPOCH=40
NUM_WORKERS=4



# Run the python script
python main_pretrain.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --accum_iter $ACCUM_ITER \
  --model $MODEL \
  --input_size $INPUT_SIZE \
  --mask_ratio $MASK_RATIO \
  --in_chans $IN_CHANS \
  --weight_decay $WEIGHT_DECAY \
  --lr $LR \
  --blr $BLR \
  --min_lr $MIN_LR \
  --warmup_epochs $WARMUP_EPOCHS \
  --data_path $DATA_PATH \
  --data_list $DATA_LIST \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --device $DEVICE \
  --resume $RESUME \
  --start_epoch $START_EPOCH \
  --num_workers $NUM_WORKERS

