#!/usr/bin/bash
echo "Pretraining MAE"

# Change to the directory containing the python script
# example: cd /path/to/python.script
# Not needed in the current script - it's already in current dir

# Set the variables
# Training Parameters
BATCH_SIZE=128
MAX_STEPS=100000
ACCUM_ITER=1 # default
# Model Parameters
MODEL='mae_vit_base_patch16' # default
INPUT_SIZE=224
MASK_RATIO=0.6 # default 0.75
IN_CHANS=4
# Optimizer Parameters
WEIGHT_DECAY=0.05 # default - also, w/o weight decay there's nans - network collapse probs.
LR=10e-4
BLR=10e-4
MIN_LR=10e-4
#WARMUP_EPOCHS=10
# Dataset Parameters
DATA_PATH='/data_hdd/nrw_dop10/nrw_dop10_tars/nrw_dop10-{0000..0099}.tar'
DATA_LIST='/mnt/cluster/data_hdd/nrw_dop10/nrw_25k.txt'
OUTPUT_DIR='./experiment_test_300124/'
LOG_DIR='./experiment_test_300124/'
DEVICE='cuda' # default
RESUME="/home/jazib/projects/mae/experiment_baseline_050124/checkpoint-19.pth"
#START_EPOCH=0
NUM_WORKERS=19



# Run the python script
python light_pretrain.py \
  --batch_size $BATCH_SIZE \
  --max_steps $MAX_STEPS \
  --accum_iter $ACCUM_ITER \
  --model $MODEL \
  --input_size $INPUT_SIZE \
  --mask_ratio $MASK_RATIO \
  --in_chans $IN_CHANS \
  --weight_decay $WEIGHT_DECAY \
  --lr $LR \
  --blr $BLR \
  --min_lr $MIN_LR \
  --data_path $DATA_PATH \
  --data_list $DATA_LIST \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOG_DIR \
  --num_workers $NUM_WORKERS

#  --resume $RESUME \ <- add between device and start_epoch