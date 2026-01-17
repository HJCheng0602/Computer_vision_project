#!/bin/bash


DATA_DIR="data/processed_data64/test/5"
MODEL_DIR="checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64_Resnet_SN/generator_epoch_100.pth"
OUTPUTDIR="data/visualization_outputs"
MODEL_TYPE="resnet_SN"
TEST_NUM=5


TESTNAME="Z64_Glr0.002_Dlr0.0002_Res64_BS64_Resnet_SN"

python utils/model_utils.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUTDIR" \
    --test_num $TEST_NUM