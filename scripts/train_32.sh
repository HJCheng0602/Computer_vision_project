#!/bin/bash


DATA_DIR="/data/jincheng/cv_project/data/processed_data"
MODEL_DIR="/data/jincheng/cv_project/models/model_32"
BATCH_SIZE=64
EPOCHS=100

mkdir -p "$MODEL_DIR"

TESTNAME="Z64_Glr0.002_Dlr0.0002_Res32_BS64"

python /data/jincheng/cv_project/train.py \
    --Z_latent_space 64 \
    --G_lr 2e-3 \
    --D_lr 2e-4 \
    --beta1 0.9 \
    --beta2 0.999 \
    --resolution 32 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dirdataset "$DATA_DIR" \
    --test_save_dir "test_outputs" \
    --log_name "$TESTNAME" \