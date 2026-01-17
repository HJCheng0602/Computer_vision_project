
DATA_DIR="data/processed_data64"
CHECKPOINT_PATH="checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64_Resnet_SN/generator_epoch_100.pth"
MODEL_TYPE="resnet_SN"

python eval.py \
    --model_type "$MODEL_TYPE" \
    --data_dir "$DATA_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH"