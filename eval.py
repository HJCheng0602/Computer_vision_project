# eval_baseline.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from utils.FragmentDataset import FragmentDataset
from utils.model import Generator_Init, Generator_wider, Generator_res
from test import test  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = 64  
BATCH_SIZE = 64


def run_baseline(model, args):
    print(f"Loading baseline model from: {args.checkpoint_path}")
    
    test_dataset = FragmentDataset(args.data_dir, vox_type='test', dim_size=RESOLUTION)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("如果是路径错误，请在代码里修改 CHECKPOINT_PATH。")
        return

    print("Running evaluation on Test Set...")
    test(model, test_dataloader, save_dir="test_outputs_baseline", device=DEVICE, epoch=0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Directory of test dataset")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="Init", help="Type of model to use (e.g., Init, Wider_G, Wider_GSN, resnet_SN)")
    args = parser.parse_args()
    if args.model_type == "Init":
        model = Generator_Init(cube_len=RESOLUTION, z_latent_space=64).to(DEVICE)
    elif args.model_type == "Wider_G":
        model = Generator_wider(cube_len=RESOLUTION, z_latent_space=64).to(DEVICE)    
    elif args.model_type == "Wider_GSN":
        model = Generator_wider(cube_len=RESOLUTION, z_latent_space=64).to(DEVICE)
    elif args.model_type == "resnet_SN":
        model = Generator_res(cube_len=RESOLUTION, z_latent_space=64).to(DEVICE)
    run_baseline(model, args)