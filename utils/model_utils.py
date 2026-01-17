# Try to implement proper metric for test function

# Try to implement some post-processing methds for visual evaluation, Display your generated results
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from pyvox.models import Vox
from pyvox.writer import VoxWriter

from model import Generator_Init, Generator_res, Generator_wider
from visualize import plot, plot_join

available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_vox_file(voxel_data, filename):
    try:
        dense_data = voxel_data > 0.5 
        vox_model = Vox.from_dense(dense_data)
        
        VoxWriter(filename, vox_model).write()
        print(f"VOX saved: {filename}")
        
    except Exception as e:
        print(f"Failed to save vox file: {e}")

def posprocessing(fake, mesh_frag):
    fake_binary = (fake > 0.5).astype(np.float32)
    final_vox = np.maximum(fake_binary, mesh_frag)
    return final_vox

def generate(model, vox_frag):
    mesh_frag = torch.Tensor(vox_frag).unsqueeze(0).unsqueeze(0).float().to(available_device)
    
    with torch.no_grad():
        output_g_encode = model.forward_encode(mesh_frag)
        fake = model.forward_decode(output_g_encode)
    
    fake = fake + mesh_frag
    fake = torch.clamp(fake, 0, 1)
    
    fake = fake.detach().cpu().numpy()
    mesh_frag = mesh_frag.detach().cpu().numpy()
    return fake, mesh_frag

if __name__ == "__main__":
    print("=== Start Inference & Visualization ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CHECKPOINT_PATH = "checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64/generator_epoch_100.pth" 
    # TEST_DATA_PATH = "data/processed_data64/test/5/ER_23-n007-t1649439072.npy"
    # OUTPUT_DIR = "output64/vox_results"
    # CUBE_LEN = 64
    
    # detect the following args
    # like python xx.py --model_type Init --output_dir output --model_path checkpoint.pth --test_data_folder data/test --test_num 10 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="Init", help="Type of model to use (e.g., Init, Wider_G, Wider_GSN, resnet_SN)")
    parser.add_argument("--output_dir", type=str, default="output64/vox_results", help="Directory to save output results")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--data_dir", type=str, help="Path to the test fragment data")
    parser.add_argument("--test_num", type=int, default=1, help="Number of test samples to generate")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    CHECKPOINT_PATH = args.model_path
    TEST_DATA_PATH = args.data_dir
    TEST_NUM = args.test_num
    CUBE_LEN = 64
    
    if args.model_type == "Init":
        model = Generator_Init(cube_len=CUBE_LEN, z_latent_space=64).to(device)
    elif args.model_type == "Wider_G":
        model = Generator_wider(cube_len=CUBE_LEN, z_latent_space=64).to(device)    
    elif args.model_type == "Wider_GSN":
        model = Generator_wider(cube_len=CUBE_LEN, z_latent_space=64).to(device)
    elif args.model_type == "resnet_SN":
        model = Generator_res(cube_len=CUBE_LEN, z_latent_space=64).to(device)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[Warning] Test data not found: {TEST_DATA_PATH}")
        exit()
        test_frag_raw = np.zeros((CUBE_LEN, CUBE_LEN, CUBE_LEN), dtype=np.uint8)
        test_frag_raw[20:40, 20:40, 20:40] = 1
        selected_frag_id = 1
    else:
        # test_frag_raw = np.load(TEST_DATA_PATH)
        # frags_ids = np.unique(test_frag_raw)[1:] 
        # selected_frag_id = np.random.choice(frags_ids, size=1, replace=False)[0]
        selected_files = os.listdir(TEST_DATA_PATH)
        if len(selected_files) == 0:
            print(f"[Warning] No test files found in: {TEST_DATA_PATH}")
            exit()
        else:
            selected_files = selected_files[:TEST_NUM]
            test_frag_raw_list = [np.load(os.path.join(TEST_DATA_PATH, f)) for f in selected_files]
            frags_ids_list = [np.unique(vox_item)[1:] for vox_item in test_frag_raw_list]
            selected_frag_ids = [np.random.choice(ids, size=1, replace=False)[0] for ids in frags_ids_list]
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    i = 0
    for vox_item in test_frag_raw_list:
        print(f"Processing fragment shape: {vox_item.shape}")
        
        input_frag_np = (vox_item == selected_frag_ids[0]).astype(np.uint8)
        input_tensor = torch.from_numpy(input_frag_np).float().unsqueeze(0).unsqueeze(0).to(device)
        print("-> Generating restoration...")
        with torch.no_grad():
            fake_logits = model(input_tensor)
            fake_prob = torch.sigmoid(fake_logits).cpu().numpy().squeeze()
            generated_binary = (fake_prob > 0.6).astype(np.uint8)
        final_result = np.clip(generated_binary + input_frag_np, 0, 1)
        save_name = f"completed_result_{selected_frag_ids[i]}.vox"
        print(f"-> Saving results to {OUTPUT_DIR}/{save_name} ...")
        save_vox_file(input_frag_np, os.path.join(OUTPUT_DIR, f"input_frag_{selected_frag_ids[i]}.vox"))
        save_vox_file(final_result, os.path.join(OUTPUT_DIR, save_name))
        
        # gt save
        save_vox_file((vox_item > 0).astype(np.uint8), os.path.join(OUTPUT_DIR, f"gt_{selected_frag_ids[i]}.vox"))
        i += 1
    print("=== Done! ===")
    
    # print(f"-> Selected Fragment ID: {selected_frag_id}")

    # input_frag_np = (test_frag_raw == selected_frag_id).astype(np.uint8)
    
    # input_tensor = torch.from_numpy(input_frag_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # try:
    #     model = Generator(cube_len=CUBE_LEN, z_latent_space=64).to(device)
    #     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    #     model.load_state_dict(checkpoint)
    #     model.eval()
    #     print("-> Model loaded successfully.")
    # except Exception as e:
    #     print(f"[Error] Failed to load model: {e}")
    #     exit()

    # print("-> Generating restoration...")
    # with torch.no_grad():
    #     fake_logits = model(input_tensor)
    #     fake_prob = torch.sigmoid(fake_logits).cpu().numpy().squeeze()
    #     generated_binary = (fake_prob > 0.5).astype(np.uint8)

    # final_result = np.clip(generated_binary + input_frag_np, 0, 1)

    # print(f"-> Saving results to {OUTPUT_DIR} ...")
    
    # save_vox_file(input_frag_np, os.path.join(OUTPUT_DIR, "input_frag.vox"))
    # save_vox_file(final_result, os.path.join(OUTPUT_DIR, "completed_result.vox"))

    # # plot(input_frag_np, OUTPUT_DIR, "view_input.png")
    
    # # plot(final_result, OUTPUT_DIR, "view_completed.png")
    
    # # plot_join(input_frag_np, final_result, OUTPUT_DIR, "view_comparison.png")

    # print("=== Done! ===")
    