# Try to implement proper metric for test function

# Try to implement some post-processing methds for visual evaluation, Display your generated results
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from pyvox.models import Vox
from pyvox.writer import VoxWriter

from model import Generator 

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
    print("=== Start VOX Generation (with PyVox) ===")
    
    CHECKPOINT_PATH = "checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64/generator_epoch_100.pth" 
    TEST_DATA_PATH = "data/processed_data64/test/5/ER_23-n007-t1649439072.npy"
    OUTPUT_DIR = "output64/vox_results"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(TEST_DATA_PATH):
        print("Test data not found, using dummy data.")
        test_frag = np.zeros((32,32,32), dtype=np.uint8)
        test_frag[10:22, 10:22, 10:22] = 1
    else:
        test_frag = np.load(TEST_DATA_PATH)
    frags = np.unique(test_frag)[1:]
    
    selected_frag = np.random.choice(frags, size=1, replace=False)
    test_frag = (test_frag == selected_frag[0]).astype(np.uint8)

    try:
        current_len = test_frag.shape[0]
        model = Generator(cube_len=current_len, z_latent_space=64).to(available_device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=available_device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    print("Generating...")
    fake_prob, original_frag = generate(model, test_frag)
    
    fake_prob = fake_prob.squeeze()
    original_frag = original_frag.squeeze()
    
    final_result = posprocessing(fake_prob, original_frag)

    print("-" * 30)
    save_vox_file(original_frag, os.path.join(OUTPUT_DIR, "input_frag.vox"))
    save_vox_file(final_result, os.path.join(OUTPUT_DIR, "completed_result.vox"))
    print("-" * 30)
    