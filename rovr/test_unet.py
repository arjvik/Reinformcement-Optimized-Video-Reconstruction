from rovr_unet import ROVR
from video_ds import VideoDataset
from local_net import LocalNetwork
from action_lstm import ActionLSTM
from policy_net_1_unet import PolicyNetwork1UNet
from policy_net_2_unet import PolicyNetwork2UNet
from resnet_extractor import ResnetFeatureExtractor
from GPUtil import showUtilization as gpu_usage, getAvailable
from logger import Logger

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import argparse

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--vid_length', default=25, type=int, help="Length of the video")
parser.add_argument('--time_steps', default=25, type=int, help="Number of time steps")
parser.add_argument('--n_updates_per_ppo', default=5, type=int, help="Number of updates per PPO")
args = parser.parse_args()

BATCH_SIZE = 1

def parallel_and_device(model, device):
    model = model.to(device) # shift model to device
    return model

# Select the GPU with the lowest utilization
def get_available_device():
    gpus_id = getAvailable(order = 'memory', maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    if len(gpus_id) == 0:
        print("No available GPU, will proceed with CPU")
        return torch.device("cpu")
    else:
        print(f"GPU {gpus_id[0]} selected")
        return torch.device(f"cuda:{gpus_id[0]}")

device = get_available_device()

rover = ROVR(vid_length=args.vid_length, time_steps=args.time_steps, n_updates_per_ppo=args.n_updates_per_ppo)
rover = parallel_and_device(rover, device)

def load_dataset(root_folder, batch_size=1, num_workers=0, transform=None):
    dataset = VideoDataset(root_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle = True)

def calculate_preservation(org_values, computed_values):
    # Convert all to numpy arrays for easier computation
    org_values = np.array(org_values)
    computed_values = np.array(computed_values)
    
    # Avoid division by zero
    org_values = np.where(org_values == 0, np.finfo(float).eps, org_values)
    
    # Calculate the absolute percentage difference
    abs_percentage_diff = np.abs(computed_values - org_values) / org_values
    
    # Subtract from 1 to get the preservation value
    preservation_values = 1 - abs_percentage_diff

    return preservation_values.tolist()


ds = load_dataset("out/LQ", batch_size = BATCH_SIZE, num_workers = 32)

# rover.writer.add_graph(rover.train, (next(iter(ds))[0].to(device), next(iter(ds))[1].to(device), 0))

for i, batch in enumerate(ds):
    print("-----------------------ITERATION-----------------------", i)
    print(torch.cuda.memory_allocated(device)/(1024 ** 3))
    corrupted_frames, frames, masks = batch
    corrupted_frames = corrupted_frames.to(device)
    frames = frames.to(device)
    masks = masks.to(device)    
    print(torch.cuda.memory_allocated(device)/(1024 ** 3))
    
    optical_flow_by_frame, exp_optical_flow_by_frame, org_optical_flow_by_frame, corrupted_optical_flow_by_frame = rover.train(corrupted_frames, frames, i, device)
    cosine_similarities_rl = [cosine_similarity(org, rl) for org, rl in zip(org_optical_flow_by_frame, optical_flow_by_frame)]
    cosine_similarities_exp = [cosine_similarity(org, exp) for org, exp in zip(org_optical_flow_by_frame, exp_optical_flow_by_frame)]

    
    print("RL PRES", cosine_similarities_rl)
    print("RL NON PRES", cosine_similarities_exp)

    
    print("THIS IS OUR FLOW SHAPE", len(optical_flow))
    
    if i % 50 == 0:
        torch.save({
            'epoch': i,
            'model_state_dict': rover.state_dict(),
            'optimizers_state_dict': [rover.actor1_optimizer.state_dict(), rover.critic1_optimizer.state_dict(), rover.actor2_optimizer.state_dict(), rover.critic2_optimizer.state_dict(), rover.local_net_optimizer.state_dict()],
        }, rover.tensorboard_path / 'checkpoints' / f'{i}.pt')