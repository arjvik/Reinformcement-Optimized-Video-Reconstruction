from rovr import ROVR
from video_ds import VideoDataset
from local_net import LocalNetwork
from action_lstm import ActionLSTM
from policy_net_1 import PolicyNetwork1
from policy_net_2 import PolicyNetwork2
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

ds = load_dataset("out/LQ", batch_size = BATCH_SIZE, num_workers = 32)


logger = Logger()

for i, batch in enumerate(ds):
    print("-----------------------ITERATION-----------------------", i)
    corrupted_frames, frames, masks = batch
    corrupted_frames = corrupted_frames.to(device)
    frames = frames.to(device)
    masks = masks.to(device)    
    
    rover.train(corrupted_frames, frames, i)
    
    logger.log(rover.logger)
    
logger.close()