from rovr import ROVR
from video_ds import VideoDataset
from local_net import LocalNetwork
from action_lstm import ActionLSTM
from policy_net_1 import PolicyNetwork1
from policy_net_2 import PolicyNetwork2
from resnet_extractor import ResnetFeatureExtractor

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parallel_and_device(model, device):
    model = model.to(device) # shift model to device
    return model

rover = parallel_and_device(ROVR(), device)

ds = VideoDataset("all/")

for i, batch in enumerate(ds):
    print("-----------------------ITERATION-----------------------", i)
    corrupted_frames, frames, masks = batch
    corrupted_frames = corrupted_frames.unsqueeze(0).cuda()
    frames = frames.unsqueeze(0).cuda()
    masks = masks.unsqueeze(0).cuda()
    
    rover.train(corrupted_frames, frames)
    print(rover.logger)