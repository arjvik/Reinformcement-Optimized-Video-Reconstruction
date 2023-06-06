import torch
import lpips
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from torchvision.models.optical_flow import raft_small
from tqdm import tqdm

from video_ds import VideoDataset
from local_net import LocalNetwork
from action_lstm import ActionLSTM
from policy_net_1 import PolicyNetwork1
from policy_net_2 import PolicyNetwork2
from resnet_extractor import ResnetFeatureExtractor

from torch.utils.data import Dataset, DataLoader
import random

class ImageDataset(Dataset):
    def __init__(self, video, org_video):
        self.video = video.squeeze(0)
        self.num_frames = len(video)
        self.org_video = org_video.squeeze(0)

    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        #pick two distinct numbers between 5 and 10
        n = random.randint(5, 10) * (2 * random.randint(0, 1) - 1)
        m = n
        while n == m:
            m = random.randint(5, 10) * (2 * random.randint(0, 1) - 1)
        l = idx//25
        n = l + (idx+n)%25
        m = l + (idx+m)%25
        return self.video[idx], self.video(n), self.video(m), self.org_video[idx]
    
def load_video_dataset(root_folder, batch_size=1, num_workers=0):
    dataset = VideoDataset(root_folder)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle = True)

def load_image_dataset(video, org_video, batch_size=32, num_workers=0):
    dataset = ImageDataset(video, org_video)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle = True)

local_net = LocalNetwork()
local_net.cuda()
local_net_optimizer = torch.optim.Adam(local_net.parameters(), lr=1e-4)
big_video = []
big_org_video = []
ds = load_video_dataset("out/LQ", batch_size = 1, num_workers = 32)
for i, batch in enumerate(ds):
    video, org_video, _ = batch
    video = video.cuda()
    org_video = org_video.cuda()
    # image_ds = load_image_dataset(video, org_video, batch_size = 4, num_workers = 32)
    big_video.append(video)
    big_org_video.append(org_video)
image_ds = load_image_dataset(torch.cat(big_video, dim = 0), torch.cat(big_org_video, dim = 0), batch_size = 32, num_workers = 32)
for j, image_batch in enumerate(image_ds):
    target, c1, c2, org = image_batch
    local_net_optimizer.zero_grad()
    image = torch.stack([target, c1, c2], dim = 1).cuda()
    y_hat = local_net(image)
    loss = F.mse_loss(y_hat, org)
    loss.backward()
    local_net_optimizer.step()
    print(loss.item())






