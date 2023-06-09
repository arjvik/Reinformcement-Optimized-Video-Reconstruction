import torch
import lpips
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from torchvision.models.optical_flow import raft_small
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import lpips

from einops import rearrange, repeat

from video_ds_explicit import VideoDatasetExplicit
from policy_net_2 import PolicyNetwork2UNet
from resnet_extractor import ResnetFeatureExtractor

import random
import time
from itertools import cycle
from pathlib import Path

from GPUtil import showUtilization as gpu_usage, getAvailable

def clamp(x, a, b): return x if a <= x <= b else (a if a > x else b)

def load_video_dataset(root_folder, num_workers):
    dataset = VideoDatasetExplicit(root_folder)
    return DataLoader(dataset, batch_size=None, num_workers=num_workers, shuffle = True)

ds = load_video_dataset("out/LQ", num_workers = 32)

pn2 = PolicyNetwork2UNet()
pn2_optimizer = torch.optim.Adam(pn2.parameters(), lr=1e-4)

video_encoder = ResnetFeatureExtractor(pretrained = True)

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
pn2 = parallel_and_device(pn2, device)
video_encoder = parallel_and_device(video_encoder, device)


mse_loss_fn = torch.nn.MSELoss().to(device)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

path = Path('runs') / 'warm_start' / 'pn2' / 'immitation_learning' / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
(path / 'checkpoints').mkdir(parents=True)

writer = SummaryWriter(log_dir=path, flush_secs=10)

for epoch, (video, _, masks, positive, negative) in enumerate(tqdm(cycle(ds))):
    encoded_frames = video_encoder.resnet(torch.stack([video_encoder.preprocessing(f) for f in video], dim=0).to(device)).squeeze(2).squeeze(2)
    outputs = pn2.get_masked_logits(encoded_frames, video.to(device), torch.arange(20).unsqueeze(1).to(device)).sigmoid()
    loss = torch.tensor(0).float().to(device)
    for i in range(positive.shape[1]):
        ans = torch.nn.functional.one_hot(positive[:, i].to(torch.int64), num_classes=20).sum(dim=1)
        loss += torch.nn.functional.binary_cross_entropy(outputs, ans.float().to(device))
    for i in range(negative.shape[1]):
        ans = torch.nn.functional.one_hot(negative[:, i].to(torch.int64), num_classes=20).sum(dim=1)
        loss -= torch.nn.functional.binary_cross_entropy(outputs, ans.float().to(device))
    writer.add_scalar('Loss/expert_loss', loss.detach(), epoch)
    pn2_optimizer.zero_grad()
    loss.backward()
    pn2_optimizer.step()
    if (epoch % 3000) == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': pn2.state_dict(),
            'optimizer_state_dict': pn2_optimizer.state_dict(),
            'loss': loss.detach(),
        }, path / 'checkpoints' / f'{epoch}.pt')