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

from video_ds_2 import VideoDataset2
from local_net_unet_norm import LocalNetworkUNetNorm ##CHANGED FROM VIT VERSION
from action_lstm import ActionLSTM
from policy_net_1 import PolicyNetwork1
from policy_net_2 import PolicyNetwork2
from resnet_extractor import ResnetFeatureExtractor

import random
import time
from itertools import cycle
from pathlib import Path

from GPUtil import showUtilization as gpu_usage, getAvailable

def clamp(x, a, b): return x if a <= x <= b else (a if a > x else b)

class ImageDataset(Dataset):
    def __init__(self, video, orig_video):
        self.video = video
        self.orig_video = orig_video

    def __len__(self):
        return 500
    
    def __getitem__(self, idx):
        l = idx % video.shape[0]
        
        # #pick two distinct numbers between 5 and 10
        # f = random.randint(5, 19)
        # n = clamp(random.randint(5, 10) * (2 * random.randint(0, 1) - 1), 0, 24)
        # m = n
        # while n == m:
        #     m = clamp(random.randint(5, 10) * (2 * random.randint(0, 1) - 1), 0, 24)
        
        f = random.randint(2, 24)
        m = f - 1
        n = m - 1
        
        
        image = self.video[l][f]
        context1 = self.video[l][n]
        context2 = self.video[l][m]
        target = self.orig_video[l][m]
        ## CHANGED FROM VIT
        #context1 = F.interpolate(context1.unsqueeze(0), size = (128, 128), mode = "bilinear", align_corners = False).squeeze(0)
        #context2 = F.interpolate(context2.unsqueeze(0), size = (128, 128), mode = "bilinear", align_corners = False).squeeze(0)
        
        return image, context1, context2, target

def load_video_dataset(root_folder, num_workers):
    dataset = VideoDataset2(root_folder, difficulty=1)
    return DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle = True)

def load_image_dataset(video, org_video, batch_size=32):
    dataset = ImageDataset(video, org_video)
    return DataLoader(dataset, batch_size=batch_size, shuffle = True)

l = list(tqdm(load_video_dataset("out/LQ", num_workers = 32)))
video, orig_video = torch.cat([t[0] for t in l]), torch.cat([t[1] for t in l])

local_net = LocalNetworkUNetNorm()
local_net_optimizer = torch.optim.Adam(local_net.parameters(), lr=1e-4)

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
local_net = parallel_and_device(local_net, device)

mse_loss_fn = torch.nn.MSELoss().to(device)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

image_ds = load_image_dataset(video, orig_video, batch_size=24)

path = Path('runs') / 'local_net_baseline' / 'unet_lpips_norm' / 'gamma:.9993_difficulty:1' / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
(path / 'checkpoints').mkdir(parents=True)

writer = SummaryWriter(log_dir=path, flush_secs=10)

writer.add_graph(local_net, (next(iter(image_ds))[0].to(device), torch.stack(next(iter(image_ds))[1:3], dim=1).to(device)))

for i, (frame, context1, context2, target) in tqdm(enumerate(cycle(image_ds))):
    local_net_optimizer.zero_grad()
    image = frame.to(device), torch.stack([context1, context2], dim = 1).to(device)
    print(frame.shape, image[1].shape)
    y_hat = local_net(*image)
    cuda_target = target.to(device)
    mse_loss = mse_loss_fn(y_hat, cuda_target).mean()
    writer.add_scalar('Loss/mse_loss', mse_loss.detach(), i)
    lpips_loss = lpips_loss_fn(y_hat, cuda_target).mean()
    writer.add_scalar('Loss/lpips_loss', lpips_loss.detach(), i)
    gamma = 0.1 + 0.9 * (0.9993 **  i)
    writer.add_scalar('Loss/gamma', gamma, i)
    total_loss = mse_loss * gamma + lpips_loss * (1-gamma)
    writer.add_scalar('Loss/total_loss', total_loss.detach(), i)
    total_loss.backward()
    local_net_optimizer.step()
    if (i % 200) == 0:
        display = torch.cat(list(torch.cat([frame, context1, context2, target, y_hat.cpu()], dim=3)[:10]), dim=1)
        writer.add_image('Viz', display, i)
    if (i % 2000) == 0:
        torch.save({
            'epoch': i,
            'model_state_dict': local_net.state_dict(),
            'optimizer_state_dict': local_net_optimizer.state_dict(),
            'mse_loss': mse_loss.detach(),
            'lpips_loss': lpips_loss.detach(),
            }, path / 'checkpoints' / f'{i}.pt')
        