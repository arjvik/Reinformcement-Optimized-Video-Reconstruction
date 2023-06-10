import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding

from einops import rearrange
import math

class PolicyNetwork2UNet(nn.Module):
    def __init__(self, is_critic=False):
        super(PolicyNetwork2UNet, self).__init__()
        self.num_frames = 20
        self.k = 2
        self.output_size = math.comb(self.num_frames, self.k)
        self.context_size = 256
        self.num_channels = 3
        self.is_critic = is_critic
        self.temperature = .7
        self.num_resnet_features = 2048

        if is_critic:
            raise Exception("NOT IMPLEMENTED YET! ADD NORMALIZATION LAYERS TO CRITIC")

        # Define the layers for processing the image
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )

        # Define the layers for processing the 2048-dimensional vector
        self.vector_fc = nn.Sequential(
            nn.Linear(self.num_resnet_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        # Combine the outputs of the image and vector branches
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, self.output_size if not self.is_critic else 1)
        )
        
        self.idxmap = torch.zeros((self.output_size, self.num_frames))
        self.idxmask = torch.ones((self.num_frames, self.output_size))
        for i, (a, b) in enumerate((a,b) for a in range(self.num_frames) for b in range(a+1, self.num_frames)):
            self.idxmap[i][a] = 1
            self.idxmap[i][b] = 1
            self.idxmask[a][i] = 0
            self.idxmask[b][i] = 0
        
        
    def indices_to_index(self, a, b, device = None):
        """
        Takes two values 0 <= a < b < self.num_frames and outputs a combo index 0 <= idx < nCr(self.num_frames, self.k)
        Clever math trick - derivation thrown away
        """
        return a*(2*self.num_frames-a-3)//2+b-1
    def index_to_indices(self, i, device):
        """
        Takes combo index and gives two-vector of frame indices
        """
        #return self.idxmap.to(device)[i].argwhere().squeeze(1)
        return torch.topk(self.idxmap.to(device)[i], k=self.k).indices
    
    def compute_logits(self, x, context, device):
        """
        x: (b, 2048)
        context: (b, 3, 256, 256)
        out: (b, 25)
        """
        vector_out = self.vector_fc(x)
        image_out = self.image_conv(context)
        stacked = torch.cat([vector_out, image_out], dim = 1)
        return self.final_fc(stacked)
    
    def forward(self, image, context, target, device):
        logits = self.compute_logits(image, context, device)
        if self.is_critic:
            return logits.squeeze(1)
        logits = logits * self.idxmask.to(device)[target.squeeze(1)]
        probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1).max(dim=1)
        return self.index_to_indices(probs.indices, device=device), probs.values
    
    def logprob(self, image, context, target, action, device):
        if self.is_critic:
            raise Exception("DO NOT CALL LOGPROB FOR CRITIC")
        logits = self.compute_logits(image, context, device)
        logits = logits * self.idxmask.to(device)[target.squeeze(1)]
        logprobs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1).log()
        return logprobs.gather(1, self.indices_to_index(action[:, 0], action[:, 1], device).unsqueeze(1)).squeeze(1)
    
    def get_raw_probs(self, image, context, target, device):
        if self.is_critic:
            raise Exception("DO NOT CALL GETRAWPROBS FOR CRITIC")
        logits = self.compute_logits(image, context, device)
        logits = logits * self.idxmask.to(device)[target.squeeze(1)]
        probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1)
        return probs
        
                    