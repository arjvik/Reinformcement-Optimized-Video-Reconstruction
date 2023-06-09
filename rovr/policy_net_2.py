import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding

from einops import rearrange
import math

class PolicyNetwork2UNet(nn.Module):
    def __init__(self, is_critic=False):
        super(PolicyNetwork2UNet, self).__init__()
        self.num_composed_frames = 20
        self.output_size = self.num_composed_frames
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
            nn.Linear(64, self.output_size)
        )
        
    def compute_logits(self, x, context, device = None):
        """
        x: (b, 2048)
        context: (b, 3, 256, 256)
        out: (b, 25)
        """
        vector_out = self.vector_fc(x)
        image_out = self.image_conv(context)
        stacked = torch.cat([vector_out, image_out], dim = 1)
        return self.final_fc(stacked)

    def forward(self, image, context, target, device = None):
        if not self.is_critic:
            logits = self.get_masked_logits(image, context, target, device)
            # print("TOPK BEFORE SOFTMAX:", torch.topk(logits, k=2, dim = 1).values)
            probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1)
            topk = torch.topk(probs, k=2, dim=1)
            # print(f"{topk.values=}, {topk.values.log()}")
            logprob = (topk.values.log().sum(1))/2 + 0.69314
            return topk.indices.detach(), logprob.detach()
        else:
            logits = self.compute_logits(image, context, device)
            return logits.squeeze(1) # not really logits
    
    def get_masked_logits(self, image, context, target, device = None):
        if self.is_critic:
            raise Exception("DO NOT CALL get_masked_logits FOR CRITIC")
            
        logits = self.compute_logits(image, context, device)

        logits.scatter_(1, target, 0)
        logits = (logits - logits.mean(dim = 1))/(logits.std(dim = (1, ), keepdim = True) + .1)

        return logits
        

    def logprob(self, image, context, target, action):
        if self.is_critic:
            raise Exception("DO NOT CALL LOGPROB FOR CRITIC")
        
        logits = self.compute_logits(image, context)
        logits.scatter_(1, target, 0)
        probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1)
        pairedprobs = rearrange(torch.matmul(probs.unsqueeze(2), probs.unsqueeze(1)), 'b i j -> b (i j)', i=self.output_size, j=self.output_size)
        action = action[:, 0]*self.output_size+action[:, 1]
        return (pairedprobs.gather(1, action.unsqueeze(1)).log().sum(1))/2 + 0.69314