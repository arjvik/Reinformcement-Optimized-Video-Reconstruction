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
        self.is_critic = is_critic
        if self.is_critic:
            self.output_size = 1
        else:
            self.output_size = self.num_composed_frames
        self.context_size = 256
        self.num_channels = 1

        self.temperature = .7
        self.num_resnet_features = 2048


        # Define the layers for processing the image
        self.context_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )

        # Define the layers for processing the b 1 160 160
        self.video_conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1)),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Flatten()
        )

        # Combine the outputs of the image and vector branches
        self.final_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, self.output_size)
        )
        
    def compute_logits(self, x, device = None):
        """
        x: (b, 2048)
        context: (b, 3, 256, 256)
        out: (b, 25)
        """
        
        print("INTO MODEL", x.shape)
        return self.final_fc(x)

    def forward(self, image, context, target, device = None, extra = None):
        
        print("IMAGE CONTEXT SHAPES", f"{image.shape=}, {context.shape=}")
        if self.is_critic:
            image = image.unsqueeze(1)
        vector_out = self.video_conv(image)
        # image_out = self.context_conv(context)
        image_out = context.squeeze(1)
        
        print(f"{vector_out.shape=}, {image_out.shape=}")
        stacked = torch.cat([vector_out, image_out], dim = 1)
        if extra is not None:
            return self.get_masked_logits(stacked, target, device)
        print(f"{stacked.shape=}, f{vector_out.shape=}, {image_out.shape=}")
        if not self.is_critic:
            logits = self.get_masked_logits(stacked, target, device)
            # print("TOPK BEFORE SOFTMAX:", torch.topk(logits, k=2, dim = 1).values)
            probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1)
            topk = torch.topk(probs, k=2, dim=1)
            # print(f"{topk.values=}, {topk.values.log()}")
            logprob = (topk.values.log().sum(1))/2 + 0.69314
            return topk.indices.detach(), logprob.detach()
        else:
            mean = stacked.mean(dim = 0, keepdim = True)
            std = stacked.std(dim = 0, keepdim = True)
            stacked = (stacked - mean)/(std +.001)
            logits = self.compute_logits(stacked, device)
            return logits.squeeze(1) # not really logits
    
    def get_masked_logits(self, stacked, target, device = None):
        if self.is_critic:
            raise Exception("DO NOT CALL get_masked_logits FOR CRITIC")
        
        print(f"{stacked.shape=}")
        logits = self.compute_logits(stacked, device)
        
        target = target.to(torch.int64).squeeze(1)
        
        print("PLS", target.shape, logits.shape)

        logits.scatter_(1, target, 0)
        logits = (logits - logits.mean(dim = 1))/(logits.std(dim = (1, ), keepdim = True) + .1)

        return logits
        

    def logprob(self, image, context, target, action, device):
        if self.is_critic:
            raise Exception("DO NOT CALL LOGPROB FOR CRITIC")
        image = image.unsqueeze(1)
        vector_out = self.video_conv(image)
        # image_out = self.context_conv(context)
        image_out = context.squeeze(1)
    
        print(f"{vector_out.shape=}, {image_out.shape=}")
        stacked = torch.cat([vector_out, image_out], dim = 1)
        logits = self.compute_logits(stacked).to(device)
        logits.scatter_(1, target.to(device), 0)
        probs = F.gumbel_softmax(logits, tau = self.temperature, hard = False, dim=1)
        pairedprobs = rearrange(torch.matmul(probs.unsqueeze(2), probs.unsqueeze(1)), 'b i j -> b (i j)', i=self.output_size, j=self.output_size)
        action = action[:, 0]*self.output_size+action[:, 1]
        return (pairedprobs.gather(1, action.unsqueeze(1)).log().sum(1))/2 + 0.69314