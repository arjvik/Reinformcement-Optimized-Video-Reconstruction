import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding, ContextPositionalEncoding

from einops import rearrange
import math

class PolicyNetwork1UNet(nn.Module):
    def __init__(self, is_critic=False):
        super(PolicyNetwork1UNet, self).__init__()
        self.num_composed_frames = 25
        self.image_size = int(math.sqrt(self.num_composed_frames) * 16)
        self.num_channels = 3
        self.is_critic = is_critic
        self.temperature = .5
        self.image_size = 80
        
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Expanding path (decoder)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 3, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(3)
        self.conv9 = nn.Conv2d(3, 1, kernel_size=1)
        self.bn9 = nn.BatchNorm2d(1)
        self.dropout = 0.1

        if not self.is_critic:
            self.fc_final = nn.Linear(400, 25)

        else:
            self.fc_final = nn.Linear(400, 1)


    def unet(self, x):
        x1 = nn.functional.relu(self.bn1(self.conv1(x)))
        x2 = nn.functional.relu(self.bn2(self.conv2(self.maxpool(x1))))
        x3 = nn.functional.relu(self.bn3(self.conv3(self.maxpool(x2))))
        x4 = nn.functional.relu(self.bn4(self.conv4(self.maxpool(x3))))
        

        # Expanding path (decoder)
        x = nn.functional.relu(self.bn_up1(self.upconv1(x4)))
        x = torch.cat([x, x3], dim=1)
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        
        x = nn.functional.relu(self.bn_up2(self.upconv2(x)))
        x = torch.cat([x, x2], dim=1)
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        
        x = nn.functional.relu(self.bn_up3(self.upconv3(x)))
        x = torch.cat([x, x1], dim=1)
        x = nn.functional.relu(self.bn7(self.conv7(x)))

        x = nn.functional.relu(self.bn8(self.conv8(x)))
        x = nn.functional.relu(self.bn9(self.conv9(self.maxpool(x))))
        x = self.maxpool(x)
        
        return x

    def compute_logits(self, x, context):
        
        inp = torch.cat([x, context], dim=1)
        image = self.unet(inp)
        image = rearrange(image, 'b c h w -> b (c h w)')
        mean = image.mean(dim=1, keepdim=True)
        std = image.std(dim=1, keepdim=True)
        normalized_image = (image - mean) / std
        return self.fc_final(normalized_image)

    def forward(self, image, context, device = None):
        logits = self.compute_logits(image, context)
        
        if not self.is_critic:
            logits = (logits - logits.mean(dim = 1))/(logits.std(dim = (1, ), keepdim = True) + .1)
            probs = F.gumbel_softmax(logits, tau = self.temperature, hard= False, dim = 1)
            maxes = probs.max(dim=1)
            return maxes.indices.detach(), maxes.values.log().detach()
        else:
            return logits.squeeze(1)
    
    def logprob(self, image, context, action):
        if self.is_critic:
            raise Exception("DO NOT CALL LOGPROB FOR CRITIC")

        
        logits = self.compute_logits(image, context)
        probs = F.gumbel_softmax(logits, tau = self.temperature, hard= False, dim = 1)
        return probs.gather(1, action.unsqueeze(1)).log().squeeze(1)

    

