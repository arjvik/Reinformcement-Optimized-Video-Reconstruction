import torch
import torch.nn as nn

from einops import rearrange

class LocalNetworkUNet(nn.Module):
    def __init__(self):
        super(LocalNetworkUNet, self).__init__()

        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Expanding path (decoder)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.conv8 = nn.Conv2d(64, 3, kernel_size=1)
        
    def forward(self, x, context):
        
        inp = torch.cat([x.unsqueeze(1), context], dim=1)
        x = rearrange(inp, 'b n c h w -> b (n c) h w')
        
        # Contracting path (encoder)
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(self.maxpool(x1)))
        x3 = nn.functional.relu(self.conv3(self.maxpool(x2)))
        x4 = nn.functional.relu(self.conv4(self.maxpool(x3)))
        
        # Expanding path (decoder)
        x = nn.functional.relu(self.upconv1(x4))
        x = torch.cat([x, x3], dim=1)
        x = nn.functional.relu(self.conv5(x))
        
        x = nn.functional.relu(self.upconv2(x))
        x = torch.cat([x, x2], dim=1)
        x = nn.functional.relu(self.conv6(x))
        
        x = nn.functional.relu(self.upconv3(x))
        x = torch.cat([x, x1], dim=1)
        x = nn.functional.relu(self.conv7(x))
        
        # Output layer
        x = self.conv8(x)
        return x