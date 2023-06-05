import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding, ContextPositionalEncoding

from einops import rearrange
import math

class PolicyNetwork1(nn.Module):
    def __init__(self, is_critic=False):
        super(PolicyNetwork1, self).__init__()
        self.num_composed_frames = 25
        self.image_size = int(math.sqrt(self.num_composed_frames) * 16)
        self.context_size = 80
        self.patch_size = 16
        self.num_channels = 3
        self.batch_size = 1
        self.num_heads = 16
        self.encoder_layers = 6
        self.dropout = 0.1

        self.num_image_patches = self.image_size // self.patch_size
        self.num_context_patches = self.context_size // self.patch_size

        self.is_critic = is_critic

        self.image_positional_encoding = ImagePositionalEncoding(num_image_patches=self.num_image_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size)
        self.context_positional_encoding = ImagePositionalEncoding(num_image_patches=self.num_context_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size)

        self.context_encoder = nn.ModuleList(
            [EncoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        if not self.is_critic:
            self.fc = nn.Linear(self.image_size**2 * self.num_channels, self.num_composed_frames)
        else:
            self.fc = nn.Linear(self.image_size**2 * self.num_channels, 1)

    def forward(self, image, context):
        image = self.patchify_image(image)
        context = self.patchify_context(context)
        for layer in self.context_encoder:
            context = layer(context)
        for layer in self.decoder:
            image = layer(image, context)
        image = rearrange(image, 'b (hp wp) (c ph pw) -> b (c hp ph wp pw)', b=self.batch_size, hp=self.num_image_patches, wp=self.num_image_patches, ph=self.patch_size, pw=self.patch_size, c=self.num_channels)
        if not self.is_critic:
            probs = F.softmax(self.fc(image), dim=1)
            return torch.argmax(probs, dim=1)
        else:
            score = self.fc(image)
            return score.squeeze(1)
    
    def patchify_image(self, img):
        patches = rearrange(img, 'b c (hp ph) (wp pw) -> b (hp wp) (c ph pw)', ph=self.patch_size, pw=self.patch_size)
        return self.image_positional_encoding(patches)
    
    def patchify_context(self, img):
        patches = rearrange(img, 'b c (hp ph) (wp pw) -> b (hp wp) (c ph pw)', ph=self.patch_size, pw=self.patch_size)
        return self.context_positional_encoding(patches)
    
