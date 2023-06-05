import torch
import torch.nn as nn
import torch.nn.functional as F

from rovr.common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding, ContextPositionalEncoding

from einops import rearrange
import math

class PolicyNetwork2(nn.Module):
    def __init__(self, is_critic=False):
        super(PolicyNetwork2, self).__init__()
        self.num_composed_frames = 25
        self.output_size = self.num_composed_frames - 1
        self.context_size = 256
        self.patch_size = 16
        self.image_size = int(math.sqrt(self.num_composed_frames) * self.patch_size)
        self.num_channels = 3
        self.batch_size = 32
        self.num_heads = 4
        self.encoder_layers = 3
        self.decoder_layers = 6
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
            [DecoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.decoder_layers)]
        )
        if not self.is_critic:
            self.fc = nn.Linear(self.image_size**2 * self.num_channels, self.output_size)
        else:
            self.fc = nn.Linear(self.image_size**2 * self.num_channels, 1)

    def compute_logits(self, image, context, target):
        image = self.patchify_image(image)
        context = self.patchify_context(context)
        for layer in self.context_encoder:
            context = layer(context)
        for layer in self.decoder:
            image = layer(image, context)
        image = rearrange(image, 'b (hp wp) (c ph pw) -> b (c hp ph wp pw)', b=self.batch_size, hp=self.num_image_patches, wp=self.num_image_patches, ph=self.patch_size, pw=self.patch_size, c=self.num_channels)
        logits = self.fc(image)
        return logits

    def forward(self, image, context, target):
        if not self.is_critic:
            logits = self.compute(image, context, target)
            logits.scatter_(1, target, 0)
            probs = F.softmax(image, dim=1)
            topk = torch.topk(probs, k=2, dim=1)
            logprob = topk.values.log().sum(1) + torch.log(2)
            return topk.indices, logprob
        else:
            return logits.squeeze(1) # not really logits

    def logprob(self, image, context, target, action):
        if self.is_critic:
            raise Exception("DO NOT CALL LOGPROB FOR CRITIC")
        logits = self.compute(image, context, target)
        logits.scatter_(1, target, 0)
        probs = F.softmax(image, dim=1)
        pairedprobs = torch.matmul(probs.unsqueeze(2), probs.unsqueeze(1)).reshape(self.batch_size, -1)
        action = action[:, 0]*self.output_classification_head_size+action[:, 1]
        return pairedprobs.gather(1, action.unsqueeze(1)).log().sum(1) + torch.log(2)
    
    def patchify_image(self, img):
        patches = rearrange(img, 'b c (hp ph) (wp pw) -> b (hp wp) (c ph pw)', ph=self.patch_size, pw=self.patch_size)
        return self.image_positional_encoding(patches)
    
    def patchify_context(self, img):
        patches = rearrange(img, 'b c (hp ph) (wp pw) -> b (hp wp) (c ph pw)', ph=self.patch_size, pw=self.patch_size)
        return self.context_positional_encoding(patches)