import torch.nn as nn

from rovr.common_layers import EncoderBlock, DecoderBlock, ImagePositionalEncoding, ContextPositionalEncoding

from einops import rearrange

class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.image_size = 512
        self.context_size = 256
        self.patch_size = 32
        self.num_channels = 3
        self.num_context = 2
        self.batch_size = 32
        self.num_heads = 16
        self.encoder_layers = 6
        self.dropout = 0.1

        self.num_image_patches = self.image_size // self.patch_size
        self.num_context_patches = self.context_size // self.patch_size

        self.image_positional_encoding = ImagePositionalEncoding(num_image_patches=self.num_image_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size)
        self.context_positional_encoding = ContextPositionalEncoding(num_context_patches=self.num_context_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size, num_context=self.num_context)

        self.context_encoder = nn.ModuleList(
            [EncoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        self.fc = nn.Linear(self.image_size**2 * self.num_channels, self.image_size**2 * self.num_channels)

    def forward(self, image, context):
        image = self.patchify_image(image)
        context = self.patchify_context(context)
        for layer in self.context_encoder:
            context = layer(context)
        for layer in self.decoder:
            image = layer(image, context)
        image = rearrange(image, 'b p (ph pw c) -> b (p ph pw c)', b=self.batch_size, p=self.num_image_patches**2, ph=self.patch_size, pw=self.patch_size, c=self.num_channels)
        image = self.fc(image)
        return rearrange(image, 'b (h w c) -> b h w c', b=self.batch_size, h=self.image_size, w=self.image_size)
    
    def patchify_image(self, img):
        # input: b x 512 x 512 x 3
        # output: b x 256 x 3072
        patches = rearrange(img, 'b (hp ph) (wp pw) c -> b (hp wp) (ph pw c)', ph=self.patch_size, pw=self.patch_size)
        return self.image_positional_encoding(patches)
    
    def patchify_context(self, img):
        # input: b x 2 x 256 x 256 x 3
        # output: b x 128 x 3072
        patches = rearrange(img, 'b n (hp ph) (wp pw) c -> b (n hp wp) (ph pw c)', ph=self.patch_size, pw=self.patch_size)
        return self.context_positional_encoding(patches)