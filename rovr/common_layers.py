import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class ImagePositionalEncoding(nn.Module):
    # input: b x 256 x 3072
    # output: b x 256 x 3072
    def __init__(self, num_image_patches, patch_size, num_channels, batch_size):
        super(ImagePositionalEncoding, self).__init__()
        self.num_image_patches = num_image_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.batch_size = batch_size

        self.positional_encoder = nn.Linear(1, self.patch_size**2 * self.num_channels)
    
    def forward(self, x):
        positions = repeat(self.positional_encoder(torch.arange(self.num_image_patches**2)
                                                        .float()
                                                        .unsqueeze(1)
                                                        .to(next(self.positional_encoder.parameters()).device)),
                           'p s -> b p s', b=self.batch_size, p=self.num_image_patches**2)
        return x + positions

class ContextPositionalEncoding(nn.Module):
    # input: b x 2 x 64 x 3072
    # output: b x 64 x 6144
    def __init__(self, num_context_patches, patch_size, num_channels, batch_size, num_context):
        super(ContextPositionalEncoding, self).__init__()
        self.num_context_patches = num_context_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.num_context = num_context

        self.patch_positional_encoder = nn.Linear(1, self.patch_size**2 * self.num_channels)
        self.context_positional_encoder = nn.Linear(1, self.patch_size**2 * self.num_channels)
    
    def forward(self, x):
        patch_positions = repeat(self.patch_positional_encoder(torch.arange(self.num_context_patches**2)
                                                                    .float()
                                                                    .unsqueeze(1)
                                                                    .to(next(self.patch_positional_encoder.parameters()).device)),
                                 'p s -> b n p s', b=self.batch_size, n=self.num_context, p=self.num_context_patches**2)
        context_positions = repeat(self.context_positional_encoder(torch.arange(self.num_context)
                                                                        .float()
                                                                        .unsqueeze(1)
                                                                      .to(next(self.context_positional_encoder.parameters()).device)),
                                   'n s -> b n p s', b=self.batch_size, n=self.num_context, p=self.num_context_patches**2)
        return x + rearrange(patch_positions + context_positions, 'b n p s -> b (n p) s')

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(SelfAttentionBlock, self).__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x += self.attention(x, x, x)[0]
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CrossAttentionBlock, self).__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, encoder_output):
        x = self.layer_norm(x)
        x += self.attention(x, encoder_output, encoder_output)[0]
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(FeedForwardBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderBlock, self).__init__()

        self.attention = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.cross_attention = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(hidden_dim, dropout)

    def forward(self, x, encoder_output):
        x = x + self.attention(x)
        x = x + self.cross_attention(x, encoder_output)
        x = x + self.feed_forward(x)
        return x