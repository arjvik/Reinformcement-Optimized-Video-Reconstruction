import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class PolicyNetwork2(nn.Module):
    def __init__(self):
        super(PolicyNetwork2, self).__init__()
        self.num_composed_frames = 49
        self.image_size = int(math.sqrt(self.num_composed_frames) * 32)
        self.context_size = 512
        self.patch_size = 32
        self.num_channels = 3
        self.num_context = 1
        self.batch_size = 32
        self.num_heads = 16
        self.encoder_layers = 6
        self.dropout = 0.1

        self.num_image_patches = self.image_size // self.patch_size
        self.num_context_patches = self.context_size // self.patch_size

        self.image_positional_encoding = ImagePositionalEncoding(num_image_patches=self.num_image_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size)
        self.context_positional_encoding = ContextPositionalEncoding(num_context_patches=self.num_context_patches, patch_size=self.patch_size, num_channels=self.num_channels, batch_size=self.batch_size, num_context=self.num_context)

        self.context_encoder = nn.ModuleList(
            [EncoderBlock(self.num_context * self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        self.image_encoder = MultiHeadAttention(self.patch_size**2 * self.num_channels, self.num_heads)
        self.decoder = nn.ModuleList(
            [DecoderBlock(self.patch_size**2 * self.num_channels, self.num_heads, self.dropout) for _ in range(self.encoder_layers)]
        )
        #goes to -1 b/c the first frame is filler
        self.fc = nn.Linear(self.image_size**2 * self.num_channels, self.num_composed_frames-1)

    def forward(self, image, context, target):
        image = self.patchify_image(image)
        context = self.patchify_context(context)
        for layer in self.context_encoder:
            context = layer(context) 
        image = self.image_encoder(image, image, image)
        for layer in self.decoder:
            image = layer(image, context)
        image = rearrange(image, 'b p (h w c) -> b (h p w c)', b=self.batch_size, p=self.num_image_patches**2, h=self.patch_size, w=self.patch_size, c=self.num_channels)
        image = self.fc(image)
        image.scatter_(1, target, 0)
        image = F.softmax(image, dim=1)
        return torch.topk(image, 2, dim=1)
    
    def patchify_image(self, img):
        # input: b x 512 x 512 x 3
        # output: b x 256 x 3072
        patches = rearrange(img, 'b (h ph) (w pw) c -> b (h w) (ph pw c)', ph=self.patch_size, pw=self.patch_size)
        return self.image_positional_encoding(patches)
    
    def patchify_context(self, img):
        # input: b x 2 x 256 x 256 x 3
        # output: b x 2 64 x 6144
        patches = rearrange(img, 'b n (h ph) (w pw) c -> b n (h w) (ph pw c)', ph=self.patch_size, pw=self.patch_size)
        return self.context_positional_encoding(patches)
        
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
        positions = repeat(self.positional_encoder(torch.arange(self.num_image_patches**2).float().unsqueeze(1).to(device)), 'p s -> b p s', b=self.batch_size, p=self.num_image_patches**2)
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
        patch_positions = repeat(self.patch_positional_encoder(torch.arange(self.num_context_patches**2).float().unsqueeze(1).to(device)), 'p s -> b n p s', b=self.batch_size, n=self.num_context, p=self.num_context_patches**2)
        context_positions = repeat(self.context_positional_encoder(torch.arange(self.num_context).float().unsqueeze(1).to(device)), 'n s -> b n p s', b=self.batch_size, n=self.num_context, p=self.num_context_patches**2)
        return rearrange(x + patch_positions + context_positions, 'b n p s -> b p (n s)', b=self.batch_size, n=self.num_context, p=self.num_context_patches**2)

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(SelfAttentionBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x += self.dropout(self.attention(x, x, x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CrossAttentionBlock, self).__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        x = self.layer_norm(x)
        x += self.dropout(self.attention(x, encoder_output, encoder_output))
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
##########################################################################
# THE FOLLOWING IS FROM CHATGPT BECAUSE I CAN'T IMPLEMENT ATTENTION FAST #
##########################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        # self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y, z):
        batch_size = x.size(0)
        
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(z).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_probs, value)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # out = self.fc(attended_values)
        return attended_values
    


