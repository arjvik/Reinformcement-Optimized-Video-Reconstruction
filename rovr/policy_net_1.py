import torch
import torch.nn as nn
from torchvision.models import vit_b_32

class PolicyNetwork1(nn.Module):
    def __init__(self, num_frames=49, patch_size=32, param_mode = 'reset', modifications = None):
        super(VideoTransformer, self).__init__()

        self.vit_model = vit_b_32(pretrained=True)     
        self.modify_params(param_mode, modifications)

        self.patch_size = patch_size
        self.embed_dim = 768

        self.vit_model.heads.head = nn.Linear(self.embed_dim, num_frames) 
        
        
    def modify_params(self, param_mode, modifications):
        if modifications is None:
            return
        
        blocks, reset_head, reset_conv_proj = modifications
        
        encoder_blocks = self.vit_model.encoder.layers
        head = self.vit_model.heads.head
        conv_proj = self.vit_model.conv_proj
        
        for i, layer in enumerate(encoder_blocks):
            if i in blocks:
                if param_mode == 'freeze':
                    for param in layer.parameters():
                        param.requires_grad = False
                elif param_mode == 'reset':
                    for param in layer.parameters():
                        if param.dim() > 1: # exclude bias terms
                            nn.init.kaiming_normal_(param)
        
        if reset_head:
            for param in head.parameters():
                if param_mode == 'freeze':
                    param.requires_grad = False
                elif param_mode == 'reset':
                    if param.dim() > 1: # exclude bias terms
                        nn.init.kaiming_normal_(param)
                        
        if reset_conv_proj:
            for param in conv_proj.parameters():
                if param_mode == 'freeze':
                    param.requires_grad = False
                elif param_mode == 'reset':
                    if param.dim() > 1: # exclude bias terms
                        nn.init.kaiming_normal_(param)               


    def forward(self, x, lstm_token):
        
        x[:, :, 0:32, 0:32] = lstm_token #kick out first slot for the lstm token
        
        logits = self.vit_model(x)
                                
        return logits
