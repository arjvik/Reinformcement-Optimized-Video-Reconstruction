import torch
import torch.nn as nn
from UNetBackbone import UNetBackbone

class CustomLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes, device):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.lstm = nn.LSTMCell(num_classes, hidden_dim).to(device)

        self.conv_video_frame = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        ).to(device)
        
        self.fc_encoded_video = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU()
        ).to(device)

        self.fc_action = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU()
        ).to(device)

        self.fc = nn.Linear(hidden_dim + 128 + 896, num_classes).to(device)
        
        self.softmax = nn.Softmax(dim=1).to(device)
        
        self.backbone = UNetBackbone(in_channels = 3, out_channels = 3, context_vector_dim = 3).to(device)
        
    def forward(self, encoded_video, video_frame, action):
        
        batch_size = video_frame.size(0)
        hx = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cx = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        encoded_video = encoded_video.to(self.device)
        video_frame = video_frame.to(self.device)
        action = action.to(self.device)
        
        encoded_video = self.fc_encoded_video(encoded_video.squeeze(-2))
                
                
        outputs = []
        for i in range(video_frame.size(1)):  # Iterate over sequence length
            
            action = self.fc_action(action)

            video_frame_i = self.conv_video_frame(video_frame[:, i])
            
                        
                
            print(encoded_video.shape, action.shape, video_frame_i.shape)
            
            x = torch.cat([video_frame_i, encoded_video.view(1, -1), action], dim=1)
            
                        
            out = self.fc(x)
            out = self.softmax(out)
            
            top_probs, top_indices = torch.topk(out, k=3, dim=1)
            
            print(encoded_video.shape, top_indices)
            
            context_package = encoded_video[:, top_indices, :]

            action = out.to(self.device)
            

            hx, cx = self.lstm(action, (hx, cx))
            
            outputs.append(out.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
