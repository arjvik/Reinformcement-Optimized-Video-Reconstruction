import torch
import torch.nn as nn

from einops import rearrange

class ActionLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, batch_size):
        super(ActionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTMCell(3 + 3 * 16 * 16 * 3, hidden_dim) # adjust the input size
        self.fc = nn.Linear(hidden_dim, 80 * 80 * 3)

        self.hx = torch.zeros(batch_size, self.hidden_dim)
        self.cx = torch.zeros(batch_size, self.hidden_dim)

    def forward(self, action, new_tensor):
        
        
        local_device = action.device
        
        self.hx = self.hx.to(local_device)
        self.cx = self.cx.to(local_device)
        
                
        action = action.float() / 48
        new_tensor = rearrange(new_tensor, 'b n c ph pw -> b (n c ph pw)') # flatten the new tensor

        input_tensor = torch.cat([action, new_tensor], dim=1)  # concatenate action and new_tensor along the second dimension

        self.hx, self.cx = self.lstm(input_tensor, (self.hx, self.cx))

        out = self.fc(self.hx)
        out = rearrange(out, 'b (c ph pw) -> b c ph pw', ph=80, pw=80) 
            
        return out

    def reset_hidden_states(self):
        self.hx = torch.zeros(self.batch_size, self.hidden_dim)
        self.cx = torch.zeros(self.batch_size, self.hidden_dim)
