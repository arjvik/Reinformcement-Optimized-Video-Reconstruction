import torch
import torch.nn as nn
class ActionLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, batch_size, device):
        super(ActionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTMCell(3, hidden_dim).to(device)
        self.fc = nn.Linear(hidden_dim, 3072).to(device) 

        self.hx = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        self.cx = torch.zeros(batch_size, self.hidden_dim).to(self.device)

    def forward(self, action):
        action = action.float() / 48
        
        print(action.shape)

        self.hx, self.cx = self.lstm(action, (self.hx, self.cx))

        out = self.fc(self.hx)
        out = out.view(-1, 32, 32, 3)  
            
        return out

    def reset_hidden_states(self):
        self.hx = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
        self.cx = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
