class CustomLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, device):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTMCell(3, hidden_dim).to(device)
        self.fc = nn.Linear(hidden_dim, 1).to(device)
        
    def forward(self, frames):
        batch_size = frames.size(0)
        hx = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cx = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        # Normalizing frame values
        frames = frames.float() / 48

        outputs = []
        for i in range(frames.size(1)):  
            frame_i = frames[:, i]
            
            hx, cx = self.lstm(frame_i, (hx, cx))
            
            out = self.fc(hx)
            
            outputs.append(out.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
