import torch
import torchvision.models as models
import torchvision.transforms as transforms


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.linear = torch.nn.Linear(2048, 32*32*3) # learnable linear layer to project features
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.resnet = self.resnet.to(self.device)
            self.linear = self.linear.to(self.device)
            print("on gpu!")
        else:
            self.device = torch.device('cpu')
            print("on cpu")
        
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False  # freeze all the model parameters
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) # remove FC layer
        
        self.preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        feature_map = torch.zeros((batch_size, seq_len, 32, 32, 3)).to(self.device)

        for b in range(batch_size):
            for s in range(seq_len):
                frame = self.preprocessing(x[b, s].cpu()).unsqueeze(0).to(self.device)
                feature = self.resnet(frame)
                
                # pass the 2048-dim vector through the linear layer to project it to 32x32x3
                feature = self.linear(feature.view(-1)).view(32, 32, 3)
                
                feature_map[b, s] = feature

        return feature_map
