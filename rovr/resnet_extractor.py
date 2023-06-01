import torch
import torchvision.models as models
import torchvision.transforms as transforms

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.resnet = self.resnet.to(self.device)
            print("Model loaded on GPU.")
        else:
            self.device = torch.device('cpu')
            print("GPU not available. Model loaded on CPU.")
        
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze all the model parameters
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def normalize(self, features):
        return (features / 0.5) - 1
            
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        features = []
        for b in range(batch_size):
            for s in range(seq_len):
                img = x[b, s]
                img = self.preprocessing(img.cpu()).unsqueeze(0)
                img = img.to(self.device)
                feature = self.resnet(img)
                feature = feature.view(feature.size(0), -1)
                features.append(feature)
        return torch.stack(features)
