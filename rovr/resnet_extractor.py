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
            print("on gpu!")
        else:
            self.device = torch.device('cpu')
            print("on cpu")
        
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False  # freeze all the model parameters
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.preprocessing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def normalize(self, features):
        return (features / 0.5) - 1
            
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        grid_size = h // 32  # 512 // 32 = 16, so the grid will be 16x16

        feature_map = torch.zeros((batch_size, seq_len, 3, 224, 224)).to(self.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                for i in range(grid_size):
                    for j in range(grid_size):
                        # extract the patch from the original image
                        patch = x[b, s, :, i*32:(i+1)*32, j*32:(j+1)*32]
                        patch = self.preprocessing(patch.cpu()).unsqueeze(0).to(self.device)
                        feature = self.resnet(patch)
                        feature = feature.view(1, 3, 32, 32)
                        # place the feature back into the final feature map at the same location
                        feature_map[b, s, :, i*32:(i+1)*32, j*32:(j+1)*32] = feature

        return feature_map
