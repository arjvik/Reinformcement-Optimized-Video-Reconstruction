import torch
import torchvision.models as models
import torchvision.transforms as transforms

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.linear = torch.nn.Linear(2048, 16*16*3) # learnable linear layer to project features

        if pretrained:
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
        local_device = x.device
        batch_size, seq_len, c, h, w = x.size()
        feature_map = torch.zeros((batch_size, 3, 80, 80)).to(local_device)  # change to channel first

        for b in range(batch_size):
            for s in range(seq_len):
                feature = self.encode(x[b, s])
                idx = self.calculate_index(s)
                feature_map[b, :, idx[0]:idx[0]+16, idx[1]:idx[1]+16] = feature  # change to channel first

        return feature_map

    def calculate_index(self, idx):
        # define your mapping here
        return (idx // 5 * 16, idx % 5 * 16)

    def encode(self, x):
        local_device = x.device
        x = self.preprocessing(x).unsqueeze(0).to(local_device)
        feature = self.resnet(x)
        feature = self.linear(feature.view(-1)).view(3, 16, 16)  # change to channel first
        return feature

    def insert_encoded_frame_batch(self, indices, full_frame_batch, encoded_frame_batch):
        
        for b in range(full_frame_batch.size(0)):  # Iterate over the batch dimension
            encoded_frame = self.encode(full_frame_batch[b])  # Encode the full frame into a 3x16x16 representation
            idx = self.calculate_index(indices[b])  # Calculate the index for each batch
            encoded_frame_batch[b, :, idx[0]:idx[0]+16, idx[1]:idx[1]+16] = encoded_frame  # Insert it into the encoded_frame_batch at the correct spot
        return encoded_frame_batch



    def extract_patch(self, indices, feature_map):
        local_device = feature_map.device
        patches = []
        for b, batch_indices in enumerate(indices):
            batch_patches = []
            for idx in batch_indices:
                i, j = self.calculate_index(idx)
                patch = feature_map[b, :, i:i+16, j:j+16].unsqueeze(0)  # change to channel first
                batch_patches.append(patch)
            patches.append(torch.cat(batch_patches, 0))
        return torch.stack(patches).to(local_device)
