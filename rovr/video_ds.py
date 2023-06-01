import cv2
import os
import numpy as np
import torch
from torchvision.transforms import functional as F

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.videos = os.listdir(folder)
    
    def corrupt_frame(self, frame):
        # Create a random mask
        mask = np.ones_like(frame)
        h, w, _ = frame.shape
        mask_x = np.random.randint(0, w-50)
        mask_y = np.random.randint(0, h-50)
        mask[mask_y:mask_y+50, mask_x:mask_x+50, :] = 0
        
        # Apply mask to frame
        corrupted_frame = frame * mask
        return corrupted_frame, mask
    
    def __len__(self):
        return len(self.videos)
    
    def normalize(self, frame):
        #Normalize between -1 and 1
        return (frame / 0.5) - 1
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.folder, self.videos[idx])
        vidcap = cv2.VideoCapture(video_path)
        
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        sampling_rate = int(np.ceil(fps))
        
        frames = []
        corrupted_frames = []
        masks = []
        i = 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break

            if i % sampling_rate == 0:
                # Resize frame
                frame = cv2.resize(frame, (480, 270))
                
                # Corrupt frame
                corrupted_frame, mask = self.corrupt_frame(frame)
                
                if self.transform is not None:
                    frame = self.transform(frame)
                    corrupted_frame = self.transform(corrupted_frame)
                    mask = self.transform(mask)
                
                frames.append(self.normalize(frame))
                corrupted_frames.append(self.normalize(corrupted_frame))
                masks.append(mask)
            i += 1
        
        return torch.stack(corrupted_frames), torch.stack(frames), torch.stack(masks)
