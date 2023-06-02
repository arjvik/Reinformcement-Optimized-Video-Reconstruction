import cv2
import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.videos = os.listdir(folder)
    
    def corrupt_frame(self, frame):
        # regular random mask logic
        mask = np.ones_like(frame)
        h, w, _ = frame.shape
        mask_x = np.random.randint(0, w-50)
        mask_y = np.random.randint(0, h-50)
        mask[mask_y:mask_y+50, mask_x:mask_x+50, :] = 0
        corrupted_frame = frame * mask
        return corrupted_frame, mask
    
    def __len__(self):
        return len(self.videos)
    
    def normalize(self, frame):
        # normalize it in -1 to 1 range
        return (frame / 127.5) - 1
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.folder, self.videos[idx])
        vidcap = cv2.VideoCapture(video_path)
        
        frames = []
        corrupted_frames = []
        masks = []
        i = 0
        
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            
            # resize and crop to get 512x512
            h, w, _ = frame.shape
            new_size = min(h, w)
            startx = w//2 - new_size//2
            starty = h//2 - new_size//2
            frame = frame[starty:starty+new_size, startx:startx+new_size]
            frame = cv2.resize(frame, (512, 512))
            
            # corrupt the frame
            corrupted_frame, mask = self.corrupt_frame(frame)
            
            if self.transform is not None:
                frame = self.transform(frame)
                corrupted_frame = self.transform(corrupted_frame)
                mask = self.transform(mask)
            
            frames.append(self.normalize(frame))
            corrupted_frames.append(self.normalize(corrupted_frame))
            masks.append(mask)

            i += 1
            if i == 49:
                break
                
        if i < 49:
            raise Exception(f"video {self.videos[idx]} has less than 49 frames!")
        
        return torch.stack(corrupted_frames), torch.stack(frames), torch.stack(masks)

    def load(self):
        print('loading videos... hang tight!')
        for i in tqdm(range(len(self.videos))):
            self.__getitem__(i)
