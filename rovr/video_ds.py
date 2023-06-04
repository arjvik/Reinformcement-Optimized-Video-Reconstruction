import cv2
import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import random

class VideoDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.subfolders = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

    def corrupt_frame(self, frame, frame_index):
        h, w, _ = frame.shape
        mask = np.ones_like(frame)

        section_height = h // 3
        slice_width = w // 16

        section_idx = frame_index // 16  # find the section index
        slice_idx = frame_index % 16  # find the slice index within the section

        raster_center_x = slice_idx * slice_width + slice_width // 2
        raster_center_y = section_idx * section_height + section_height // 2

        # apply random jitter to the center point
        jitter_x = random.randint(-25, 25)
        jitter_y = random.randint(-125, 125)
        raster_center_x += jitter_x
        raster_center_y += jitter_y

        start_x = max(0, raster_center_x - 225 // 2)
        end_x = min(w, start_x + 225)
        start_y = max(0, raster_center_y - 125 // 2)
        end_y = min(h, start_y + 125)

        mask[start_y:end_y, start_x:end_x, :] = 0
        corrupted_frame = frame * mask

        return corrupted_frame, mask

    def normalize(self, frame):
        return (frame / 127.5) - 1

    def __len__(self):
        return len(self.subfolders) * 2 # each folder corresponds to two videos

    def __getitem__(self, idx):
        subfolder = self.subfolders[idx // 2]
        video_folder = os.path.join(self.root_folder, subfolder)
        frames = sorted(os.listdir(video_folder))
        
        left_video = []
        right_video = []

        for i in range(48):
            frame_path = os.path.join(video_folder, frames[i])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1024, 512))
            
            left_frame, right_frame = np.split(frame, 2, axis=1) 
            
            corrupted_left, mask_left = self.corrupt_frame(left_frame, i)
            corrupted_right, mask_right = self.corrupt_frame(right_frame, i)

            if self.transform is not None:
                left_frame = self.transform(left_frame)
                right_frame = self.transform(right_frame)
                corrupted_left = self.transform(corrupted_left)
                corrupted_right = self.transform(corrupted_right)

            if idx % 2 == 0:
                left_video.append((self.normalize(corrupted_left), self.normalize(left_frame), mask_left))
            else:
                right_video.append((self.normalize(corrupted_right), self.normalize(right_frame), mask_right))

        if idx % 2 == 0:
            corrupted_frames, frames, masks = zip(*left_video)
        else:
            corrupted_frames, frames, masks = zip(*right_video)
            
        return torch.from_numpy(np.array(corrupted_frames)).permute(0, 3, 1, 2), torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2), torch.from_numpy(np.array(masks)).permute(0, 3, 1, 2)
