import cv2
import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import random

class VideoDataset2(Dataset):
    def __init__(self, root_folder, transform=None, difficulty = 2):
        self.root_folder = root_folder
        self.transform = transform
        self.subfolders = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
        self.brightness = 40
        self.noise = 20
        self.difficulty = difficulty

    def corrupt_frame(self, frame, frame_index):
        frame_index = frame_index // 2
        h, w, _ = frame.shape
        mask = np.ones_like(frame)
        
        if self.difficulty >= 2:
            n = random.randint(0, 4) 
            if n < 1:
                corrupted_frame = np.clip(frame+self.brightness, 0, 255)
                return corrupted_frame, mask
            if n < 2:
                noise_matrix = np.random.randint(-self.noise, self.noise, frame.shape, dtype=np.int32)
                corrupted_frame = np.clip(frame.astype(np.int32) + noise_matrix, 0, 255).astype(np.uint8)
                return corrupted_frame, mask
        if self.difficulty > 0:
            for i in range(1):
                section_height = h // 3
                slice_width = w // 8

                extra_frame_index = random.randint(0, 100) // 2

                section_idx = extra_frame_index // 8  # find the section index
                slice_idx = extra_frame_index % 8  # find the slice index within the section

                raster_center_x = slice_idx * slice_width + slice_width // 2
                raster_center_y = section_idx * section_height + section_height // 2

                # apply random jitter to the center point
                jitter_x = random.randint(-25 // 2, 25 // 2)
                jitter_y = random.randint(-125 // 2, 125 // 2)
                raster_center_x += jitter_x
                raster_center_y += jitter_y

                start_x = max(0, raster_center_x - (225 // 2) // 2)
                end_x = min(w, start_x + 225 // 2)
                start_y = max(0, raster_center_y - (125 // 2) // 2)
                end_y = min(h, start_y + 125 // 2)

                mask[start_y:end_y, start_x:end_x, :] = 0

        h, w, _ = frame.shape
        mask = np.ones_like(frame)


        # Define the number of frames per section
        frames_per_section = 8

        # Calculate the section index based on the current frame index
        section_idx = frame_index // frames_per_section 

        # Calculate the horizontal position of the box within the section
        position_idx = frame_index % frames_per_section

        # Calculate the vertical start and end points of the box
        start_y = section_idx * h // 3
        end_y = start_y + 100

        # Calculate the horizontal start and end points of the box
        start_x = position_idx * w // 8
        end_x = start_x + 150

        # Ensure the box does not go beyond the frame boundaries
        start_x, end_x = max(0, start_x), min(w, end_x)
        start_y, end_y = max(0, start_y), min(h, end_y)

        # Set the box area in the mask to 0
        mask[start_y:end_y, start_x:end_x, :] = 0

        # Apply the mask to the frame to create the corrupted frame
        corrupted_frame = frame * mask

        return corrupted_frame, mask

        


    def __len__(self):
        return len(self.subfolders) * 2 # each folder corresponds to two videos

    def __getitem__(self, idx):
        subfolder = self.subfolders[idx // 2]
        video_folder = os.path.join(self.root_folder, subfolder)
        frames = sorted(os.listdir(video_folder))

        left_video = []
        right_video = []
        if (len(frames) != 50):
            print(f"ERROR {len(frames)=} {video_folder=}")
        for i in range(0, 50, 2):
            frame_path = os.path.join(video_folder, frames[i])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1024, 512)) # resize the frame to 1024x512

            left_frame, right_frame = np.split(frame, 2, axis=1) 
            left_frame = cv2.resize(left_frame, (256, 256)) # resize the left frame to 256x256
            right_frame = cv2.resize(right_frame, (256, 256)) # resize the right frame to 256x256

            corrupted_left, mask_left = self.corrupt_frame(left_frame, i)
            corrupted_right, mask_right = self.corrupt_frame(right_frame, i)

            if self.transform is not None:
                left_frame = self.transform(left_frame)
                right_frame = self.transform(right_frame)
                corrupted_left = self.transform(corrupted_left)
                corrupted_right = self.transform(corrupted_right)

            if idx % 2 == 0:
                left_video.append((corrupted_left, left_frame, mask_left))
            else:
                right_video.append((corrupted_right, right_frame, mask_right))

        if idx % 2 == 0:
            corrupted_frames, frames, masks = zip(*left_video)
        else:
            corrupted_frames, frames, masks = zip(*right_video)

        return torch.from_numpy(np.array(corrupted_frames)).permute(0, 3, 1, 2) / 255, torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2) / 255, torch.from_numpy(np.array(masks)).permute(0, 3, 1, 2) / 255
