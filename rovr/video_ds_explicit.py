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
        self.l = np.random.permutation(20)[:7]
        self.f = np.random.permutation(20)
        self.helper = [[i, i] for i in range(6)]

    def new_random(self):
        self.l = np.random.permutation(20)[:7]
        self.f = np.random.permutation(20)
        f = self.f
        self.helper = [
            [f[0], f[1], f[4], f[5]], 
            [f[2], f[3], f[6], f[7]],
            [f[8], f[9], f[12], f[13]],
            [f[10], f[11], f[14], f[15]],
            [f[16], f[17]],
            [f[18], f[19]]
        ]
    def __len__(self):
        return len(self.subfolders) * 2 # each folder corresponds to two videos

    def corrupt_mask(self, frame, frame_index, mask):
        h, w, _ = frame.shape
        section_height = h // 3
        slice_width = w // 8

        section_idx = frame_index // 8  # find the section index
        slice_idx = frame_index % 8  # find the slice index within the section

        raster_center_x = slice_idx * slice_width + slice_width // 2
        raster_center_y = section_idx * section_height + section_height // 2

        # apply random jitter to the center point
        jitter_x = random.randint(-25 // 2, 25 // 2)
        jitter_y = random.randint(-125 // 2, 125 // 2)
        raster_center_x += jitter_x
        raster_center_y += jitter_y

        start_x = max(0, raster_center_x - (200 // 2) // 2)
        end_x = min(w, start_x + 200 // 2)
        start_y = max(0, raster_center_y - (100 // 2) // 2)
        end_y = min(h, start_y + 100 // 2)

        mask[start_y:end_y, start_x:end_x, :] = 0

        return mask

    def corrupt_frame(self, frame, locations):
        h, w, _ = frame.shape
        mask = np.ones_like(frame)

        for location in locations:
            mask = self.corrupt_mask(frame, location, mask)

        corrupted_frame = frame * mask

        return corrupted_frame, mask


    def __getitem__(self, idx):
        subfolder = self.subfolders[idx // 2]
        video_folder = os.path.join(self.root_folder, subfolder)
        frames = sorted(os.listdir(video_folder))

        self.new_random()
        frame_masks = self.choose_frame_masks()
        solutions = self.generate_solutions()
        negative_solutions = self.generate_negative_solutions()

        left_video = []
        right_video = []
        
        if (len(frames) != 50):
            print(f"ERROR {len(frames)=} {video_folder=}")

        for i in range(0, 40, 2):
            frame_path = os.path.join(video_folder, frames[self.f[i//2]])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1024, 512)) # resize the frame to 1024x512

            left_frame, right_frame = np.split(frame, 2, axis=1) 
            left_frame = cv2.resize(left_frame, (256, 256)) # resize the left frame to 256x256
            right_frame = cv2.resize(right_frame, (256, 256)) # resize the right frame to 256x256

            if idx % 2 == 0:
                corrupted, mask = self.corrupt_frame(left_frame, frame_masks[i//2])
                left_video.append((corrupted, left_frame, mask))
            else:
                corrupted, mask = self.corrupt_frame(right_frame, frame_masks[i//2])
                right_video.append((corrupted, right_frame, mask))

        if idx % 2 == 0:
            corrupted_frames, frames, masks = zip(*left_video)
        else:
            corrupted_frames, frames, masks = zip(*right_video)

        return torch.from_numpy(np.array(corrupted_frames)).permute(0, 3, 1, 2) / 255, torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2) / 255, torch.from_numpy(np.array(masks)).permute(0, 3, 1, 2) / 255, torch.from_numpy(solutions), torch.from_numpy(negative_solutions)

    def choose_frame_masks(self):
        frame_masks = []
        for i in range(20):
            if i in self.helper[0]:
                frame_masks.append([self.l[0], self.l[1], self.l[3], self.l[5]])
            elif i in self.helper[1]:
                frame_masks.append([self.l[0], self.l[1], self.l[4], self.l[6]])
            elif i in self.helper[2]:
                frame_masks.append([self.l[1], self.l[2], self.l[3], self.l[6]])
            elif i in self.helper[3]:
                frame_masks.append([self.l[1], self.l[2], self.l[4], self.l[5]])
            elif i in self.helper[4]:
                frame_masks.append([self.l[0], self.l[2], self.l[3], self.l[5]])
            elif i in self.helper[5]:
                frame_masks.append([self.l[0], self.l[2], self.l[4], self.l[6]])
        return np.array(frame_masks)
    def generate_neg_solutions(self):
        neg_solutions = np.empty((20, 8, 2))

    def generate_solutions(self):
        solutions = np.empty((20, 32, 2))
        for i in range(20):
            if i in self.helper[0]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[2] for q in self.helper[5]]),
                    np.array([[p, q] for p in self.helper[3] for q in self.helper[4]]), 
                    np.array([[q, p] for p in self.helper[2] for q in self.helper[5]]),
                    np.array([[q, p] for p in self.helper[3] for q in self.helper[4]])
                    ), axis = 0)
            elif i in self.helper[1]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[2] for q in self.helper[4]]),
                    np.array([[p, q] for p in self.helper[3] for q in self.helper[4]]), 
                    np.array([[q, p] for p in self.helper[2] for q in self.helper[4]]),
                    np.array([[q, p] for p in self.helper[3] for q in self.helper[4]])
                    ), axis = 0)
            elif i in self.helper[2]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[0] for q in self.helper[5]]),
                    np.array([[p, q] for p in self.helper[1] for q in self.helper[5]]), 
                    np.array([[q, p] for p in self.helper[0] for q in self.helper[5]]),
                    np.array([[q, p] for p in self.helper[1] for q in self.helper[5]])
                    ), axis = 0)
            elif i in self.helper[3]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[0] for q in self.helper[5]]),
                    np.array([[p, q] for p in self.helper[1] for q in self.helper[4]]), 
                    np.array([[q, p] for p in self.helper[0] for q in self.helper[5]]),
                    np.array([[q, p] for p in self.helper[1] for q in self.helper[4]])
                    ), axis = 0)
            elif i in self.helper[4]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[1] for q in self.helper[2]]),
                    np.array([[q, p] for p in self.helper[1] for q in self.helper[2]]), 
                    ), axis = 0)
            elif i in self.helper[5]:
                solutions[i] = np.concatenate((
                    np.array([[p, q] for p in self.helper[0] for q in self.helper[2]]),
                    np.array([[q, p] for p in self.helper[0] for q in self.helper[2]]), 
                    ), axis = 0)
        return solutions
    
    def generate_negative_solutions(self):
        neg_solutions = np.empty((20, 16, 2))
        for i in range(20):
            for j in range(4):
                if i in self.helper[j]:
                    temp = self.helper[j].copy()
                    temp.remove(i)
                    neg_solutions[i] = np.concatenate((
                        np.array([  [temp[0], temp[1]],
                                    [temp[0], temp[2]],
                                    [temp[1], temp[0]],
                                    [temp[1], temp[2]],
                                    [temp[2], temp[0]],
                                    [temp[2], temp[1]],
                                    ]),
                        np.array([[p, self.helper[(i+1)%4][l]] for p in temp[:2] for l in range(2)]),
                        np.array([[self.helper[(i+1)%4][l], p] for p in temp[:2] for l in range(2)]),
                        np.array([[temp[2], self.helper[(i+1)%4][0]]]),
                        np.array([[self.helper[(i+1)%4][0], temp[2]]]),
                        ), axis = 0)
            if i in self.helper[4]:
                temp = self.helper[4].copy()
                temp.remove(i)
                neg_solutions[i] = np.concatenate((
                    np.array([[p, q] for p in temp for q in self.helper[1]]),
                    np.array([[q, p] for p in temp for q in self.helper[1]]),
                    np.array([[p, q] for p in temp for q in self.helper[2]]),
                    np.array([[q, p] for p in temp for q in self.helper[2]]) 
                    ), axis = 0)
            if i in self.helper[5]:
                temp = self.helper[5].copy()
                temp.remove(i)
                neg_solutions[i] = np.concatenate((
                    np.array([[p, q] for p in temp for q in self.helper[2]]),
                    np.array([[q, p] for p in temp for q in self.helper[2]]), 
                    np.array([[p, q] for p in temp for q in self.helper[1]]),
                    np.array([[q, p] for p in temp for q in self.helper[1]])
                    ), axis = 0)
        return neg_solutions
        

                
