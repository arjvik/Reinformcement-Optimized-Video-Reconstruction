import os
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

class Logger:
    def __init__(self):
        self.initialize()

    def initialize(self):
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        log_dir = os.path.join("logs", current_time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.steps = 0
        self.episodes = 0

    def log(self, data):
        # Increment episode count
        self.episodes += 1

        # Log values based on steps
        for key, values in data.items():
            if key == "image":
                img = data[key].numpy()  # Assuming images are torch tensors
                self.writer.add_image(f'Image/{key}', img, self.steps)
            elif key == "selected_frame" or key == "context_frames":
                continue  # Skip these here, we handle them separately below
            elif isinstance(values, list) and all(isinstance(i, (int, float)) for i in values):
                for value in values:
                    self.writer.add_scalar(f'Steps/{key}', value, self.steps)
                    self.steps += 1

        # Handle frame data and log it
        selected_frame = data.get('selected_frame')
        context_frames = data.get('context_frames')
        if selected_frame is not None and context_frames is not None:
            for i in range(len(selected_frame)):
                frame_text = f"Target frame: {selected_frame[i]}, Context frames: {context_frames[i][0]}, {context_frames[i][1]}"
                self.writer.add_text('Frames', frame_text, self.steps)
                self.steps += 1  # Increment step for each frame set logged



    def close(self):
        self.writer.close()
