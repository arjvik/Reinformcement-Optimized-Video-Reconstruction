import os
import time
from torch.utils.tensorboard import SummaryWriter

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

        # Calculating averages for episode based logging
        averages = {k: sum(v) / len(v) for k, v in data.items()}

        # Log values based on steps
        for key, values in data.items():
            for value in values:
                self.writer.add_scalar(f'Steps/{key}', value, self.steps)
                self.steps += 1

        # Log values based on episodes
        for key, value in averages.items():
            self.writer.add_scalar(f'Episodes/{key}', value, self.episodes)

    def close(self):
        self.writer.close()