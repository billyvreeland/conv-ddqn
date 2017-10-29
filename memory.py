
from collections import deque
import random

import numpy as np


class Memory:
    """
    Replay buffer implementation.


    """

    def __init__(self, size=1e5):
        """
        Inits Memory class with maximum size of deque.
        """
        self.size = size
        self.replay_buffer = deque(maxlen=size)

    def update(self, experience_tuple):
        self.replay_buffer.append(experience_tuple)

    def random_sample(self, num_samples=64):
        if len(self.replay_buffer) <= num_samples:
            samples = self.replay_buffer
        else:
            samples = random.sample(self.replay_buffer, num_samples)

        item_idx = range(len(samples[0]))
        return(map(np.array, [[s[i] for s in samples] for i in item_idx]))
