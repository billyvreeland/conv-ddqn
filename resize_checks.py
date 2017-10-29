

import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt

from gym_utils import take_steps

env = gym.make('CarRacing-v0')

env.reset()

for i in range(200):
    state, _, _, _ = env.step((0, 1, 0))
    if i % 10 == 0:
        env.render('rgb_array')
        plt.imshow(state)
        plt.show()

env.close()