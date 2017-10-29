"""
"""

import itertools

import numpy as np
import cv2


def take_steps(gym_env, action, action_step_length):
    step_renders = [None] * action_step_length
    step_rewards = 0
    took_all_steps = False
    for i in range(action_step_length):
        gym_env.render('rgb_array')
        step_render, reward, done, info = gym_env.step(action)
        step_renders[i] = cv2.resize(step_render, (16, 16), interpolation=cv2.INTER_AREA)

        step_rewards += reward
        if done:
            break
    if not any(x is None for x in step_renders):
        took_all_steps = True
        # step_renders = np.dstack((step_renders[0], step_renders[3], step_renders[6])) / 255.  # move and better document this
    else:
        step_renders = [x for x in step_renders if x is not None]
    step_renders = np.dstack(step_renders) / 255.
    return(step_renders, step_rewards, done, took_all_steps)


def build_action_space(gym_env, num_actions):
    """
    Discretizes continuous action space
    """
    print(num_actions)

    action_space_low = gym_env.env.action_space.low
    action_space_high = gym_env.env.action_space.high
    action_list = map(
        lambda low, high, num: np.linspace(low, high, num),
        action_space_low, action_space_high, num_actions)

    print(action_list)

    actions = np.vstack(list(itertools.product(*action_list)))

    idx = np.logical_not(np.logical_and(actions[:, 1] > 0, actions[:, 2] > 0))

    print(actions[idx])

    return(actions[idx])
