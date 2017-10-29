"""
"""

from collections import deque

import numpy as np
import cv2
import gym

from agent import Agent
from gym_logger import GymLogger
from gym_utils import take_steps


def train_and_test_agent(
    environment_name='CarRacing-v0', num_episodes=100, max_test_length=5,
    target_update_freq=1, initial_test_freq=2, replay_buffer_size=5e4, training_batch_size=32,
    learning_rate=0.001, rewards_threshold=900, action_step_length=3):

    # Set up environment and agent
    env = gym.make(environment_name)
    agent = Agent(
        env, replay_buffer_size=replay_buffer_size, learning_rate=learning_rate,
        training_batch_size=training_batch_size, num_actions=(5, 4, 4))

    # Set up logging
    meta_data = ({
        'env': environment_name, 'target_update_freq': target_update_freq,
        'replay_buffer_size': replay_buffer_size, 'training_batch_size': training_batch_size,
        'learning_rate': learning_rate})
    vars_to_track = ('episode', 'testing', 'epsilon', 'step_count', 'rewards')
    logger = GymLogger(meta_data, vars_to_track)

    # Start by filling replay buffer using random actions
    while len(agent.memory.replay_buffer) < 100:
        env.reset()
        state = None
        done = False
        while not done:
            action_num = np.random.choice(agent.num_actions)
            action = agent.action_space[action_num, :]
            next_state, reward, done, took_all_steps = take_steps(env, action, action_step_length)
            if took_all_steps and state is not None:
                if state.shape[2] == 9 and next_state.shape[2] == 9:  # Fix this
                    agent.memory.update((state, action_num, reward, next_state, done))
            state = next_state

    # Use testing flag to determine action selection below
    testing = False
    test_rewards = deque()
    best_test_rewards = 0

    for episode in range(num_episodes):

        # Throw away first few frames when camera is zooming in
        env.reset()
        for _ in range(10):
            state, _, _, _ = take_steps(env, (0, 1, 0), action_step_length)

        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            if not testing:
                action_num, action = agent.determine_action(state)
            else:
                action_num, action = agent.act(state)
            next_state, reward, done, took_all_steps = take_steps(env, action, action_step_length)
            episode_reward += reward
            if took_all_steps and state is not None:
                if state.shape[2] == 9 and next_state.shape[2] == 9:  # Fix this
                    agent.memory.update((state, action_num, reward, next_state, done))
                    agent.learner.train(agent.memory.random_sample(num_samples=training_batch_size))
            state = next_state
            step_count += 1
            if done:
                logger.update((episode, testing, agent.epsilon, step_count, episode_reward))

        if testing:

            test_rewards.append(episode_reward)
            mean_test_rewards = np.mean(test_rewards)
            print('episode: ' + str(episode))
            print('step count: ' + str(step_count))
            print('learning rate: ' + str(learning_rate))
            print('epsilon: ' + str(agent.epsilon))
            print('mean test rewards: ' + str(mean_test_rewards))
            print('test episodes: ' + str(len(test_rewards)))
            print('\n')

            if mean_test_rewards > best_test_rewards:
                print('New best test result')
                logger.save_model_weights(agent.learner.target_model.get_weights(), 'best')
                best_test_rewards = mean_test_rewards

            if mean_test_rewards < rewards_threshold:
                testing = False
                test_rewards.clear()
            elif len(test_rewards) == max_test_length:
                print('Success!')
                logger.save_model_weights(agent.learner.target_model.get_weights(), episode)
                break

        if (episode >= target_update_freq) and (episode % target_update_freq == 0):
            agent.learner.update_target()

        if (episode > initial_test_freq) and (episode % initial_test_freq) == 0:
            testing = True

        agent.udpate_epsilon()

    logger.save_history()
    env.close()


train_and_test_agent()
