

import numpy as np

from memory import Memory
from conv_ddqn import ConvDDQN
from gym_utils import build_action_space


class Agent:
    def __init__(
        self, gym_env,
        epsilon_parms=(.5, 0.1, 0.002), replay_buffer_size=1e5, gamma=0.99, learning_rate=0.001,
        training_batch_size=64, num_actions=(3, 3, 2)):
        """
        Args (where not clear from name):
            epsilon_parms: a tuple w/ epsilon settings for exploration: initial value, final value,
                and linear decay rate

        """

        # Set up exploration parameters
        self.epsilon_high = epsilon_parms[0]
        self.epsilon_low = epsilon_parms[1]
        self.epsilon = self.epsilon_high
        self.epsilon_linear_decay = epsilon_parms[2]

        # Discretize action space
        print(num_actions)
        self.action_space = build_action_space(gym_env, num_actions=num_actions)
        self.num_actions = self.action_space.shape[0]

        self.memory = Memory(size=replay_buffer_size)

        self.learner = ConvDDQN(
            gamma=gamma, learning_rate=learning_rate, training_batch_size=training_batch_size,
            num_actions=self.num_actions)

    def act(self, state, use_target_model=True):
        state = np.expand_dims(state, axis=0)
        if use_target_model:
            action_num = np.argmax(self.learner.target_model.predict(state))
        else:
            action_num = np.argmax(self.learner.primary_model.predict(state))
        return((action_num, self.action_space[action_num, :]))

    def determine_action(self, state):
        if self.epsilon > np.random.rand():
            action_num = np.random.choice(self.num_actions)
            return(action_num, self.action_space[action_num, :])
        else:
            return(self.act(state))

    def udpate_epsilon(self):
        if self.epsilon > self.epsilon_low:
            self.epsilon -= self.epsilon_linear_decay