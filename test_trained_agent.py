
import pickle

import gym

from Agent import Agent

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, '/tmp/lunar_lander-1', force=True)

agent = Agent(replay_buffer_size=1e5, training_batch_size=64, learning_rate=1e-3)


saved_weights = pickle.load(open(
	'./test_results/2017-10-10 04:13:04.920454--7a0a1dbe-2a73-4a3f-a611-3388aba46aee/episode_622_weights.pickle',
	'rb'))

agent.learner.target_model.set_weights(saved_weights)

state = env.reset()
done = False
episode_reward = 0
while not done:
	action = agent.learner.act(state)
	# action = env.action_space.sample()
	next_state, reward, done, info = env.step(action)
	episode_reward += reward
	state = next_state

print(episode_reward)

env.monitor.close()
