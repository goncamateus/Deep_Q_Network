""" 
Code made with help of https://keon.io/deep-q-learning/
"""

""" 
PARAMETERS
episodes - a number of games we want the agent to play.
gamma - aka decay or discount rate, to calculate the future discounted reward.
epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
epsilon_min - we want the agent to explore at least this amount.
learning_rate - Determines how much neural net learns in each iteration.

"""

import os
import random
import time
from collections import deque

import h5py
import numpy as np

import gym
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

ENVIRON = 'MsPacman-ram-v0'

class DQNAgent:
	def __init__(self, state_size=128, action_size=9):
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9
		self.action_size = action_size
		self.state_size = state_size
		if 'DQN_{}.h5'.format(ENVIRON) in os.listdir('.'):
			print("---------LOADING {} MODEL -------------".format(ENVIRON))
			self.dqn = load_model('DQN_{}.h5'.format(ENVIRON))
		else:
			self.dqn = self._build_dqn()

	def _build_dqn(self):
		dqn = Sequential()
		dqn.add(Dense(24,input_dim=self.state_size, activation='relu'))
		dqn.add(Dense(36,activation='relu'))
		dqn.add(Dense(24, activation='relu'))
		dqn.add(Dense(self.action_size, activation='linear'))
		dqn.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return dqn

	def train_dqn(self, state, reward, epochs=5):
		self.dqn.fit(state, reward, epochs=epochs, verbose=0)

	def act(self, state, env):
		if np.random.rand() <= self.epsilon:
			return env.action_space.sample()

		act_values = self.dqn.predict(state)

		return np.argmax(act_values[0])

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay(self, batch_size):		
		if batch_size > len(self.memory):
			minibatch = self.memory
		else:
			bests = sorted(self.memory, key=lambda mem: mem[2])[-batch_size:]
			minibatch = random.sample(bests, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + \
							self.gamma*np.amax(self.dqn.predict(next_state)[0])

			target_f = self.dqn.predict(state)
			target_f[0][action] = target

			self.train_dqn(state, target_f)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save_dqn(self):
		self.dqn.save('DQN_{}.h5'.format(ENVIRON))


def main():
	epochs = 10

	try:
		env = gym.make(ENVIRON)
		agent = DQNAgent()

		for e in range(epochs):
			state = env.reset()
			state = np.reshape(state, [1,128])

			for time_t in range(10000): # run for 500 steps

				env.render()

				action = agent.act(state, env)
				time.sleep(0.01)
				next_state, reward, done, _ = env.step(action) # take action

				next_state = np.reshape(next_state, [1,128])

				agent.remember(state, action, reward, next_state, done)

				state = next_state

				if done:
					print('Episode: {}/{}, score: {}'.format(e+1, epochs, time_t))
					break

			agent.replay(32)
		agent.save_dqn()
		env.close()
	except KeyboardInterrupt:
		agent.save_dqn()
		env.close()
		print("-----------------Keyboard Interruption------------------")
		print("-------------------Saved {} DQN-------------------------".format(ENVIRON))
if __name__ == '__main__':
	main()
