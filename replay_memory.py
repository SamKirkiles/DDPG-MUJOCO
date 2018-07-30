import pickle
import numpy as np
import os
import random

class Memory:

	def __init__(self,replay_size=None,action_size=None,state_size=None,batch_size=None):

		self.buffer = 4
		self.batch_size=batch_size
		self.replay_size = replay_size
		self.state_size = state_size
		self.action_size = action_size

		self.states = np.empty((self.replay_size,self.state_size),dtype=np.float32)
		self.rewards = np.empty((self.replay_size),dtype=np.float32)
		self.terminal = np.empty((self.replay_size),dtype=np.bool)
		self.actions = np.empty((self.replay_size,self.action_size),dtype=np.float32)

		self.state_buffer = np.empty((self.batch_size,self.state_size),dtype=np.float32)
		self.next_state_buffer = np.empty((self.batch_size,self.state_size),dtype=np.float32)
		
		self.current = 0
		self.filled = False


	def replay_length(self):
		return min(self.current,self.replay_size)

	def add(self,state,action,reward,terminate,next_state):

		self.states[self.current] = state
		self.rewards[self.current] = reward
		self.terminal[self.current] = terminate
		self.actions[self.current] = action

		self.current += 1

		if self.current == self.replay_size:
			self.current = 0
			self.filled = True



	def sample(self):

		
		indexes = []

		while len(indexes) != self.batch_size:
			if self.filled:
				index = np.random.randint(0,self.replay_size-2)

				# Reject this because we are writing here
				if index >= self.current and index - self.buffer <= self.current:
					continue
			else:

				index = np.random.randint(self.buffer-1,self.current -2)

			if self.terminal[(index-self.buffer):index].any():
				continue
			
			self.state_buffer[len(indexes)] = self.states[index]
			self.next_state_buffer[len(indexes)] = self.states[index+1]
			indexes.append(index)

		action = self.actions[indexes]
		terminal = self.terminal[indexes]
		reward = self.rewards[indexes]

		return self.state_buffer, action, reward, self.next_state_buffer, terminal
