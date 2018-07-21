import gym
import mujoco_py
from replay_memory import Memory

from models import Actor, Critic
import tensorflow as tf

class DDPG():

	def __init__(self,parameters):
		
		
		self.parameters = parameters
		self.env = gym.make(self.parameters['env']) 
		self.nA = self.env.action_space.sample().shape[0]
		self.state_size = self.env.reset().shape[0]

		# Build our replay memory
		self.memory = Memory(
			replay_size=self.parameters['replay_size'],
			action_size=self.nA,
			state_size=self.state_size,
			batch_size=self.parameters['batch_size']
		)

		# Create actor and critic
		self.actor = Actor(scope="actor",action_size=self.nA,state_size=self.state_size,lr=self.parameters['actor_learning_rate'],copy_model=None)
		self.critic = Critic(scope="critic",action_size=self.nA,state_size=self.state_size,lr=self.parameters['critic_learning_rate'],copy_model=None)

		self.target_actor = Actor(scope="target_actor",action_size=self.nA,state_size=self.state_size,copy_model=self.actor)
		self.target_critic = Critic(scope="target_critic",action_size=self.nA,state_size=self.state_size,copy_model=self.critic)

	def train(self):	

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())


		# Fill Replay Memory
		state = self.env.reset()
		fill_amount = 0
		while fill_amount < self.parameters['replay_init_size']:

			action = self.actor.get_action(sess,state[None,...])
			next_state, reward,done,_  = self.env.step(self.env.action_space.sample())

			if done:
				state = self.env.reset()
			else:
				fill_amount += 1
				self.memory.add(state,action[0],reward,done,next_state)
				state = next_state

		# Main Loop

		state = self.env.reset()
		for e in range(self.parameters['num_episodes']):
			while True:
				action = self.actor.get_action(sess,state[None,...])
				next_state, reward,done,_  = self.env.step(action)

				self.memory.add(state,action[0],reward,done,next_state)
				
				self.env.render()
				
				s_state_buffer, s_action, s_reward, s_next_state_buffer,s_terminal = self.memory.sample()

				if done:
					break

				state = next_state