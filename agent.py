import gym
import mujoco_py
import numpy as np
from replay_memory import Memory
from noise import OrnsteinUhlenbeckActionNoise
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
		# Replace this with 
		# ac = ActorCritic()
		# ActorCritic.update(batch)
		self.actor = Actor(scope="actor",action_size=self.nA,state_size=self.state_size,tau=self.parameters['tau'],lr=self.parameters['actor_learning_rate'],copy_model=None)
		self.critic = Critic(scope="critic",action_size=self.nA,state_size=self.state_size,tau=self.parameters['tau'],lr=self.parameters['critic_learning_rate'],copy_model=None)

		self.target_actor = Actor(scope="target_actor",action_size=self.nA,state_size=self.state_size,tau=self.parameters['tau'],copy_model=self.actor)
		self.target_critic = Critic(scope="target_critic",action_size=self.nA,state_size=self.state_size,tau=self.parameters['tau'],copy_model=self.critic)

	def train(self):	

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		# Create global step and increment operation
		global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
		increment_global_step = tf.assign_add(global_step_tensor,1)

		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())

		run_id = np.random.randint(10000)

		trainwriter = tf.summary.FileWriter(logdir='./logs/' + str(run_id),graph=sess.graph)

		# Get action noise
		action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nA), sigma=float(self.parameters['sigma']) * np.ones(self.nA))

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


		for e in range(self.parameters['num_episodes']):

			state = self.env.reset()

			ep_reward = 0 

			while True:

				action = self.actor.get_action(sess,state[None,...]) + action_noise()
				#action = np.clip(action,self.env.action_space.low[0],self.env.action_space.high[0])
				next_state, reward,done,_  = self.env.step(action)


				self.memory.add(state,action[0],reward,done,next_state)
				
				if self.parameters['render']:
					self.env.render()

				ep_reward += reward
				
				s_state, s_action, s_reward, s_next_state,s_terminal = self.memory.sample()

				s_targets = s_reward + self.parameters['gamma'] * self.target_critic.get_target(sess,s_next_state,s_action)


				critic_update_summary = self.critic.update(sess,trainwriter,s_state,s_action,s_targets)
				action_grads = self.critic.action_grads(sess,s_state,s_action)
				self.actor.update(sess,trainwriter,s_state,action_grads[0])

				self.target_actor.copy_params(sess)
				self.target_critic.copy_params(sess)

				sess.run(increment_global_step)
					

				if done:

					reward_summary = tf.Summary(value=[tf.Summary.Value(tag="ep_rewards", simple_value=ep_reward)])
					trainwriter.add_summary(reward_summary,e)

					break

				state = next_state