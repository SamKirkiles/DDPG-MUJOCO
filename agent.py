import gym
import mujoco_py
import numpy as np
from replay_memory import Memory
from noise import OrnsteinUhlenbeckActionNoise
from actor_critic import ActorCritic
import tensorflow as tf
import os
from gym.wrappers import Monitor
from terminaltables import AsciiTable


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
		self.actor_critic = ActorCritic(
			actor_lr=parameters['actor_learning_rate'],
			critic_lr=parameters['critic_learning_rate'],
			gamma=parameters['gamma'],
			state_size=self.state_size,
			action_size=self.nA,
			tau=parameters['tau']
		)


	def train(self):	

		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		# Create global step and increment operation
		global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
		increment_global_step = tf.assign_add(global_step_tensor,1)

		# Create model saver
		saver = tf.train.Saver()

		sess = tf.Session(config=config)

		if not self.parameters['restore']:
			sess.run(tf.global_variables_initializer())
		else:
			saver.restore(sess, tf.train.latest_checkpoint('./saves'))

		self.actor_critic.set_moving_to_target(sess)
		run_id = np.random.randint(10000)

		trainwriter = tf.summary.FileWriter(logdir='./logs/' + str(run_id),graph=sess.graph)

		# Get action noise
		action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nA), sigma=float(self.parameters['sigma']) * np.ones(self.nA))

		# Fill Replay Memory
		state = self.env.reset()
		fill_amount = 0
		while fill_amount < self.parameters['replay_init_size']:

			action = self.env.action_space.sample()
			next_state, reward,done,_  = self.env.step(action)

			if done:
				state = self.env.reset()
			else:
				fill_amount += 1
				self.memory.add(state,action,reward,done,next_state)
				state = next_state
		
		# Main Loop
		steps = 0
		
		for i in range(self.parameters['num_epochs']):

			avg_epoch_rewards = 0
			num_epochs = 1
			for e in range(self.parameters['num_episodes']):
				 
				state = self.env.reset()

				ep_reward = 0 

				# Perform rollout
				while True:
					noise = action_noise()
					action = self.actor_critic.pi(sess,state[None,...])	
					action += noise
					action = np.clip(action,self.env.action_space.low[0],self.env.action_space.high[0])
					
					assert action.shape == self.env.action_space.shape
					
					"""
					# UNCOMMENT TO PRINT ACTIONS
					a0 = tf.Summary(value=[tf.Summary.Value(tag="action_0", simple_value=action[0,0])])
					trainwriter.add_summary(a0,steps)
					a1 = tf.Summary(value=[tf.Summary.Value(tag="action_1", simple_value=action[0,1])])
					trainwriter.add_summary(a1,steps)
					a2 = tf.Summary(value=[tf.Summary.Value(tag="action_2", simple_value=action[0,2])])
					trainwriter.add_summary(a2,steps)
					steps += 1
					"""

					next_state, reward,done,_  = self.env.step(action)


					self.memory.add(state,action,reward,done,next_state)
					
					if self.parameters['render_train']:
						self.env.render()


					ep_reward += reward
					
					if done:

						reward_summary = tf.Summary(value=[tf.Summary.Value(tag="ep_rewards", simple_value=ep_reward)])
						trainwriter.add_summary(reward_summary,i*self.parameters['num_episodes']+e)
						action_noise.reset()
						break

					state = next_state

				avg_epoch_rewards = avg_epoch_rewards + (ep_reward - avg_epoch_rewards)/num_epochs
				num_epochs += 1

				# Perform train
				for t in range(self.parameters['num_train_steps']):
					s_state, s_action, s_reward, s_next_state,s_terminal = self.memory.sample()
					# Train actor critic model
					self.actor_critic.update(sess=sess, filewriter=trainwriter, state_batch=s_state, next_state_batch=s_next_state, action_batch=s_action, reward_batch=s_reward,done_batch=s_terminal)
					sess.run(increment_global_step)

			# Print out epoch stats here

			table_data = [
			['Epoch','Average Reward'],
			[str(i) + "/" +str(self.parameters['num_epochs']), str(avg_epoch_rewards)]
			]
			
			
			table = AsciiTable(table_data,"Training Run: " + str(run_id))

			save_path = saver.save(sess, "./saves/model.ckpt")
			
			os.system('clear') 
			print("Model saved in path: %s" % save_path + "\n" + table.table)

	def test(self):
		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		saver = tf.train.Saver()
		sess = tf.Session(config=config)

		saver.restore(sess, tf.train.latest_checkpoint('./saves'))
		

		while True:
			 
			state = self.env.reset()

			# Perform rollout
			while True:
				action = self.actor_critic.pi(sess,state[None,...])	
				action = np.clip(action,self.env.action_space.low[0],self.env.action_space.high[0])

				assert action.shape == self.env.action_space.shape
				
				next_state, reward,done,_  = self.env.step(action)
				
				self.env.render()
				
				if done:

					break

				state = next_state
