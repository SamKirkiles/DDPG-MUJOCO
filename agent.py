import gym
import mujoco_py
from replay_memory import Memory
from actor import Actor
from critic import critic

class DDPG():

	def __init__(self,parameters):
		
		
		self.parameters = parameters
		self.env = gym.make(self.parameters['env']) 
		self.nA = self.env.action_space.sample().shape[0]
		self.state_size = self.env.reset().shape[0]

		# Build our replay memory
		self.memory = Memory(
			replay_size=self.parameters['replay_size'],
			state_size=self.state_size,
			batch_size=self.parameters['batch_size']
		)

		# Create actor and critic
		self.actor = Actor(self.nA)
		self.critic = Critic()

		self.target_actor = Actor(self.nA)
		self.target_critic = Critic(self.nA)

	def train(self):
		
		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)

		state = self.env.reset()

		for e in range(self.parameters['num_episodes']):
			while True:

				next_state, reward,done,_  = self.env.step(self.env.action_space.sample())
				self.env.render()
				#print(next_state)
				if done:
					break