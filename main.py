from agent import DDPG

def main():

	params = {
		'actor_learning_rate':0.0001,
		'critic_learning_rate':0.0001,
		'gamma':0.99,
		'tau':0.1,
		'num_episodes':10000,
		'replay_size':1000000,
		'replay_init_size':1000,
		'batch_size':64,
		'env':'Hopper-v2'
	}
	
	agent = DDPG(params)
	agent.train()

if __name__ == "__main__":
	main()