from agent import DDPG

def main():

	params = {
		'actor_learning_rate':1e-4,
		'critic_learning_rate':1e-3,
		'gamma':0.99,
		'tau':0.001,
		'sigma':0.2,
		'num_epochs':5000,
		'num_episodes':1000,
		'replay_size':1000000,
		'num_train_steps':50,
		'replay_init_size':1000,
		'batch_size':64,
		'render_train':False,
		'restore':False,
		'env':'Hopper-v2'
	}
	
	agent = DDPG(params)
	agent.train()

if __name__ == "__main__":
	main()