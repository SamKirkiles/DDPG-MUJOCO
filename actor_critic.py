
import tensorflow as tf

class ActorCritic():

	def __init__(self,gamma,state_size,action_size,actor_lr,critic_lr):
			
		self.gamma = gamma
		self.state_size = state_size
		self.action_size = action_size
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr

		# Build actor and critic graphs
		self.actor_model = self._build_actor(scope="actor",reuse=None)
		self.critic_model = self._build_critic(scope="critic",reuse=None)
		self.target_actor_model = self._build_actor(scope="target_actor",reuse=None)
		self.target_critic_model = self._build_critic(scope="target_critic",reuse=None)

		# Build copy operations on weights built above
		self.actor_copy_tensors, self.critic_copy_tensors = self._build_copy_ops()

		# Create input placeholders. If nothing is fed into action_placeholder, the actor is used for calculating gradients of critic
		# Feed in action placeholder when updating critic. The update should be divided into multiple sess.run
		self.state_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size),name="input_placeholder")
		self.action_placeholder = tf.placeholder_with_default(self.actor_model, (None,self.action_size), name="action_placeholder")
		self.target_placeholder = tf.placeholder(dtype=tf.float32,shape=(None),name="target_placeholder")

		# Update operations
		self.critic_optimizer = self._build_critic_train_ops()

	def _build_actor(self,scope,reuse=None):
		# Build tensorflow graph for actor
		with tf.variable_scope(self.scope,reuse=reuse):			
			x = tf.layers.dense(inputs=self.state_placeholder,units=400,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=300,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=self.action_size,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			
			return tf.nn.tanh(x)


	def _build_critic(self,action,scope):
		# Build tensorflow graph for critic
		with tf.variable_scope(scope,reuse=reuse):
			x = tf.layers.dense(inputs=self.state_placeholder,units=400,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.concat([x,self.action_placeholder],axis=-1)
			x = tf.layers.dense(inputs=x,units=300,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

			return tf.nn.tanh(x)
	
	def _get_weights(self,scope):
		# Gets the weights for a specific scope
		return [weight for weight in tf.trainable_variables(scope=scope) if 'LayerNorm' not in weight.name]


	def _build_copy_ops(self):
		actor_copy_tensors = [w1.assign(tf.multiply(w1,tau) + tf.multiply(w2,1-tau)) for w1,w2 in zip(self.get_weights("target_actor"),self.get_weights("actor"))]
		critic_copy_tensors = [w1.assign(tf.multiply(w1,tau) + tf.multiply(w2,1-tau)) for w1,w2 in zip(self.get_weights("target_critic"),self.get_weights("critic"))]

		return actor_copy_tensors, critic_copy_tensors

	def _build_critic_train_ops(self):
		# Build train operation for critic. This operation must take states, actions, and rewards from sampled batch. 
		mse = tf.losses.mean_squared_error(labels=self.target_placeholder,predictions=self.critic_model)
		return tf.train.AdamOptimizer(self.critic_lr).minimize(mse)

	def _build_actor_train_ops():
		# Calculates and applies the gradients to the actor model
		# We do not pass anything into the action placeholder so the default value of the actor model gets used in the critic 
		actor_weights = self.get_weights("actor")
		critic_loss = -tf.reduce_mean(self.critic_model)
		model_grads = tf.gradients(critic_loss,actor_weights)
		grads_weights = zip(model_grads,actor_weights)
		return tf.train.AdamOptimizer(learning_rate=self.actor_lr).apply_gradients(grads_weights)


	def pi(self,sess,state):
		# Maps state to action
		return sess.run(self.actor_model,feed_dict={self.state_placeholder:state})

	# Update the actor and critic towards a sample batch
	def update(self, state_batch, action_batch, reward_batch):
		pass