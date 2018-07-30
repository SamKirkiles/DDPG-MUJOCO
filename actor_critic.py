import numpy as np
import tensorflow as tf

class ActorCritic():

	def __init__(self,actor_lr,critic_lr,gamma,state_size,action_size,tau):
			
		self.gamma = gamma
		self.state_size = state_size
		self.action_size = action_size
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.tau = tau

		# Create input placeholders. If nothing is fed into action_placeholder, the actor is used for calculating gradients of critic
		# Feed in action placeholder when updating critic. The update should be divided into multiple sess.run
		self.state_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size),name="input_placeholder")

		# Build actor graph
		self.actor_model = self._build_actor(scope="actor",reuse=None)

		self.action_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.action_size), name="action_placeholder")
		self.target_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,1),name="target_placeholder")

		# Build critic model
		self.critic_model = self._build_critic(scope="critic",action_op=self.action_placeholder,reuse=None)
		self.critic_with_actor = self._build_critic(scope="critic",action_op=self.actor_model,reuse=True)

		# Build target models
		self.target_actor_model = self._build_actor(scope="target_actor",reuse=None)
		self.target_critic_model = self._build_critic(scope="target_critic",action_op=self.action_placeholder,reuse=None)

		# Build copy operations on weights built above
		self.actor_copy_tensors, self.critic_copy_tensors = self._build_copy_ops()

		# Update operations
		self.critic_optimizer,self.critic_loss_summary = self._build_critic_train_ops()
		self.actor_optimizer, self.actor_loss_summary = self._build_actor_train_ops()

	def _build_actor(self,scope,reuse=None):
		# Build tensorflow graph for actor
		with tf.variable_scope(scope,reuse=reuse):			
			x = tf.layers.dense(inputs=self.state_placeholder,units=400)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=300)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=self.action_size,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			return tf.nn.tanh(x)


	def _build_critic(self,scope,action_op,reuse=None):
		# Build tensorflow graph for critic
		with tf.variable_scope(scope,reuse=reuse):
			x = tf.layers.dense(inputs=self.state_placeholder,units=400,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-2))
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.concat([x,action_op],axis=-1)
			x = tf.layers.dense(inputs=x,units=300,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-2))
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-2))
			return x
	
	def _get_weights(self,scope):
		# Gets the weights for a specific scope
		return [weight for weight in tf.trainable_variables(scope=scope)]

	def set_moving_to_target(self,sess):
		actor_copy_tensors = [w1.assign(w2) for w1,w2 in zip(self._get_weights("target_actor"),self._get_weights("actor"))]
		critic_copy_tensors = [w1.assign(w2) for w1,w2 in zip(self._get_weights("target_critic"),self._get_weights("critic"))]
		sess.run([actor_copy_tensors,critic_copy_tensors])

	def _build_copy_ops(self):
		actor_copy_tensors = [w1.assign(tf.multiply(w2,self.tau) + tf.multiply(w1,1-self.tau)) for w1,w2 in zip(self._get_weights("target_actor"),self._get_weights("actor"))]
		critic_copy_tensors = [w1.assign(tf.multiply(w2,self.tau) + tf.multiply(w1,1-self.tau)) for w1,w2 in zip(self._get_weights("target_critic"),self._get_weights("critic"))]

		return actor_copy_tensors, critic_copy_tensors

	def _build_critic_train_ops(self):
		# Build train operation for critic. This operation must take states, actions, and rewards from sampled batch. 
		mse = tf.losses.mean_squared_error(labels=self.target_placeholder,predictions=self.critic_model)
		critic_loss_summary = tf.summary.scalar("critic_loss",mse)
		mse += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return tf.train.AdamOptimizer(self.critic_lr).minimize(mse,var_list=self._get_weights("critic")),critic_loss_summary

	def _build_actor_train_ops(self):
		# Calculates and applies the gradients to the actor model
		# We do not pass anything into the action placeholder so the default value of the actor model gets used in the critic 
		actor_weights = self._get_weights("actor")
		critic_loss = -tf.reduce_mean(self.critic_with_actor)
		actor_loss_summary = tf.summary.scalar("actor_loss",critic_loss)
		model_grads = tf.gradients(critic_loss, actor_weights)
		grads_weights = zip(model_grads,actor_weights)
		optimize = tf.train.AdamOptimizer(learning_rate=self.actor_lr).apply_gradients(grads_weights)
		return optimize, actor_loss_summary

	def pi(self,sess,state):
		# Maps state to action
		return sess.run(self.actor_model,feed_dict={self.state_placeholder:state})

	def Q(self,sess,state,action):
		return sess.run(self.critic_model,feed_dict={self.state_placeholder:state,self.action_placeholder:action})

	# Update the actor and critic towards a sample batch
	def update(self, sess, filewriter, state_batch, next_state_batch, action_batch, reward_batch,done_batch):
		# Calculate update target from bellman equation 
		target_next_action = sess.run(self.target_actor_model,feed_dict={self.state_placeholder:next_state_batch})
		Q_next = sess.run(self.target_critic_model,feed_dict={self.state_placeholder:next_state_batch,self.action_placeholder:target_next_action})
		targets = reward_batch + (1-done_batch) * self.gamma * Q_next[0]

		# Update critic
		_,critic_loss = sess.run([self.critic_optimizer,self.critic_loss_summary],feed_dict={self.state_placeholder:state_batch,self.action_placeholder:action_batch,self.target_placeholder:targets[:,None]})
		filewriter.add_summary(critic_loss,tf.train.global_step(sess,tf.train.get_global_step(graph=sess.graph)))
		# Update actor using policy gradient and copy tensors
		_,actor_loss = sess.run([self.actor_optimizer,self.actor_loss_summary],feed_dict={self.state_placeholder:state_batch})
		filewriter.add_summary(actor_loss,tf.train.global_step(sess,tf.train.get_global_step(graph=sess.graph)))
		
		# Update stationary targets
		sess.run([self.actor_copy_tensors, self.critic_copy_tensors])