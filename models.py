import tensorflow as tf

class Model():
	def __init__(self,scope,action_size,state_size,lr,copy_model=None):
		self.scope = scope
		self.state_size = state_size
		self.action_size = action_size
		self.model,self.loss,self.optimize = self.build_model(lr)
		self.should_copy_model = copy_model
			
	def build_model(self,lr):
		pass

	def get_weights(self):
		return [weight for weight in tf.trainable_variables(scope=self.scope) if 'LayerNorm' not in weight.name]

	def copy_params(self,sess,tau):

		if self.should_copy_model is not None and self.copy_model is None:
			self.copy_tensors = [tf.assign_add(tau * w1,(1-tau) * w2) for w1,w2 in zip(self.get_weights(),self.model.get_weights())]

		if self.copy_model is not None:
			sess.run(self.copy_tensors)
		else:
			raise ValueError("Tried to copy params when no copy model was provided")

class Critic(Model):

	def __init__(self,scope,action_size,state_size,lr=0.0001,copy_model=None):
		Model.__init__(self,scope=scope,action_size=action_size,state_size=state_size,lr=lr,copy_model=copy_model)
		self.action_gradients = self.build_action_grads()

	def build_model(self,lr):
		with tf.variable_scope(self.scope):
			self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size))
			self.action_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.action_size))
			self.target_placeholer = tf.placeholder(dtype=tf.float32,shape=(None))
			
			x = tf.layers.dense(inputs=self.input_placeholder,units=400)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.concat([x,self.action_placeholder],axis=-1)
			x = tf.layers.dense(inputs=x,units=300)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			loss = tf.losses.mean_squared_error(x,self.target_placeholer)
			optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

			return tf.nn.tanh(x), loss, optimize
	
	def build_action_grads(self):
		return tf.gradients(self.model,self.action_placeholder)
	
	def update(self,sess,states,values):
		self.loss = sess.run(optimize,feed_dict={self.input_placeholder:states,self.value_placeholder:values})

	def get_target(self,sess,state):
		return sess.run(self.model,feed_dict={self.input_placeholder:state})

		
class Actor(Model):

	def __init__(self,scope,action_size,state_size,lr=0.0001,copy_model=None):
		Model.__init__(self,scope=scope,action_size=action_size,state_size=state_size,lr=lr,copy_model=copy_model)
		self.optimize = build_grads

	def build_model(self,lr):
		with tf.variable_scope(self.scope):
			self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size))
			self.target_placeholder = tf.placeholder(dtype=tf.float32,shape=(None))
			
			x = tf.layers.dense(inputs=self.input_placeholder,units=400)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=300)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=self.action_size,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			
			return tf.nn.tanh(x),None,None

	def build_grads(self):
		self.action_grad_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.action_size))
		model_weights = self.get_weights()
		model_grads = tf.gradients(self.model,model_weights,-self.action_grad_placeholder)
		grads_weights = zip(model_grads,model_weights)
		return = tf.train.AdamOptimizer(learning_rate=0.0001).apply_gradients(grads_weights)

	def get_action(self,sess,state):
		return sess.run(self.model,feed_dict={self.input_placeholder:state})


	