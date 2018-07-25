import tensorflow as tf

class Model():
	def __init__(self,scope,action_size,state_size,tau,lr,copy_model=None):
		self.scope = scope
		self.state_size = state_size
		self.action_size = action_size
		self.lr = lr
		self.model,self.loss_summary,self.optimize = self.build_model()
		self.copy_model = copy_model
		
		if self.copy_model is not None:
			self.copy_tensors = [w1.assign(tf.multiply(w1,tau) + tf.multiply(w2,1-tau)) for w1,w2 in zip(self.get_weights(),self.copy_model.get_weights())]

	def build_model(self):
		pass

	def get_weights(self):
		return [weight for weight in tf.trainable_variables(scope=self.scope) if 'LayerNorm' not in weight.name]

	def copy_params(self,sess):

		if self.copy_model is not None:
			sess.run(self.copy_tensors)
		else:
			raise ValueError("Tried to copy params when no copy model was provided")

class Critic(Model):

	def __init__(self,scope,action_size,state_size,tau,lr=0.0001,copy_model=None):
		Model.__init__(self,scope=scope,action_size=action_size,state_size=state_size,tau=tau,lr=lr,copy_model=copy_model)
		self.action_gradients = self.build_action_grads()

	def build_model(self):
		with tf.variable_scope(self.scope):
			self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size),name="input_placeholder")
			self.action_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.action_size),name="action_placeholder")
			self.target_placeholer = tf.placeholder(dtype=tf.float32,shape=(None),name="target_placeholder")
			
			x = tf.layers.dense(inputs=self.input_placeholder,units=400,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.concat([x,self.action_placeholder],axis=-1)
			x = tf.layers.dense(inputs=x,units=300,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			loss = tf.losses.mean_squared_error(x,self.target_placeholer)
			loss_summary = tf.summary.scalar('critic_loss',loss)
			optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

			return tf.nn.tanh(x), loss_summary, optimize
	
	def build_action_grads(self):
		return tf.gradients(self.model,self.action_placeholder)
	
	def action_grads(self,sess,states,actions):
		return sess.run(self.action_gradients,feed_dict={self.input_placeholder:states,self.action_placeholder:actions})

	def update(self,sess,filewriter,states,actions,values):
		_,summary = sess.run([self.optimize,self.loss_summary],feed_dict={self.input_placeholder:states,self.action_placeholder:actions,self.target_placeholer:values})
		filewriter.add_summary(summary,tf.train.global_step(sess,tf.train.get_global_step(graph=sess.graph)))

	def get_target(self,sess,state,action):
		return sess.run(self.model,feed_dict={self.input_placeholder:state,self.action_placeholder:action})

		
class Actor(Model):

	def __init__(self,scope,action_size,state_size,tau,lr=0.0001,copy_model=None):
		Model.__init__(self,scope=scope,action_size=action_size,state_size=state_size,tau=tau,lr=lr,copy_model=copy_model)
		self.optimize = self.build_grads()

	def build_model(self):
		with tf.variable_scope(self.scope):
			self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.state_size))
			
			x = tf.layers.dense(inputs=self.input_placeholder,units=400,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=300,kernel_initializer=tf.keras.initializers.he_normal())
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=self.action_size,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			
			return tf.nn.tanh(x),None,None

	def build_grads(self):
		self.action_grad_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,self.action_size))
		model_weights = self.get_weights()
		model_grads = tf.gradients(self.model,model_weights,-self.action_grad_placeholder)
		grads_weights = zip(model_grads,model_weights)
		return tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(grads_weights)

	def update(self,sess,filewriter,states,action_grads):
		sess.run(self.optimize,feed_dict={self.input_placeholder:states,self.action_grad_placeholder:action_grads})

	def get_action(self,sess,state):
		return sess.run(self.model,feed_dict={self.input_placeholder:state})


	