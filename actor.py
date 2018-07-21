import tensorflow as tf

class Actor():

	def __init__(self,nA,scope,state_size):

		self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=(None,state_size))
		self.nA = nA
		self.scope = scope
		self.model = self.build_model()


	def build_model(self):
		with tf.variable_scope(self.scope):
			x = tf.layers.dense(inputs=self.input_placeholder,units=400)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=300)
			x = tf.contrib.layers.layer_norm(inputs=x)
			x = tf.nn.relu(x)
			x = tf.layers.dense(inputs=x,units=self.nA,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
			return tf.nn.tanh(x)


