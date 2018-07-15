
def Critic():

	self.input_placeholder = tf.placeholder(dtype=tf.float32,shape=None)

	def build_model():

		x = tf.layers.dense(inputs=self.input_placeholder,units=400)
		x = tf.contrib.layers.layer_norm(inputs=x)
		x = tf.nn.relu(x)
		x = tf.layers.dense(inputs=x,units=300)
		x = tf.contrib.layers.layer_norm(inputs=x)
		x = tf.nn.relu(x)
		x = tf.layers.dense(inputs=x,units=1,kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
		return tf.nn.tanh(x)

	self.model = build_model()



