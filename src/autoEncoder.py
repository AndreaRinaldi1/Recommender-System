import numpy as np
import tensorflow as tf

class autoEncoder:

	#Hack to get a tensor sampled from the bernoulli distribution with probability to be one is proba
	def bernoulli_tensor(self, proba, inp):
		return tf.floor(proba + tf.random_uniform(tf.shape(inp), seed = self.randSeed, dtype=inp.dtype))

	def __init__(self,parameters, name, dim, batch, randSeed, optimizer):

		self.dim = dim
		self.name = name
		self.randSeed = randSeed
		self.n_hidden = parameters["hidden_units"]
		self.keepProb = 1-parameters["hideProb"]
		self.gaussianProb = parameters["gaussianProb"]
		self.gaussianStd = parameters["gaussianStd"]
		self.mmProb = parameters["mmProb"]
		self.weights = self._init_weights()
		

		#tensorflow constants and placeholders
		self.input = tf.placeholder(tf.float32, [ None, self.dim], name="input")
		self.input_mask = tf.placeholder(tf.bool, [None, self.dim], name="input_mask")
		self.keepProbTensor = tf.placeholder(tf.float32, shape=[], name="keepNodeProba")
		self.gaussianProbTensor = tf.placeholder(tf.float32, shape=[], name="gaussianNoise_proba")
		self.mmProbTensor = tf.placeholder(tf.float32, shape=[], name="MinMax_probability")
		self.hiddenFactor = tf.constant(parameters["hiddenFactor"], dtype = tf.float32, shape=[], name="hiddenFactor")
		self.visibleFactor = tf.constant(parameters["visibleFactor"], dtype = tf.float32, shape=[], name="visibleFactor")
		self.regul = tf.constant(parameters["regularization"], dtype = tf.float32, shape=[], name="regularization")

		inputShape = self.input.get_shape()
		mask = self.bernoulli_tensor(self.keepProbTensor, self.input)

		with tf.name_scope("noise"):
			self.noisy_input = self.gaussianNoise(self.input, self.gaussianProbTensor, self.gaussianStd)
			self.noisy_input = self.minMaxNoise(self.noisy_input, self.mmProbTensor)
			self.noisy_input = tf.div(self.noisy_input, self.keepProbTensor) * mask
			self.noisy_input.set_shape(inputShape)

		with tf.name_scope("masks"):
			self.hidden_mask = tf.logical_and(tf.equal(mask,0), tf.logical_not(self.input_mask))
			self.hidden_mask.set_shape(inputShape)

			self.visible_mask = tf.logical_and(tf.equal(mask,1), tf.logical_not(self.input_mask))
			self.visible_mask.set_shape(inputShape)

		with tf.name_scope("layers"):
			self.hidden = tf.nn.tanh(tf.add(tf.matmul(self.noisy_input, self.weights['W1']),self.weights['b1']))
			self.reconstruction = tf.nn.tanh(tf.add(tf.matmul(self.hidden, self.weights['W2']), self.weights['b2']))


		with tf.name_scope("loss"):
			self.hidden_cost = tf.reduce_sum(tf.squared_difference(tf.boolean_mask(self.input, self.hidden_mask),tf.boolean_mask(self.reconstruction, self.hidden_mask)))*self.hiddenFactor
			self.visible_cost = tf.reduce_sum(tf.squared_difference(tf.boolean_mask(self.input, self.visible_mask),tf.boolean_mask(self.reconstruction, self.visible_mask)))*self.visibleFactor
			self.regul_cost = self.regul * tf.add(tf.nn.l2_loss(self.weights['W1']),tf.nn.l2_loss(self.weights['W2']))
			self.cost = self.hidden_cost+self.visible_cost+self.regul_cost

		with tf.name_scope("optimizer"):
			self.optimizer = optimizer.minimize(self.cost, global_step=batch)
		
	def gaussianNoise(self, X, gaussianProb, std):

		if gaussianProb == 0:
			return X

		mask = self.bernoulli_tensor(gaussianProb, X)
		gaussianNoise = tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=std, seed=self.get_randSeed(), dtype=X.dtype) * \
mask
		return X + gaussianNoise


	def minMaxNoise(self, X, mmProb):
		if mmProb == 0:
			return X
		#Get the min and max of the current matrix
		minimum = tf.reduce_min(X)
		maximum = tf.reduce_max(X)

		#Set half of the values to the min
		mmProbHalf = 1 - mmProb/2.0
		binaryToMin = self.bernoulli_tensor( mmProbHalf, X)
		indicesToMin = tf.to_int32(tf.where(tf.equal(binaryToMin, tf.constant(1, dtype=X.dtype))))
		numIndices = tf.shape(indicesToMin)[0]

		X = tf.add(tf.multiply(X, binaryToMin),tf.scatter_nd(indicesToMin, tf.fill([numIndices], minimum), shape=tf.shape(X)))

		binaryToMax = self.bernoulli_tensor(mmProbHalf, X)
		indicesToMax = tf.to_int32(tf.where(tf.equal(binaryToMax, tf.constant(1, dtype=X.dtype))))
		numIndices = tf.shape(indicesToMax)[0]

		return tf.add(tf.multiply(X, binaryToMax),tf.scatter_nd(indicesToMax, tf.fill([numIndices], maximum), shape=tf.shape(X)))

	def _init_weights(self):
		weights = {}
		with tf.name_scope("hidden_layer"):
			weights["W1"] = tf.get_variable('W1', shape=[self.dim, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
			weights["b1"] = tf.get_variable('b1', shape=[self.n_hidden], initializer=tf.zeros_initializer())

		with tf.name_scope("reconstruction_layer"):
			weights["W2"] = tf.get_variable('W2', shape=[self.n_hidden, self.dim],initializer=tf.contrib.layers.xavier_initializer())
			weights["b2"] = tf.get_variable('b2', shape=[self.dim], initializer=tf.zeros_initializer())

		return weights		

	def fit(self, sess, input, input_mask):
		feedDict = {
			self.input:         input,
			self.keepProbTensor:     self.keepProb,
			self.gaussianProbTensor: self.gaussianProb,
			self.mmProbTensor:      self.mmProb,
			self.input_mask:    input_mask,
		}
		cost, opt = sess.run((self.cost, self.optimizer), feed_dict = feedDict)
		return cost


	def predict(self, sess, input):
		feedDict = {
			self.input:         input,
			self.mmProbTensor:      0.0,
			self.gaussianProbTensor: 0.0,
			self.keepProbTensor:     1.0,
		}
		predictions = sess.run((self.reconstruction), feed_dict=feedDict)
		return predictions	

	#Make sure that we get random values
	def get_randSeed(self):
		if self.randSeed is None:
			return None
		self.randSeed += 1
		return self.randSeed
