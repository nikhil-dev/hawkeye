import tensorflow as tf

import tensorflow.contrib.eager as tfe
SEQUENCE_LEN = 5

class HawkeyeYOLO(tfe.Network):
	"""A convolutional neural net inspired by You Only Look Once (https://arxiv.org/abs/1506.02640)"""
	def __init__(self, data_format, im_size):
		super(HawkeyeYOLO, self).__init__(name='')
		if data_format == 'channels_first':
		  self._input_shape = [-1, 3, im_size[0], im_size[1]]
		else:
		  assert data_format == 'channels_last'
		  self._input_shape = [-1, im_size[0], im_size[1], 3]
		self.conv1 = self.track_layer(
			tf.layers.Conv2D(8, 20, data_format=data_format, activation=tf.nn.relu))
		self.conv2 = self.track_layer(
			tf.layers.Conv2D(16, 20, data_format=data_format, activation=tf.nn.relu))
		self.fc1 = self.track_layer(tf.layers.Dense(1024, activation=tf.nn.relu))
		self.num_output_neurons = (im_size[0]/10) * (im_size[1]/10) * 3 # (x, y, confidence)
		self.fc2 = self.track_layer(tf.layers.Dense(self.num_output_neurons))
		self.dropout = self.track_layer(tf.layers.Dropout(0.5))
		self.max_pool2d = self.track_layer(
			tf.layers.MaxPooling2D(
				pool_size=2, strides=2, padding='SAME', data_format=data_format))

	def call(self, inputs):
		x = tf.reshape(inputs, self._input_shape)
		x = self.conv1(x)
		x = self.max_pool2d(x)
		x = self.conv2(x)
		x = self.max_pool2d(x)
		x = tf.layers.flatten(x)
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x


class RNNHawkeye(tfe.Network):
	"""A recurrent neural net that first applies the HawkeyeYOLO network before the LSTM layer."""
	def __init__(self, keep_prob):
		super(RNNHawkeye, self).__init__(name='')
		self.keep_prob = keep_prob
		self.cell = self.track_layer(tf.nn.rnn_cell.BasicLSTMCell(128))
		self.relu = self.track_layer(tf.layers.Dense(540, activation=tf.nn.relu))

	def call(self, videos, hawkeye_net, sequence_length=SEQUENCE_LEN, training=False):
		# Flip the order of time step and batch so it becomes [time_step, batch, h, w, channel]
		frames = tf.transpose(videos, [1, 0, 2, 3, 4])

		# Initialize cell state and break up frames
		batch_size = int(frames.shape[1])
		outputs = []
		state = self.cell.zero_state(batch_size, tf.float32)
		frames = tf.unstack(frames, axis=0)

		# Iterate and get outputs
		for frame in frames:
			frame = hawkeye_net(frame)
			output, state = self.cell(frame, state)
			outputs.append(output)

		# Put back outputs across time steps into 1 tensor & perform dropout
		outputs = tf.stack(outputs, axis=0)
		if training:
			outputs = tf.nn.dropout(outputs, self.keep_prob)

		sequence_length_vec = tf.fill([batch_size], sequence_length)
		batch_range = [i for i in range(batch_size)]
		indices = tf.stack([sequence_length_vec - 1, batch_range], axis=1)
		hidden_states = tf.gather_nd(outputs, indices)
		res = self.relu(hidden_states)
		return res
