import tensorflow as tf
import tensorflow.contrib.eager as tfe

from constants import *

class HawkeyeYOLO(tfe.Network):
	"""A convolutional neural net inspired by You Only Look Once (https://arxiv.org/abs/1506.02640)"""
	def __init__(self, data_format, num_layer_1_filters, num_layer_2_filters, kernel_size):
		super(HawkeyeYOLO, self).__init__(name='')
		if data_format == 'channels_first':
		  self._input_shape = [-1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]]
		else:
		  assert data_format == 'channels_last'
		  self._input_shape = [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
		self.conv1 = self.track_layer(
			tf.layers.Conv2D(num_layer_1_filters, kernel_size, data_format=data_format, activation=tf.nn.relu))
		self.conv2 = self.track_layer(
			tf.layers.Conv2D(num_layer_2_filters, kernel_size, data_format=data_format, activation=tf.nn.relu))
		self.fc1 = self.track_layer(tf.layers.Dense(1024, activation=tf.nn.relu))
		self.fc2 = self.track_layer(tf.layers.Dense(NUM_OUTPUT_NEURONS))
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
	def __init__(self, keep_prob, data_format):
		super(RNNHawkeye, self).__init__(name='')
		if data_format == 'channels_first':
		  self.img_shape = [3, IMAGE_SIZE[0], IMAGE_SIZE[1]]
		else:
		  assert data_format == 'channels_last'
		  self.img_shape = [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
		self.keep_prob = keep_prob
		self.cell = self.track_layer(tf.nn.rnn_cell.BasicLSTMCell(128))
		self.relu = self.track_layer(tf.layers.Dense(NUM_OUTPUT_NEURONS, activation=tf.nn.relu))

	def call(self, videos, hawkeye_net, sequence_length):
		input_shape = [-1, sequence_length] + self.img_shape
		videos = tf.reshape(videos, input_shape)

		#  [batch_size, time_step, h, w, channel] -> [time_step, batch_size, h, w, channel]
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
			output = self.relu(output)
			output = tf.nn.dropout(output, self.keep_prob)
			outputs.append(output)
		
		outputs = tf.stack(outputs, axis=0)
		outputs = tf.transpose(outputs, [1, 0, 2])

		return outputs
