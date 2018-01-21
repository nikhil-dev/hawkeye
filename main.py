import argparse
import functools
import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

from data_utils import *
from models import HawkeyeYOLO, RNNHawkeye
from constants import *


def loss(outputs, targets):
  """ L2 loss between outputs and targets"""
  assert np.array_equal(tf.shape(targets).numpy(), tf.shape(outputs).numpy())
  return tf.reduce_mean(tf.square(tf.subtract(outputs, targets)))

def cnn_loss(outputs, targets):
	targets = tf.reshape(targets, [-1, NUM_OUTPUT_NEURONS])
	return loss(outputs, targets)

def rnn_loss(outputs, targets):
	targets = tf.reshape(targets, [-1, SEQUENCE_LEN, NUM_OUTPUT_NEURONS])
	return loss(outputs, targets)

def cnn_test(model, eval_data):
  avg_loss = tfe.metrics.Mean("cnn_loss")
  for (images, labels) in tfe.Iterator(eval_data):
	predictions = model(images)
	avg_loss(cnn_loss(predictions, labels))
  print("eval/cnn_loss: %.6f\n" % avg_loss.result())
  with tf.contrib.summary.always_record_summaries():
	tf.contrib.summary.scalar("cnn_loss", avg_loss.result())

def train_cnn_one_epoch(model, optimizer, dataset, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""
  tf.train.get_or_create_global_step()

  def model_loss(labels, images):
	prediction = model(images)
	loss_value = cnn_loss(prediction, labels)
	print loss_value
	return loss_value

  for x in (tfe.Iterator(dataset)):
	images, labels = x
	with tf.contrib.summary.record_summaries_every_n_global_steps(10):
	  batch_model_loss = functools.partial(model_loss, labels, images)
	  optimizer.minimize(
		  batch_model_loss, global_step=tf.train.get_global_step())


def train_rnn_one_epoch(model, optimizer, dataset, hawkeye_net, log_interval=None):
  tf.train.get_or_create_global_step()

  def model_loss(targets, videos, hawkeye_net):
	prediction = model(videos, hawkeye_net, SEQUENCE_LEN)
	loss_value = rnn_loss(prediction, targets)
	print loss_value
	return loss_value

  for (batch, (videos, targets)) in enumerate(tfe.Iterator(dataset)):
	#with tf.contrib.summary.record_summaries_every_n_global_steps(log_interval):
	batch_model_loss = functools.partial(model_loss, targets, videos, hawkeye_net)
	optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())

def rnn_test(model, eval_data, hawkeye_net):
	avg_loss = tfe.metrics.Mean("loss")
	for (videos, targets) in tfe.Iterator(eval_data):
		prediction = model(videos, hawkeye_net, SEQUENCE_LEN)
		avg_loss(rnn_loss(prediction, targets))
	print("eval/loss: %.6f\n" % avg_loss.result())
	with tf.contrib.summary.always_record_summaries():
		tf.contrib.summary.scalar('training/hptuning/metric', avg_loss.result())

def main(FLAGS):
  # Decide device and data format
  '''
  (device, data_format) = ('/gpu:0', 'channels_first')
  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
	(device, data_format) = ('/cpu:0', 'channels_last')

  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
	device = "/cpu:0"
  else:
	device = "/gpu:0"
  '''
  (device, data_format) = ('/cpu:0', 'channels_last')

  # Setup model, optimizer, logging
  hawkeye_net = HawkeyeYOLO(data_format=data_format,
	num_layer_1_filters=FLAGS.num_layer_1_filters,
	num_layer_2_filters=FLAGS.num_layer_2_filters,
	kernel_size=FLAGS.kernel_size)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  log_dir = os.path.join(FLAGS.dir, "summaries")
  tf.gfile.MakeDirs(log_dir)
  checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
  
  if FLAGS.pretrain_cnn:
	# Prepare data
	data_dir = FLAGS.dir
	train_data = load_training_dataset(data_dir)
	eval_data = load_eval_dataset(data_dir)
	
	# Batch the dataset
	train_data = train_data.batch(FLAGS.batch_size)

	train_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "cnn_train"), flush_millis=10000)
	test_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "cnn_eval"), flush_millis=10000, name="eval")

	# Run training for specified epochs
	with tf.device(device):
		with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):
			for epoch in range(FLAGS.num_epochs):
				start = time.time()
				with train_summary_writer.as_default():
					train_cnn_one_epoch(hawkeye_net, optimizer, train_data, 
						log_interval=FLAGS.log_interval)
				end = time.time()
				print("train/time for epoch #%d: %.2f" % (epoch, end - start))
			with test_summary_writer.as_default():
				pass
				#cnn_test(hawkeye_net, eval_data)
		
		# Save variables
		global_step = tf.train.get_or_create_global_step()
		all_variables = (
			hawkeye_net.variables
			+ optimizer.variables()
			+ [global_step])
		tfe.Saver(all_variables).save(
		  checkpoint_prefix, global_step=global_step)
  else:
	# Prepare data
	data_dir = FLAGS.dir
	train_data = load_sequence_training_dataset(data_dir)
	eval_data = load_sequence_eval_dataset(data_dir)
	
	# Batch the dataset
	train_data = train_data.batch(FLAGS.batch_size)

	# Train RNN
	model = RNNHawkeye(keep_prob=FLAGS.keep_probability, data_format=data_format)
	train_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "rnn_train"), flush_millis=10000)
	test_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "rnn_eval"), flush_millis=10000, name="eval")

	# Run training for specified epochs
	with tf.device(device):
		with tfe.restore_variables_on_create(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)):
			for epoch in range(FLAGS.num_epochs):
				start = time.time()
				with train_summary_writer.as_default():
				  train_rnn_one_epoch(model, optimizer, train_data,
					hawkeye_net, FLAGS.log_interval)
				end = time.time()
				print("train/time for epoch #%d: %.2f" % (epoch, end - start))
			with test_summary_writer.as_default():
				pass
				#rnn_test(model, eval_data, hawkeye_net)
		
		# Save variables
		global_step = tf.train.get_or_create_global_step()
		all_variables = (
		  hawkeye_net.variables
		  + model.variables
		  + optimizer.variables()
		  + [global_step])
		tfe.Saver(all_variables).save(
		  checkpoint_prefix, global_step=global_step)

if __name__ == "__main__":
  tfe.enable_eager_execution()
  parser = argparse.ArgumentParser()
  parser.add_argument(
	  "--dir",
	  type=str,
	  default="/Users/rambler/hawkeye/dataset/",
	  help="Directory to access data files and save logs.")
  parser.add_argument(
	  "--checkpoint_dir",
	  type=str,
	  default="/Users/rambler/hawkeye/checkpoints/",
	  help="Directory to save model params.")
  parser.add_argument(
	  "--log_interval",
	  type=int,
	  default=10,
	  metavar="N",
	  help="Log training loss every log_interval batches.")
  parser.add_argument(
	  "--num_epochs",
	  type=int,
	  default=1,
	  help="Number of epochs to train.")
  parser.add_argument(
	  "--batch_size",
	  type=int,
	  default=128,
	  help="Batch size for training and eval.")
  parser.add_argument(
	  "--keep_probability",
	  type=float,
	  default=0.5,
	  help="Keep probability for dropout between LSTM layers.")
  parser.add_argument(
	  "--learning_rate",
	  type=float,
	  default=0.01,
	  help="Learning rate to be used during training.")
  parser.add_argument(
	  "--no_gpu",
	  action="store_true",
	  default=False,
	  help="Disables GPU usage even if a GPU is available.")
  parser.add_argument(
	  "--pretrain-cnn",
	  action="store_true",
	  default=False,
	  help="Trains the HawkeyeYOLO CNN instead of the RNN. Both "
		   "networks cannot be trained at the same time.")
  parser.add_argument(
	  "--kernel-size",
	  type=int,
	  default=20,
	  help="Kernel size to use for convolutional layers.")
  parser.add_argument(
	  "--num-layer-1-filters",
	  type=int,
	  default=8,
	  help="Number of convolutional filters to use in the first convolutional layer.")
  parser.add_argument(
	  "--num-layer-2-filters",
	  type=int,
	  default=16,
	  help="Number of convolutional filters to use in the second convolutional layer.")


  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)
