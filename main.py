import argparse
import functools
import os
import sys
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from data_utils import load_training_dataset, load_eval_dataset
from data_utils import load_sequence_training_dataset, load_sequence_eval_dataset
from models import HawkeyeYOLO, RNNHawkeye

IMAGE_SIZE = [100, 180]
SEQUENCE_LEN = 5


def loss(outputs, targets):
  return tf.reduce_mean(tf.square(tf.subtract(outputs, targets)))

def cnn_test(model, eval_data):
  avg_loss = tfe.metrics.Mean("loss")
  for (images, labels) in tfe.Iterator(eval_data):
	predictions = model(images)
	avg_loss(loss(labels, predictions))
  print("eval/loss: %.6f\n" % avg_loss.result())
  with tf.contrib.summary.always_record_summaries():
	tf.contrib.summary.scalar("loss", avg_loss.result())

def train_cnn_one_epoch(model, optimizer, dataset, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  tf.train.get_or_create_global_step()

  def model_loss(labels, images):
	prediction = model(images)
	loss_value = loss(prediction, labels)
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
	prediction = model(videos, hawkeye_net)
	loss_value = loss(prediction, targets)
	return loss_value

  for (batch, (videos, targets)) in enumerate(tfe.Iterator(dataset)):
	#with tf.contrib.summary.record_summaries_every_n_global_steps(log_interval):
	batch_model_loss = functools.partial(model_loss, targets, videos, hawkeye_net)
	optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())


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
  print("Using device %s." % device)
  '''
  (device, data_format) = ('/cpu:0', 'channels_last')

  # Setup model, optimizer, logging
  hawkeye_net = HawkeyeYOLO(data_format=data_format, im_size=IMAGE_SIZE)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  log_dir = os.path.join(FLAGS.dir, "summaries")
  tf.gfile.MakeDirs(log_dir)
  
  if FLAGS.pretrain_cnn:
	# Prepare data
	data_dir = FLAGS.dir
	train_data = load_training_dataset(data_dir, IMAGE_SIZE)
	eval_data = load_eval_dataset(data_dir, IMAGE_SIZE)
	
	# Batch the dataset
	train_data = train_data.batch(FLAGS.batch_size)

	train_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "cnn_train"), flush_millis=10000)
	test_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "cnn_eval"), flush_millis=10000, name="eval")

	# Run training for specified epochs
	with tf.device(device):
	  for epoch in range(FLAGS.num_epochs):
		start = time.time()
		with train_summary_writer.as_default():
		  train_cnn_one_epoch(hawkeye_net, optimizer, train_data, log_interval=FLAGS.log_interval)
		end = time.time()
		print("train/time for epoch #%d: %.2f" % (epoch, end - start))
		with test_summary_writer.as_default():
		  cnn_test(hawkeye_net, eval_data)
  else:
	# Prepare data
	data_dir = FLAGS.dir
	train_data = load_sequence_training_dataset(data_dir, IMAGE_SIZE)
	eval_data = load_sequence_eval_dataset(data_dir, IMAGE_SIZE)
	
	# Batch the dataset
	train_data = train_data.batch(FLAGS.batch_size)

	model = RNNHawkeye(keep_prob=FLAGS.keep_probability)
	# Train RNN using output from CNN as input
	train_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "rnn_train"), flush_millis=10000)
	test_summary_writer = tf.contrib.summary.create_file_writer(
		os.path.join(log_dir, "rnn_eval"), flush_millis=10000, name="eval")

	# Run training for specified epochs
	with tf.device(device):
	  for epoch in range(FLAGS.num_epochs):
		start = time.time()
		with train_summary_writer.as_default():
		  train_rnn_one_epoch(model, optimizer, train_data, hawkeye_net, FLAGS.log_interval)
		end = time.time()
		print("train/time for epoch #%d: %.2f" % (epoch, end - start))

if __name__ == "__main__":
  tfe.enable_eager_execution()
  parser = argparse.ArgumentParser()
  parser.add_argument(
	  "--dir",
	  type=str,
	  default="/Users/rambler/Desktop/cricket/",
	  help="Directory to access data files and save logs.")
  parser.add_argument(
	  "--log_interval",
	  type=int,
	  default=10,
	  metavar="N",
	  help="Log training loss every log_interval batches.")
  parser.add_argument(
	  "--num_epochs", type=int, default=20, help="Number of epochs to train.")
  parser.add_argument(
	  "--batch_size",
	  type=int,
	  default=64,
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

  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)
