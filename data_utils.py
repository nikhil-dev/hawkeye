import tensorflow as tf
import tensorflow.contrib.eager as tfe
import random
import os
import functools
import pdb

BALL_IMG_SIZE = [20,20]
BALL_IMG_SIZES = [[3,3],[6,6],[9,9],[12,12],[15,15]]
IM_RANGE = range(1,11)
random.shuffle(IM_RANGE)
TRAIN_RANGE = IM_RANGE[:1]
TEST_RANGE = IM_RANGE[8:]
BALL_RANGE = range(1,3)
FULL_CONFIDENCE = 1
SEQUENCE_LEN = 5
ALLOWANCE_FROM_EDGE = 20


"""This module contains utilities for generating training data using data augmentation"""

def is_good_trajectory(traj, sequence_len, im_size):
	if len(traj) != sequence_len:
		return False
	for height, width in traj:
		if height < ALLOWANCE_FROM_EDGE or height > im_size[0] - ALLOWANCE_FROM_EDGE:
			return False
		if width < ALLOWANCE_FROM_EDGE or width > im_size[1] - ALLOWANCE_FROM_EDGE:
			return False
	return True

def generate_ball_location_sequence(sequence_len, im_size):
	traj = []
	while not is_good_trajectory(traj, sequence_len, im_size):
		initial_pt = (random.randint(ALLOWANCE_FROM_EDGE, im_size[0] - ALLOWANCE_FROM_EDGE),
				random.randint(ALLOWANCE_FROM_EDGE, im_size[1] - ALLOWANCE_FROM_EDGE))
		traj.append(initial_pt)
		while len(traj) < sequence_len:
			next_height = traj[-1][0] + random.randint(-2,2)
			next_width = traj[-1][1] + random.randint(-4,4)
			traj.append((next_height, next_width))
	return traj

def load_sequence_training_dataset(data_dir, im_size):
	base_images = load_base_images_to_memory(data_dir, im_size, TRAIN_RANGE)
	ball_images = load_ball_images_to_memory(data_dir, BALL_IMG_SIZE)
	gen = functools.partial(sequence_data_generator, base_images, ball_images, im_size)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def load_sequence_eval_dataset(data_dir, im_size):
	base_images = load_base_images_to_memory(data_dir, im_size, TEST_RANGE)
	ball_images = load_ball_images_to_memory(data_dir, BALL_IMG_SIZE)
	gen = functools.partial(sequence_data_generator, base_images, ball_images, im_size)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def sequence_data_generator(base_images, ball_images, im_size):
	num_output_neurons = (im_size[0]/10) * (im_size[1]/10) * 3
	for base_image in base_images:
		ball_image = random.choice(ball_images)
		sequence_len = SEQUENCE_LEN
		sequence = []
		locations = generate_ball_location_sequence(sequence_len, im_size)
		loc_sequence = []
		for ball_loc in locations:
			bm = tf.image.resize_images(ball_image, random.choice(BALL_IMG_SIZES))
			sequence.append(place_ball_on_image(base_image, bm, ball_loc))
			h = ball_loc[0]
			w = ball_loc[1]
			indices = [[h/10, w/18, 0], [h/10, w/18, 1], [h/10, w/18, 2]]
			values = [h, w, FULL_CONFIDENCE]
			target = tf.SparseTensor(
				indices=indices,
				values=values,
				dense_shape=[im_size[0]/10, im_size[1]/10, 3])
			target = tf.sparse_tensor_to_dense(target)
			target = tf.reshape(target, [-1])
			loc_sequence.append(target)
		yield tf.convert_to_tensor(sequence), tf.convert_to_tensor(loc_sequence)

def load_training_dataset(data_dir, im_size):
	"""Training data for CNN YOLO-like network"""
	base_images = load_base_images_to_memory(data_dir, im_size, TRAIN_RANGE)
	ball_images = load_ball_images_to_memory(data_dir, BALL_IMG_SIZE)
	gen = functools.partial(data_generator, base_images, ball_images, im_size)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def load_eval_dataset(data_dir, im_size):
	"""Eval data for CNN YOLO-like network"""
	base_images = load_base_images_to_memory(data_dir, im_size, TEST_RANGE)
	ball_images = load_ball_images_to_memory(data_dir, BALL_IMG_SIZE)
	gen = functools.partial(data_generator, base_images, ball_images, im_size)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def data_generator(base_images, ball_images, im_size):
	"""Data generator for CNN YOLO-like network"""
	for _ in range(30):
		for base_image in base_images:
			ball_image = tf.image.resize_images(random.choice(ball_images), random.choice(BALL_IMG_SIZES))
			ball_loc = [random.randint(ALLOWANCE_FROM_EDGE, im_size[0] - ALLOWANCE_FROM_EDGE),
				random.randint(ALLOWANCE_FROM_EDGE, im_size[1] - ALLOWANCE_FROM_EDGE)]
			h = ball_loc[0]
			w = ball_loc[1]
			indices = [[h/10, w/18, 0], [h/10, w/18, 1], [h/10, w/18, 2]]
			values = [h, w, FULL_CONFIDENCE]
			target = tf.SparseTensor(
				indices=indices,
				values=values,
				dense_shape=[im_size[0]/10, im_size[1]/10, 3])
			target = tf.sparse_tensor_to_dense(target)
			target = tf.reshape(target, [-1])
			final_img = place_ball_on_image(base_image, ball_image, ball_loc)
			yield final_img, target

def place_ball_on_image(image, ball, loc): # loc = (height, width)
	# Compute the padding
	image_dim = tf.shape(image)
	ball_dim = tf.shape(ball)

	# Compute padding for ball
	padding_top = loc[0] - ball_dim[0]/2
	padding_bottom = image_dim[0] - padding_top - ball_dim[0]
	padding_left = loc[1] - ball_dim[1]/2
	padding_right = image_dim[1] - padding_left - ball_dim[1]
	padding = tf.convert_to_tensor([[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]])
	padded_ball = tf.pad(ball, padding)
	
	# Pad ball with image
	mask = tf.pad(tf.zeros(tf.shape(ball)), padding, constant_values=1)
	res = tf.add(tf.multiply(image, mask), padded_ball)
	return res

def load_base_images_to_memory(data_dir, im_size, im_range):
	images = []
	for i in im_range:
		path = os.path.join(data_dir, str(i)) + ".jpg"
		img = read_image_from_disk(path, im_size)
		images.append(img)
	return images

def load_ball_images_to_memory(data_dir, im_size):
	images = []
	for i in range(1, 3):
		path = os.path.join(data_dir, "ball_" + str(i) + ".jpg")
		images.append(read_image_from_disk(path, im_size))
	return images

def read_image_from_disk(image_path, im_size):
	# tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
	path_length = len(image_path)
	file_extension = tf.substr(image_path, path_length - 3, 3)
	file_cond = tf.equal(file_extension, 'jpg')

	image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))
	image  = tf.image.convert_image_dtype(image,  tf.float32)
	image  = tf.image.resize_images(image, im_size, tf.image.ResizeMethod.AREA)

	return image