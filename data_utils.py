import random
import os
import functools
import pdb
import sys

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skvideo.io

from constants import *

"""This module contains utilities for generating training data using data augmentation"""

def get_initial_pt_and_increments():
	""" Generates an initial point in one of the four quarters of the image """
	i = random.randint(1, 4)
	if i == 1: # start at the top left
		initial_pt = (random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[0]/2),
				random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[1]/2))
		increments = ([5, 10], [7, 15])
	elif i == 2: # start at the bottom left
		initial_pt = (random.randint(IMAGE_SIZE[0]/2, IMAGE_SIZE[0] - ALLOWANCE_FROM_EDGE),
			random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[1]/2))
		increments = ([-10, -5], [7, 15])
	elif i == 3: # start at the bottom right
		initial_pt = (random.randint(IMAGE_SIZE[0]/2, IMAGE_SIZE[0] - ALLOWANCE_FROM_EDGE),
			random.randint(IMAGE_SIZE[1]/2, IMAGE_SIZE[1] - ALLOWANCE_FROM_EDGE))
		increments = ([-10, -5], [-15, -7])
	else: # start at the top right
		initial_pt = (random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[0]/2),
			random.randint(IMAGE_SIZE[1]/2, IMAGE_SIZE[1] - ALLOWANCE_FROM_EDGE))
		increments = ([5,10], [-15, -7])
	return initial_pt, increments


def is_good_trajectory(traj):
	if len(traj) != SEQUENCE_LEN:
		return False
	for height, width in traj:
		if height < ALLOWANCE_FROM_EDGE or height > IMAGE_SIZE[0] - ALLOWANCE_FROM_EDGE:
			return False
		if width < ALLOWANCE_FROM_EDGE or width > IMAGE_SIZE[1] - ALLOWANCE_FROM_EDGE:
			return False
	return True

def generate_ball_location_sequence():
	traj = []
	while not is_good_trajectory(traj):
		traj = []
		initial_pt, inc = get_initial_pt_and_increments()
		traj.append(initial_pt)
		while len(traj) < SEQUENCE_LEN:
			next_height = traj[-1][0] + random.randint(inc[0][0], inc[0][1])
			next_width = traj[-1][1] + random.randint(inc[1][0], inc[1][1])
			traj.append((next_height, next_width))
	return traj

def load_sequence_training_dataset(data_dir):
	ball_images = load_ball_images_to_memory(data_dir)
	gen = functools.partial(sequence_data_generator, data_dir, ball_images)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def load_sequence_eval_dataset(data_dir):
	ball_images = load_ball_images_to_memory(data_dir)
	gen = functools.partial(sequence_data_generator, data_dir, ball_images)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def sequence_data_generator(data_dir, ball_images):
	videogen = skvideo.io.vreader(os.path.join(data_dir, '1.mp4'))
	total_vids_produced = 0
	frames = []
	for frame in videogen:
		if total_vids_produced >= NUM_TRAIN_VIDEOS:
			return
		while len(frames) < SEQUENCE_LEN:
			frames.append(frame)
			continue
		assert len(frames) == SEQUENCE_LEN
		sequence = []
		ball_image = random.choice(ball_images)
		locations = generate_ball_location_sequence()
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
				dense_shape=[IMAGE_SIZE[0]/10, IMAGE_SIZE[1]/10, 3])
			target = tf.sparse_tensor_to_dense(target)
			target = tf.reshape(target, [-1])
			loc_sequence.append(target)
		total_vids_produced += 1
		frames.clear()
		yield tf.convert_to_tensor(sequence), tf.convert_to_tensor(loc_sequence)

def load_training_dataset(data_dir):
	"""Training data for CNN YOLO-like network"""
	ball_images = load_ball_images_to_memory(data_dir)
	gen = functools.partial(data_generator, data_dir, ball_images)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def load_eval_dataset(data_dir):
	"""Eval data for CNN YOLO-like network"""
	ball_images = load_ball_images_to_memory(data_dir)
	gen = functools.partial(data_generator, data_dir, ball_images)
	return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))

def data_generator(data_dir, ball_images):
	"""Data generator for CNN YOLO-like network"""
	videogen = skvideo.io.vreader(os.path.join(data_dir, '1.mp4'))
	for i, base_image in enumerate(videogen):
		if i >= NUM_TRAIN_IMAGES:
			return
		base_image = tf.convert_to_tensor(base_image)
		base_image = tf.image.convert_image_dtype(base_image,  tf.float32)
		if i < SKIP_FIRST_N_FRAMES:
			continue
		ball_image = tf.image.resize_images(random.choice(ball_images), random.choice(BALL_IMG_SIZES))
		ball_loc = [random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[0] - ALLOWANCE_FROM_EDGE),
			random.randint(ALLOWANCE_FROM_EDGE, IMAGE_SIZE[1] - ALLOWANCE_FROM_EDGE)]
		h = ball_loc[0]
		w = ball_loc[1]
		indices = [[h/10, w/18, 0], [h/10, w/18, 1], [h/10, w/18, 2]]
		values = [h, w, FULL_CONFIDENCE]
		target = tf.SparseTensor(
			indices=indices,
			values=values,
			dense_shape=[IMAGE_SIZE[0]/10, IMAGE_SIZE[1]/10, 3])
		target = tf.sparse_tensor_to_dense(target)
		target = tf.reshape(target, [-1])
		final_img = place_ball_on_image(base_image, ball_image, ball_loc)
		yield final_img, target

def place_ball_on_image(image, ball, loc): # loc = (height, width)
	# Compute the padding
	# save_tensor_image(ball, 'ball')
	# save_tensor_image(image, 'image')
	image_dim = tf.shape(image)
	ball_dim = tf.shape(ball)

	# Add some noise to the ball's color
	ball = tf.multiply(ball, random.uniform(0.0, 1.))
	
	# Compute padding for ball
	padding_top = loc[0] - ball_dim[0]/2
	padding_bottom = image_dim[0] - padding_top - ball_dim[0]
	padding_left = loc[1] - ball_dim[1]/2
	padding_right = image_dim[1] - padding_left - ball_dim[1]
	padding = tf.convert_to_tensor([[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]])
	padded_ball = tf.pad(ball, padding)

	# Pad ball with image and add noise to ball
	background_fraction = random.uniform(0.0, 1.0)
	mask = tf.pad(tf.fill(tf.shape(ball), background_fraction), padding, constant_values=1)
	assert(np.array_equal(tf.shape(image).numpy(), tf.shape(mask).numpy()))
	assert(np.array_equal(tf.shape(image).numpy(), tf.shape(padded_ball).numpy()))
	res = tf.add(tf.multiply(image, mask), padded_ball * (1. - background_fraction))
	# save_tensor_image(res, 'final')
	# sys.exit()
	return res

def load_base_images_to_memory(data_dir, im_range):
	images = []
	for i in im_range:
		path = os.path.join(data_dir, str(i)) + ".jpg"
		img = read_jpg_image(path)
		images.append(img)
	return images

def load_ball_images_to_memory(data_dir):
	images = []
	for i in BALL_RANGE:
		path = os.path.join(data_dir, "ball_" + str(i) + ".png")
		images.append(read_ball_img(path))
	return images

def read_ball_img(image_path):
	""" Read image from path and return as a tensor of shape [h, w, channels] with the [h, w] == IMAGE_SIZE of type float32 """
	# tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png

	image = tf.image.decode_png(tf.read_file(image_path), channels=3)
	image = tf.image.convert_image_dtype(image,  tf.float32)
	return image

def save_tensor_image(img_tensor, str):
	assert len(tf.shape(img_tensor).numpy()) == 3
	file_path = os.path.join('/Users/rambler/Desktop/', str + '.jpeg')
	if os.path.exists(file_path):
		return
	img_tensor = tf.image.convert_image_dtype(img_tensor,  tf.uint8)
	img = Image.fromarray(img_tensor.numpy(), "RGB")
	img.save(file_path)
