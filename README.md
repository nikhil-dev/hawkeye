# HawkeyeNet

## Motivation

A perception system for ball tracking in cricket and tennis. Watch these short videos to understand what a hawkeye does in [tennis](https://www.youtube.com/watch?v=XhQyVnwBXBs) and [cricket](https://www.youtube.com/watch?v=XwXhQ0yUbjw).

The above systems cost millions of dollars, and are only available to professional sports players at the world's top stadiums. The goal of this work is to bring a (less powerful) version of this system to everyday recreational players via a smartphone camera.


## Technical Details

The model consists of a recurrent neural network augmented with a [YOLO](https://arxiv.org/abs/1506.02640)-like network. The YOLO-like network is pre-trained by framing ball tracking as a regression problem. The recurrent units enable us to take advantage of spatio-temporal properties of ball tracking in sports.

The training is performed by using a fixed set of base images and generating stochastic ball trajectory sequences (similar to data augmentation). The thus generated ```(image, label)``` frames are used for training.

The system can track cricket and tennis balls using a smartphone camera (much lower resolution than professional sports cameras used for Hawkeye systems in stadiums), from a distance, in the presence of motion blur, varied backgrounds and occlusion, while being able to run on a mobile device.

This model has been ported to Tensorflow Eager, which gives us a better Pythonic expression of the model (define-by-run as opposed to define-and-run) and makes it easier to debug in many cases.

## Instructions to run locally

1. Collect a dataset with images that contain backgrounds representative of your use case (I trained it on a variety of backgrounds. I will be releasing the dataset I used soon).
2. ```pip install tf-nightly``` (we need nightly builds for TF Eager unless you have >=1.5)
3. ```python main.py --pretrain-cnn``` to pretrain they YOLO like network. ```python main.py``` to train the whole system.


## Future work

1. Train across multiple sports to see if we can get a better overall model.

2. I will be working on building a UI and deploying this to an Android app to take it out to the field and use it in my cricket matches. Once I've fine-tuned it, the goal is to publish this and make it accessible in the App Store.
