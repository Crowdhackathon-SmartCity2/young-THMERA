import gym
import gym_traffic
#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import random
import numpy as np
from collections import deque

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from gym.wrappers import Monitor
import gym
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm

GAME = 'crowdhackathon' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


img_rows , img_cols = 1, 36
#Convert image into Black and white
img_channels = 4 #We stack 4 frames


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


monitor = True
env = gym.make('Traffic-Simple-gui-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    for t in tqdm(range(1000)):
        #env.render()
        #print(observation)
        action = env.action_space.sample()
        # action = 0
        time.sleep(0.5)
        observation, reward, done, info = env.step(action)
        #print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break