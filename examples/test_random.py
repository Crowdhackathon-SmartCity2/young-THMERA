import gym
from keras.models import load_model
import random

from sklearn.preprocessing import MinMaxScaler

import gym_traffic
from gym.wrappers import Monitor
import gym
import numpy as np
import time
#env_m = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/usr/share/sumo/tools/')


import traci


def get_halting_cars(env):

    halting = []
    for lane in env.env.env.lanes:
        halting.append(traci.lane.getLastStepHaltingNumber(lane))

    return sum(halting)
monitor = True
# env_m = gym.make('Traffic-Simple-gui-v0')
env_r = gym.make('Traffic-Simple-gui-v0')
model = load_model('my_model_final692.h5')
if monitor:
    # env_m = Monitor(env_m, "output/traffic/simple/random", force=True)
    env_r = Monitor(env_r, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    # observation_m = env_m.reset()
    observation_r = env_r.reset()
    prev_action = 0
    m_avg_speeds_m = []
    m_avg_speeds_r = []

    for t in tqdm(range(400)):

        # a = random.uniform(0, 1)
        #
        # prev_action = a
        # q = [[prev_action,1-prev_action]]
        # # a_t = q.mean(axis=0)
        # max_Q = np.argmax(q, axis=1)
        # action_r = np.round(max_Q.mean())
        #
        # observation, reward, done, info = env_r.step(action_r)
        # speed_r = traci.multientryexit.getLastStepMeanSpeed(env_r.env.env.detector)
        # m_avg_speeds_r.append(speed_r)

        # a = random.uniform(0, 1)
        a = 1
        prev_action = a
        q = [[prev_action, 1 - prev_action]]
        max_Q = np.argmax(q, axis=1)
        action_r = int(max_Q[0])

        observation_r, reward, done, info = env_r.step(action_r)
        # speed_r = traci.multientryexit.getLastStepMeanSpeed(env_r.env.env.detector)
        speed_r = get_halting_cars(env_r)
        m_avg_speeds_r.append(speed_r)

        # plotting
        axes = plt.gca()
        axes.set_xlim([0, 400])
        axes.set_ylim([0., 15])

        plt.title('Adaptive Traffic Control')
        plt.ylabel('Average Speed (m/s)')
        plt.xlabel('Time (s)')

        # plt.plot(m_avg_speeds_m, color="blue")
        plt.plot(m_avg_speeds_r, color="red")
        plt.pause(0.05)
        # plt.show()

        #print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    # plt.plot(m_avg_speeds)
    plt.show()