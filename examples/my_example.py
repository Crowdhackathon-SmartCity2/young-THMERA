import gym
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import gym_traffic
from gym.wrappers import Monitor
import gym
import numpy as np
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/usr/share/sumo/tools/')


import traci


monitor = True
env = gym.make('Traffic-Simple-gui-v0')
model = load_model('my_model_10000.h5')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    prev_action = 0
    m_avg_speeds = []
    for t in tqdm(range(1000)):
        #env.render()
        #print(observation)
        # action = env.action_space.sample()

        # if t%30==0:
        #     action = abs(prev_action-1)
        # prev_action = action
        o = observation
        o = np.reshape(o[0], (1, 36))
        # minmax = MinMaxScaler()
        # minmax.fit(o)
        # o = minmax.transform(o)
        preds = model.predict(o)
        action = np.argmax(preds)


        # action = 0
        # time.sleep(0.5)
        observation, reward, done, info = env.step(action)
        speed = traci.multientryexit.getLastStepMeanSpeed(env.env.env.detector)
        m_avg_speeds.append(speed)
        #print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    plt.plot(m_avg_speeds)
    plt.show()