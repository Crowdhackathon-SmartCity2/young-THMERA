import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf

from keras import Sequential
from keras.initializers import normal
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import gym_traffic
from gym.wrappers import Monitor
import gym
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
monitor = True

CONFIG = 'nothreshold'
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 300  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 300  # number of previous transitions to remember
BATCH = 50  # size of minibatch
FRAME_PER_ACTION = 1

global model

INPUT_DIM = 28


def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Dense(1000, input_dim=INPUT_DIM, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    print("We finish building the model")
    return model

env = gym.make('Traffic-Simple-gui-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)

model = buildmodel()

for i_episode in range(500):
    observation = env.reset()

    D = deque()
    OBSERVE = OBSERVATION
    t = 0
    # DROP = 700
    DROP = 1

    y_actual_list = []
    y_pred_list = []
    r1_overall_reward_list = []
    ts_list = []
    prev_action =0

    observation = np.reshape(observation[0], (1, INPUT_DIM))
    minmax = MinMaxScaler()
    # observation = minmax.fit_transform(observation)

    for t in range(10000):

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        test = []


        if t>OBSERVE:
            q = model.predict(observation)
            # q = model.predict(observation)
        else:
            if t % 30 == 0:
                # a = random.uniform(0, 1)
                a = abs(1-prev_action)
            prev_action = a
            q= [[prev_action,1-prev_action]]
        # a_t = q.mean(axis=0)
        max_Q = np.argmax(q, axis=1)
        action_index = np.round(max_Q.mean())


        if t > OBSERVE and t % 1 == 0:
        # if t % 30 == 0:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, DROP, INPUT_DIM))  # 32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], inputs.shape[1], ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                test_t = minibatch[i][0]
                pred_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                test_t1 = minibatch[i][3]
                done = minibatch[i][4]

                # test_t = test_t[:DROP]
                # pred_t = pred_t[:DROP]
                # test_t1 = test_t1[:DROP]

                inputs[i:i + 1] = test_t  # I saved down s_t

                targets[i] = model.predict(test_t)  # Hitting each buttom probability
                # targets[i] = model.predict(test_t)  # Hitting each buttom probability
                # test_t1 = test_t1.fillna(d_mean)
                Q_sa = model.predict(test_t1)
                # Q_sa = model.predict(test_t1)

                if done:
                    targets[i][0, action_index] = reward_t
                else:

                    # targets[i][0, int(action_index)] = reward_t + GAMMA * np.max(Q_sa)
                    # targets[i] = reward_t / 2 + targets[i]
                    # targets[i][0, int(action_index)] = (1 - GAMMA) * reward_t * 1 / 2 + GAMMA * Q_sa[0][int(action_index)]
                    # targets[i][0,:] = (1 - GAMMA) * reward_t * 1 / 2 + GAMMA * Q_sa[0]
                    targets[i][0, int(action_index)] = reward_t + GAMMA * np.max(Q_sa)

            inputs = np.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
            targets = np.reshape(targets, (targets.shape[0] * targets.shape[1], targets.shape[2]))

            print("Inputs : " + str(inputs))
            print("Targets : " + str(targets))
            model.fit(inputs, targets, batch_size=1, epochs=20, verbose=2)

        print("q :" +str(q))

        o, reward, done, info = env.step(int(action_index))
            # pred_y = list(pred['y'].values)
            # y_actual_list.extend(actual_y)
            # y_pred_list.extend(pred_y)
            # overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
            # r1_overall_reward_list.append(overall_reward)
        ts_list.append(t)

        # if done:
                # fig = plt.figure(figsize=(12, 6))
                # plt.plot(ts_list, r1_overall_reward_list, c='blue')
                # plt.plot(ts_list, [0] * len(ts_list), c='red')
                # plt.title("Cumulative R value change for Univariate Ridge (technical_20)")
                # plt.ylim([-0.04, 0.04])
                # plt.xlim([850, 1850])
                # plt.show()
            # print("el fin ...", info["public_score"])
            # break
        print("Time :" + str(t))
        print("Reward : "+ str(reward))
        o = np.reshape(o[0], (1, INPUT_DIM))
        # o = minmax.transform(o)
        D.append((observation, int(action_index), reward, o, done))
        observation = o
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        t = t + 1

        if t==5000:
            model.save('my_model_5000.h5')  # creates a HDF5 file 'my_model.h5'
        #env.render()
        #print(observation)
        # action = env.action_space.sample()
        # action = 0
        # time.sleep(0.5)
        # observation, reward, done, info = env.step(action)
        #print "Reward: {}".format(reward)
        if done:
            model.save('my_model_10000.h5')  # creates a HDF5 file 'my_model.h5'
            # del model  # deletes the existing model
            print("Episode finished after {} timesteps".format(t+1))
            break