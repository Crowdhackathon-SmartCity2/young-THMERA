import random
from collections import deque
import numpy as np
import gym
import tensorflow as tf

from keras import Sequential
from keras.initializers import normal
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import gym_traffic
from gym.wrappers import Monitor
import gym
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
monitor = False

CONFIG = 'nothreshold'
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 200  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 200  # number of previous transitions to remember
BATCH = 90  # size of minibatch


global model

INPUT_DIM = 16

def buildmodel():
    print("Now we build the model")

    model = MultiOutputRegressor(ExtraTreesRegressor(max_depth=6,
                                                          random_state=0))

    # model = xgb.sklearn.XGBClassifier(param)
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
    prev_action = random.uniform(0, 1)
    observation = observation[0]

    Q_sa = 0
    action_index = 0
    r_t = 0
    for t in range(500):

        if t>OBSERVE+1:
            q = model.predict(observation)
            # q = model.predict(observation)
        else:
            # if t % 30 == 0:
            a = random.uniform(0, 1)
                # a = abs(1-prev_action)
            prev_action = a
            q= [prev_action,1-prev_action]
        # a_t = q.mean(axis=0)
        max_Q = np.argmax(q)
        action_index = np.round(max_Q.mean())


        if t > OBSERVE and t % 1 == 0:
        # if t % 30 == 0:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, INPUT_DIM))  # 32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2
            Q_sa = np.zeros((ACTIONS))

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                test_t = minibatch[i][0]
                pred_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                test_t1 = minibatch[i][3]
                done = minibatch[i][4]

                inputs[i] = test_t  # I saved down s_t
                action_index = 1
                if t>OBSERVE+1:
                    targets[i] = model.predict(test_t)  # Hitting each buttom probability

                # targets[i] = model.predict(test_t)  # Hitting each buttom probability
                # test_t1 = test_t1.fillna(d_mean)
                    Q_sa = model.predict(test_t1)
                # Q_sa = model.predict(test_t1)

                if done:
                    targets[i][0, action_index] = reward_t
                else:
                    # if reward_t>200:
                    #     print(reward_t)
                    #     targets[i] = [0.4,0.4]
                    #     targets[i][int(action_index)] = 0.6
                    # else:
                    #     targets[i] = [0.6, 0.6]
                    #     targets[i][int(action_index)] = 0.4
                    # targets[i] = reward_t / 4 + targets[i]
                    # targets[i][int(action_index)] = (1 - GAMMA) * reward_t * 3 / 4 + GAMMA * Q_sa[0][int(action_index)]
                    targets[i][int(action_index)] = (1 - GAMMA) * reward_t * 3 / 4
                    # targets[i][0, int(action_index)] = reward_t + GAMMA * np.max(Q_sa)
                    # targets[i] = reward_t / 2 + targets[i]
                    # targets[i][0, int(action_index)] = (1 - GAMMA) * reward_t * 1 / 2 + GAMMA * Q_sa[0][int(action_index)]
                    # targets[i][0, int(action_index)] = (1 - GAMMA) * reward_t * 1 / 2 + GAMMA * np.max(Q_sa[0][int(action_index)])
                    # targets[i][0,:] = (1 - GAMMA) * reward_t * 1 / 2 + GAMMA * Q_sa[0]
                    # targets[i][0, int(action_index)] = reward_t + GAMMA * np.max(Q_sa)

            inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
            targets = np.reshape(targets, (targets.shape[0] , targets.shape[1]))

            print("Inputs : " + str(inputs))
            print("Targets : " + str(targets))

            model.fit(inputs, targets)
            # param = {}
            #
            # # use softmax multi-class classification
            # param['objective'] = 'multi:softprob'
            # # scale weight of positive examples
            # param['eta'] = 0.6
            # param['ntrees'] = 300
            # param['subsample'] = 0.93
            # param['max_depth'] = 2
            # param['silent'] = 1
            # param['n_jobs'] = 8
            # param['num_class'] = 2
            #
            # xg_train = xgb.DMatrix(inputs, label=targets)
            # # xg_test = xgb.DMatrix(X_test, label=y_test)
            # watchlist = [(xg_train, 'train')]
            # num_round = 30
            # model = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=6)
            # # preds = bst.predict(xg_test)
            # # probs = np.multiply(probs, preds)
            # # preds = np.array([np.argmax(prob) for prob in preds])


        print("q :" +str(q))

        o, reward, done, info = env.step(int(action_index))
            # pred_y = list(pred['y'].values)
            # y_actual_list.extend(actual_y)
            # y_pred_list.extend(pred_y)
            # overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
            # r1_overall_reward_list.append(overall_reward)

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
        reward = reward/100
        # o = minmax.fit_transform(o)
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