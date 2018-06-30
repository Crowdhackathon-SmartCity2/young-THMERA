import gym
import gym_traffic
from gym.wrappers import Monitor
import gym
import time
#env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
monitor = True
env = gym.make('Traffic-Simple-gui-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    prev_action = 0
    for t in tqdm(range(10000)):
        #env.render()
        #print(observation)
        # action = env.action_space.sample()

        if t%30==0:
            action = abs(prev_action-1)
        prev_action = action
        # action = 0
        # time.sleep(0.5)
        observation, reward, done, info = env.step(action)
        #print "Reward: {}".format(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break