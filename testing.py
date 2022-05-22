import gym
import ArmEnv
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


""" env = gym.make('PointToPoint-v0',gui=True,mode='T')
env.reset()

rews = []

print(env.observation_space)
print(env.action_space)

for i in range(100):
    action = env.action_space.sample()
    _, rew, _, _ = env.step(action)
    rews.append(rew)
    time.sleep(0.25)
    print(action,rew) """

env = gym.make('PointToPoint-v0',gui=True,mode='T', obs_mode = "CT")
#model = PPO('MlpPolicy',env,verbose=1,device='cuda')
model = PPO.load('models/PPO/2205_1/HA_PPOagent_2205_1_20_30.zip')
obs = env.reset()
print('Observation:',obs)
dones = False
rews = []

count = 0
for _ in range(5):
    obs = env.reset()
    for _ in range(500):
        count += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #print(action,rewards)
        rews.append(rewards)
        if(dones):
            break
    #print(count,end='\r')

    print("Cumulative REWARD:",sum(rews))