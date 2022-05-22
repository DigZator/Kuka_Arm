import gym
import ArmEnv
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC
import torch

env = gym.make('PointToPoint-v0',gui=True,record=False)
""" model = SAC('MlpPolicy',env,verbose=1,device='cuda')
model.load('model1.zip') """
policy_kwargs = dict(activation_fn=torch.nn.ReLU,net_arch=[512,512,256,128])
model = DDPG('MlpPolicy',env,verbose=1,policy_kwargs=policy_kwargs)
model.load('DDPG_model.zip')
obs = env.reset()
print('Observation:',obs)
dones = False
rews = []

count = 0
while(True):
    count += 1
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    rews.append(rewards)
    if(dones):
        break
    #print(count,end='\r')

env.close()
plt.plot(rews)
plt.savefig('rews.png')
