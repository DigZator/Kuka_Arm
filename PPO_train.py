import gym
import ArmEnv
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import pickle

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
import os

model_dir = "models/PPO"
logdir = "logs"

date = 2905

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('PointToRandomPoint-v0',mode='T', obs_mode = "T")
env = Monitor(env,f'monitor/monitor_{date}_3')

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100000,
                             render=False)

#model = SAC('MlpPolicy',env,verbose=1,device='cuda')

policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])

model = PPO('MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device='cuda',
            tensorboard_log = logdir)

n_ep = 100000

for i in range(1, 61):
    model.learn(n_ep, eval_freq = 100, reset_num_timesteps = False, tb_log_name = f"PPO_{date}_3")
    model.save(model_dir + f"/{date}_3/HA_PPOagent_{date}_3_{i}_60")