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

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('PointToPoint-v0',mode='T', T_sens = 180, obs_mode = "J")
env = Monitor(env,'monitor_2205_2')

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=400000,
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

for i in range(1, 31):
    model.learn(n_ep, eval_freq = 100, reset_num_timesteps = False, tb_log_name = "PPO2")
    model.save(model_dir + f"/2205_2/HA_PPOagent_2205_2_{i}_30")

