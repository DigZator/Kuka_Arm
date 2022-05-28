import gym
import ArmEnv
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('PointToRandomPoint-v0',gui=False,mode='T', obs_mode = "T")
obs = env.reset()
for _ in range(5):
	obs = env.reset()
	for _ in range(3):
		obs, rew, done, _ = (env.step(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])))
		print(obs)
		print(rew)
	print("\n===\n")