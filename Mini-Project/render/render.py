import gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = make_atari_env('PongNoFrameskip-v4', n_envs=16, seed=3504351757)
env = VecFrameStack(env, n_stack=4)
model = DQN.load("PongNoFrameskip-v4",env=env)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
