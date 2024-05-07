from sumolib import checkBinary
from stable_baselines3 import DDPG
import xml.etree.ElementTree as ET
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from env import SumoEnv  
import traci
import numpy as np
import time
import os, sys
import gym

def main_test():
    env = SumoEnv()
    env.reset(gui=True)
    steps = 200
    try:
        for _ in range (steps):
            state = env.get_state()
            traci.simulationStep()

    except Exception as e:
        print("An error occurred:", e)
    finally:
        # 无论发生什么，确保关闭仿真
        env.close()      

def main():
    env = SumoEnv()

    # 如果您的环境非向量化，但仍想要使用n_envs > 1，那么您需要使用这个
    # 它将会自动创建多个进程以及相应数量的环境
    # env = make_vec_env(lambda: env, n_envs=1)

    # 实例化噪声对象，给策略添加一定的探索噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # 实例化学习算法
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    # 训练模型
    model.learn(total_timesteps=10000, log_interval=10)

    # 保存模型（可选）
    model.save("ddpg_sumo_model")

    # # 测试模型
    # state = env.reset()
    # for step in range(1000):
    #     action, _ = model.predict(state, deterministic=True)
    #     state, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         state = env.reset()

    # # 关闭环境
    # env.close()

if __name__ == "__main__":

    main()
    
    print('----------------ALL ---------END-----------------------')
