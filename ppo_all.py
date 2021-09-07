
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from collections import deque
import random
import time

import gym
import os
import h5py

from env.env_continue import ENV_CONTINUE

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model_ppo')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def plot_results(log_folder):
    from scipy.signal import savgol_filter
    R = load_results(log_folder)['r']
    T = load_results(log_folder)['t']
    # _w = 7
    # _window_size = len(R) // _w if (len(R) // _w) % 2 != 0 else len(R) // _w + 1
    # filtered = savgol_filter(R, _window_size, 1)

    #plt.title('smoothed returns')
    plt.title('returns')
    plt.ylabel('Returns')
    plt.xlabel('time step')
    plt.plot(T, R)
    plt.grid()
    plt.show()


def train_ppo():

    log_dir = f"model_save/"
    env = ENV_CONTINUE(istest=False)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                clip_obs=10.)

    model = PPO('MlpPolicy', env, verbose=1, batch_size=2048, seed=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=480, log_dir=log_dir)
    model.learn(total_timesteps=int(1000000), callback = callback, log_interval = 480)
    model.save('model_save/ppo')

def test_ppo():
    log_dir = f"model_save/best_model_ppo"
    env = ENV_CONTINUE(istest=True)
    env.render = True
    env = Monitor(env, log_dir)
    model = PPO.load(log_dir)
    plot_results(f"model_save/")
    for i in range(10):
        state = env.reset()
        day = 0
        while True:
            action = model.predict(state)
            next_state, reward, done, info = env.step(action[0])
            state = next_state
            # print("trying:",day,"reward:", reward,"now profit:",env.profit)
            day+=1
            if done:
                print('stock',i,' total profit=',env.profit,' buy hold=',env.buy_hold)
                break

if __name__ == '__main__':
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    train_ppo()
    test_ppo()
