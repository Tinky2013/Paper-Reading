
from stable_baselines3 import DQN

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


class ENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        df = pd.read_csv(r'hd5_sz50/2019_sz50.csv')
        sdata = '2015-01-01'
        edata = '2020-12-30'
        df = df.set_index('time')
        df = df[(df.index >= sdata) & (df.index <= edata)]
        df = df.sort_index(ascending=True)

        close = df['CLOSE_AFTER']

        self.stock_list = df['thscode']
        self.close = close
        self.action_space = gym.spaces.Discrete(5)  # (0,1,2,3,4)
        self.observation_space = gym.spaces.Box(
            low=np.array([-5] * 31),
            high=np.array([5] * 31)
        )

        self.seq_time = 180

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = close

    def reset(self):
        self.dt = self.data_train[self.stock_list == '601336.SH']
        self.dt = np.array(self.dt.iloc[:, 2:])
        self.close1 = self.close_train[self.stock_list == '601336.SH']

        self.inventory = 0
        self.initial_money = 1000000
        self.total_money = 1000000
        self.trade_date = np.random.randint(0, len(self.close1) - self.seq_time)
        Portfolio_unit = 1
        Rest_unit = 1
        self.t = 0
        state = self.dt[self.trade_date + self.t]  # 原始state就是数据的其他特征
        add_state = np.array([Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])
        print(len(state))

        return state

    def get_reward(self, profit):
        reward = 0
        if 0 < profit <= 0.1:
            reward = 1
        if 0.1 < profit <= 0.2:
            reward = 2
        if 0.2 <= profit:
            reward = 4
        if -0.1 <= profit < 0:
            reward = -1
        if -0.2 <= profit < 0.1:
            reward = -2
        if profit < -0.2:
            reward = -4
        return reward

    def step(self, action):
        # dqn specify
        action_dict = {
            '0': 0.2, '1': 0.4, '2': -0.2, '3': -0.4, '4': -1,
        }
        action = action_dict.get(str(action))

        if action > 0 and self.total_money * action < self.close1[self.trade_date + self.t] * 100:
            action = self.close1[self.trade_date + self.t] * 100 / self.total_money

        if action > 0:  # buy
            L = self.total_money * action // (self.close1[self.trade_date + self.t] * 100)

        else:  # sell
            L = self.inventory * action

        self.inventory += L
        self.total_money -= self.close1[self.trade_date + self.t] * 100 * L
        self.Portfolio_unit = (self.total_money + self.close1[
            self.trade_date + self.t] * 100 * self.inventory) / self.initial_money
        Rest_unit = self.total_money / self.initial_money

        add_state = np.array([self.Portfolio_unit, Rest_unit])

        total_profit = (self.total_money + self.close1[
            self.trade_date + self.t - 1] * 100 * self.inventory) - self.initial_money
        reward = self.get_reward(total_profit / self.initial_money)

        self.t += 1

        done = self.seq_time < (self.t + 1)

        state = self.dt[self.trade_date + self.t]
        add_state = np.array([self.Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])

        return state, reward, done, {}


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
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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

    plt.title('smoothed returns')
    plt.ylabel('Returns')
    plt.xlabel('time step')
    plt.plot(T, R)
    plt.grid()
    plt.show()


def train_dqn():

    log_dir = f"model_save/"
    env = ENV()
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                clip_obs=10.)
    model = DQN('MlpPolicy', env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    model.learn(total_timesteps=int(1e5), callback = callback, log_interval = 100)
    model.save('model_save/dqn')

def test_dqn():
    log_dir = f"model_save/dqn"
    env = ENV()
    env.render = True
    env = Monitor(env, log_dir)
    model = DQN.load(log_dir)
    plot_results(f"model_save/")
    state = env.reset()
    r = 0
    for t in range(300):
        action = model.predict(state)
        next_state, reward, done, info = env.step(action[0])
        r += reward
        print(r)
        if done:
            print('finish')
            break

if __name__ == '__main__':
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    train_dqn()
    test_dqn()


'''
训练输出：best mean reward, last mean reward per episode, num timesteps
'''