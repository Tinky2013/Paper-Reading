
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

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

    def __init__(self,istest):
        df = pd.read_csv(r'hd5_sz50/2019_sz50.csv')
        # df = pd.read_csv(r'2019_sz5.csv')
        # df = pd.read_csv(r'2019_sz5n.csv')

        sdata = '2019-01-01'
        edata = '2019-12-31'

        df = df.set_index('time')
        # df = df[(df.index >= sdata) & (df.index <= edata)]
        df = df.sort_index(ascending=True)
        self.istest = istest
        self.test_count = 0 # for testing


        self.stock_test = ['600028.SH','600050.SH','600309.SH','600570.SH','600703.SH','600887.SH','601166.SH',
                             '601336.SH','601668.SH','601888.SH']

        self.stock_train = ['600000.SH','600009.SH','600016.SH','600031.SH','600036.SH','600048.SH','600104.SH',
                             '600196.SH','600276.SH','600438.SH','600519.SH','600547.SH','600585.SH','600588.SH',
                             '600690.SH','600745.SH','600809.SH','600837.SH','600893.SH','601012.SH','601088.SH',
                             '601211.SH','601288.SH','601318.SH','601398.SH','601601.SH','601628.SH','601688.SH',
                             '601818.SH','601857.SH','601899.SH','603288.SH','603501.SH','603986.SH']
        self.stock_list = df['thscode']
        self.close = df['CLOSE_AFTER']

        self.action_space = gym.spaces.Box(
            low=np.array([-1] * 1),
            high=np.array([1] * 1),
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-5] * 31),
            high=np.array([5] * 31)
        )

        self.seq_time = 480
        self.profit = 0
        self.buy_hold = 0

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']

    def reset(self):
        if self.istest:
            thscode = self.stock_test[self.test_count]
            self.dt = self.data_train[self.stock_list == thscode]
            self.dt = np.array(self.dt.iloc[:, 2:])
            self.close1 = self.close_train[self.stock_list == thscode]
            self.test_count+=1
        else:
            thscode = random.choice(self.stock_train)
            self.dt = self.data_train[self.stock_list == thscode]
            self.dt = np.array(self.dt.iloc[:, 2:])
            self.close1 = self.close_train[self.stock_list == thscode]

        self.inventory = 0
        self.initial_money = 1000000
        self.total_money = 1000000
        self.profit = 0
        self.buy_hold = 0

        self.trade_date = np.random.randint(0, len(self.close1) - self.seq_time)
        Portfolio_unit = 1
        Rest_unit = 1
        self.t = 0
        state = self.dt[self.trade_date + self.t]
        add_state = np.array([Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])
        # print("Stock:", thscode)

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
        if -0.2 <= profit < -0.1:
            reward = -2
        if profit < -0.2:
            reward = -4
        return reward

    def step(self, action):
        action = action[0]

        if action > 0 and self.total_money * action >= self.close1[self.trade_date + self.t] * 100: # 买入action的仓位
            L = self.total_money * action // (self.close1[self.trade_date + self.t] * 100)

        elif action > 0 and self.total_money * action < self.close1[self.trade_date + self.t] * 100: # 买不起
            L = 0

        else:  # sell
            L = int(self.inventory * action)

        # L是进仓多少（buy为正，sell为负）

        self.inventory += L
        self.total_money -= self.close1[self.trade_date + self.t] * 100 * L         # +-交易所用金额

        self.Portfolio_unit = (self.total_money + self.close1[
            self.trade_date + self.t] * 100 * self.inventory) / self.initial_money  # 资产与初始资金比例
        Rest_unit = self.total_money / self.initial_money                           # 剩余金额占比

        # add_state = np.array([self.Portfolio_unit, Rest_unit])

        # 现金+持有与初始资金的差额
        total_profit = (self.total_money + self.close1[
            self.trade_date + self.t - 1] * 100 * self.inventory) - self.initial_money
        reward = self.get_reward(total_profit / self.initial_money)                 # 传入get_reward的就是收益率
        self.profit = total_profit / self.initial_money # get profit

        self.t += 1

        done = self.seq_time < (self.t + 1)
        self.buy_hold = (self.close1[self.trade_date + self.t] - self.close1[self.trade_date]) / self.close1[
            self.trade_date]

        state = self.dt[self.trade_date + self.t]
        add_state = np.array([self.Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])

        # print("trade_date:",self.trade_date+self.t,"action:",action,"inventory:", self.inventory)

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
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(3, 1), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


def train_ppo():
    log_dir = f"model_save/"
    env = ENV(istest=False)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                clip_obs=10.)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    model.learn(total_timesteps=int(3e6), callback=callback, log_interval=10000)
    model.save('model_save/PPO_cnn')


def test_ppo():
    log_dir = f"model_save/PPO_cnn"
    env = ENV(istest=True)
    env.render = True
    env = Monitor(env, log_dir)
    model = PPO.load(log_dir)
    plot_results(f"model_save/")
    for i in range(10):
        state = env.reset()
        while True:
            action = model.predict(state)
            next_state, reward, done, info = env.step(action[0])
            state = next_state
            # print("trying:",i,"action:", action,"now profit:",env.profit)
            if done:
                print('stock', i, ' total profit=', env.profit, ' buy hold=', env.buy_hold)
                break


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


if __name__ == '__main__':
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    train_ppo()
    test_ppo()
