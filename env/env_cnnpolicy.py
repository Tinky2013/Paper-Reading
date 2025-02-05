
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

    def __init__(self,istest):
        df = pd.read_csv(r'hd5_sz50/2019_sz50n.csv')
        # df = pd.read_csv(r'2019_sz5.csv')

        nor_dt = df.drop(['time', 'thscode', 'amount', 'CLOSE_AFTER'], axis=1)
        id_dt = df[['time', 'thscode', 'amount', 'CLOSE_AFTER']]

        nor_dt = nor_dt.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        df = pd.concat([id_dt, nor_dt], axis=1)


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
        self.observation_space = gym.spaces.Box(low=-5,high=5,
                                                shape=(1, 31, 1),
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
        state = np.hstack([state, add_state]).reshape(1,-1,1)
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
        state = np.hstack([state, add_state]).reshape(1,-1,1)

        # print("trade_date:",self.trade_date+self.t,"action:",action,"inventory:", self.inventory)

        return state, reward, done, {}
