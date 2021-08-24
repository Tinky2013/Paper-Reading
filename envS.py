import numpy as np
import pandas as pd
import gym
import os
import h5py


class ENVS(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        df = pd.read_csv(r'hd5_sz50/2019_sz50.csv')
        sdata = '2019-10-01'
        edata = '2019-12-31'
        df = df.set_index('time')
        df = df[(df.index >= sdata) & (df.index <= edata)]
        df = df.sort_index(ascending=True)

        self.stock_list = df['thscode']
        self.close = df['CLOSE_AFTER']

        self.action_space = gym.spaces.Discrete(5)  # (0,1,2,3,4)
        self.observation_space = gym.spaces.Box(
            low=np.array([-5] * 31),
            high=np.array([5] * 31)
        )

        self.seq_time = 480
        self.profit = 0

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']

    def reset(self):
        self.dt = self.data_train[self.stock_list == '601336.SH']
        self.dt = np.array(self.dt.iloc[:, 2:])
        self.close1 = self.close_train[self.stock_list == '601336.SH']

        self.inventory = 0
        self.initial_money = 1000000
        self.total_money = 1000000
        self.profit = 0

        self.trade_date = np.random.randint(0, len(self.close1) - self.seq_time)
        Portfolio_unit = 1
        Rest_unit = 1
        self.t = 0
        state = self.dt[self.trade_date + self.t]
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

        if action > 0 and self.total_money * action < self.close1[self.trade_date + self.t] * 100:  # can not afford
            action = self.close1[self.trade_date + self.t] * 100 / self.total_money                 # use all the money to buy as many as possible (larger action value)

        if action > 0:  # buy
            L = self.total_money * action // (self.close1[self.trade_date + self.t] * 100)

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

        state = self.dt[self.trade_date + self.t]
        add_state = np.array([self.Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])

        return state, reward, done, {}