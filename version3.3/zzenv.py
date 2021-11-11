
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
        df = pd.read_csv(r'zztest.csv')
        self.today = '2018/1/2'

        self.istest = istest
        # if self.istest:
        #     df = pd.read_csv(r'combine/2019_zz500.csv')   # 2019测试
        #     self.today = '2019/1/2'
        #     df = pd.read_csv(r'combine/2020_zz500.csv')   # 2020测试
        #     self.today = '2020/1/2'
        #     df = pd.read_csv(r'combine/2021_zz500.csv')   # 2021测试
        #     self.today = '2021/1/4'
        # else:
        #     df = pd.read_csv(r'combine/2018_zz500.csv')   # 2018训练
        #     self.today = '2018/1/2'
        #     df = pd.read_csv(r'combine/2019_zz500.csv')   # 2019训练
        #     self.today = '2019/1/2'
        #     df = pd.read_csv(r'combine/2020_zz500.csv')   # 2020训练
        #     self.today = '2020/1/2'

        self.stock_all = df['thscode'].unique() # len=454
        self.test_count = 0  # for testing

        # self.stock_test = ['600028.SH','600050.SH','600309.SH','600570.SH','600703.SH','600887.SH','601166.SH',
        #                      '601336.SH','601668.SH','601888.SH']
        #
        # self.stock_train = ['600000.SH','600009.SH','600016.SH','600031.SH','600036.SH','600048.SH','600104.SH',
        #                      '600196.SH','600276.SH','600438.SH','600519.SH','600547.SH','600585.SH','600588.SH',
        #                      '600690.SH','600745.SH','600809.SH','600837.SH','600893.SH','601012.SH','601088.SH',
        #                      '601211.SH','601288.SH','601318.SH','601398.SH','601601.SH','601628.SH','601688.SH',
        #                      '601818.SH','601857.SH','601899.SH','603288.SH','603501.SH','603986.SH']

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
        self.flow = 0

        self.data_train = df.drop(['CLOSE_AFTER'], axis=1)
        self.close_train = df['CLOSE_AFTER']
        self.time_stump = df['time']

    def reset(self):
        if self.istest:
            thscode = self.stock_all[self.test_count]  # 测试时使用指定测试股票
            self.dt = self.data_train[self.stock_list == thscode]
            self.dt1 = np.array(self.dt.iloc[:, 3:])
            self.close1 = self.close_train[self.stock_list == thscode]
            self.time_stump1 = self.time_stump[self.stock_list == thscode]

            self.test_count+=1
            self.trade_date = 0 # 测试时从头开始跑
        else:
            thscode = random.choice(self.stock_all)   # 训练时随机选择股票
            self.dt = self.data_train[self.stock_list == thscode]
            self.dt1 = np.array(self.dt.iloc[:, 3:])
            self.close1 = self.close_train[self.stock_list == thscode]
            self.time_stump1 = self.time_stump[self.stock_list == thscode]

            self.trade_date = np.random.randint(0, len(self.close1) - self.seq_time)    # 训练时随机选择时间点开始

        self.inventory = 0
        self.initial_money = 1000000
        self.total_money = 1000000
        self.profit = 0
        self.profit_list = []       # 每个时间点的profit
        self.portfolio_list = []    # 每个step的资产
        self.stock_price = 0

        self.buy_hold = 0
        self.sp = 0
        self.maxdrawdown = 0
        self.mdd = 0
        self.romad = 0

        self.today_buy_port = 0 # 限制每日交易的

        Portfolio_unit = 1
        Rest_unit = 1
        self.t = 0
        state = self.dt1[self.trade_date + self.t]
        add_state = np.array([Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])
        #print("############### reset env ###############")
        #print("Stock:", thscode)

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
        self.stock_price = self.close1.iloc[self.trade_date + self.t]
        costing = self.stock_price * 100  # 单位仓位（100）交易的花销


        today_time = (self.time_stump1.iloc[self.trade_date + self.t]).split(' ')[0] # 获取日期到天


        if today_time != self.today:
            self.today = today_time
            self.today_buy_port = 0

        # print(self.today, self.time_stump1.iloc[self.trade_date + self.t])
        if action > 0:
            L0 = self.total_money * action // (costing * 1.0003)    # 假设可以买得起，则进仓L0，手续费万三
            costing0 = costing * 1.0003 + (L0 // 10 + 1) / 10        # 假设可以买得起时候的每100笔的花费

            if self.total_money * action >= costing0 * L0:               # 实际买得起
                L = L0
                costing = costing0
                self.today_buy_port += L0
            else:                                                   # 实际买不起
                L = 0

        else:  # 卖出
            L0 = int(self.inventory * action)       # 假设能卖，应该卖多少（-L0表示卖出量）
            if self.inventory - self.today_buy_port > (-L0):   # 今日允许卖
                L = L0
            else:
                L = -(self.inventory - self.today_buy_port) # 允许卖的条件下卖出尽可能多（L负的表示卖出）

            costing = costing * (1. - 0.001 - 0.0003) - ((-L) // 10 + 1) / 10

        # L是进仓多少（buy为正，sell为负，以100为单位1）
        self.flow = L
        self.inventory += L

        self.total_money -= costing * L

        self.Portfolio_unit = (self.total_money + self.close1.iloc[
            self.trade_date + self.t] * 100 * self.inventory) / self.initial_money  # 资产与初始资金比例
        Rest_unit = self.total_money / self.initial_money                           # 剩余金额占比

        # 现金+持有与初始资金的差额
        total_profit = (self.total_money + self.close1.iloc[
            self.trade_date + self.t - 1] * 100 * self.inventory) - self.initial_money
        # reward = self.get_reward(total_profit / self.initial_money)                 # 传入get_reward的就是收益率
        self.profit = total_profit / self.initial_money # get profit（收益率）
        self.profit_list.append(self.profit)
        self.portfolio_list.append(self.Portfolio_unit)

        self.t += 1

        if self.istest:
            done = len(self.close1)-1 < (self.t + 1)
        else:
            done = self.seq_time < (self.t + 1)

        self.buy_hold = (self.close1.iloc[self.trade_date + self.t] - self.close1.iloc[self.trade_date]) / self.close1.iloc[
            self.trade_date]

        state = self.dt1[self.trade_date + self.t]
        add_state = np.array([self.Portfolio_unit, Rest_unit]).flatten()
        state = np.hstack([state, add_state])

        # 计算夏普率
        sp_std = np.std(self.profit_list)
        if sp_std<10e-4:
            sp_std=10e-4
        self.sp = (np.mean(self.profit_list))/sp_std          # 最后输出全时间段的夏普率（无风险利率3%）

        # _r = np.log(self.profit_list).diff()
        # self.sp = _r.mean()/(_r.std() + 1e-10)

        # 计算最大回撤
        if done and self.istest:
            # 列表储存从i-th step开始的最大回撤率
            step_mdd_list = [
                (self.portfolio_list[i] - np.min(self.portfolio_list[i:]))/self.portfolio_list[i]  # i-step开始的最大回撤率
                             for i in range(self.t)
            ]
            self.mdd = np.max(step_mdd_list)  # 计算最大回撤
            self.romad = self.profit/self.mdd

        reward = self.get_reward(self.sp)

        # 检查测试中采用的策略
        # if self.istest:
            # 全部信息
            # print('trade_date: {}, action: {:.3f}, inventory: {}, cost per stock: {:.3f}, L: {}, money: {:.3f}, stock price: {:.3f}, profit: {:.3f}， portfolio unit: {:.3f}'
            #              .format(self.trade_date+self.t, action, self.inventory, costing/100, L, self.total_money, self.close1.iloc[self.trade_date + self.t], self.profit, self.Portfolio_unit))
            # 资产价值趋势
            # print('trade_date: {}, action: {:.3f}, L: {}, portfolio unit: {:.3f}'
            #              .format(self.trade_date+self.t, action, L, self.Portfolio_unit))
        return state, reward, done, {}