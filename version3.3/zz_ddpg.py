
from stable_baselines3 import DDPG

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

from env.zzenv import ENV

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
        self.save_path = os.path.join(log_dir, 'best_model_'+MODEL_PATH)
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


def train():

    log_dir = f"model_save/"
    env = ENV(istest=False)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                clip_obs=10.)

    model = DDPG('MlpPolicy', env, verbose=1, batch_size=PARAM['batch_size'], seed=PARAM['seed'], learning_starts=PARAM['learning_starts'])
    callback = SaveOnBestTrainingRewardCallback(check_freq=480, log_dir=log_dir)
    model.learn(total_timesteps=int(PARAM['total_time_step']), callback = callback, log_interval = 480)
    model.save('model_save/'+MODEL_PATH)

def test():
    log_dir = f"model_save/best_model_"+MODEL_PATH
    env = ENV(istest=True)
    env.render = True
    env = Monitor(env, log_dir)
    model = DDPG.load(log_dir)
    # plot_results(f"model_save/")
    trade_dt = pd.DataFrame([])     # trade_dt: 所有股票的交易数据
    result_dt = pd.DataFrame([])    # result_dt: 所有股票一年测试结果数据
    for i in range(TEST_STOCK_NUM):
        state = env.reset()
        stock_bh_id = 'stock_bh_'+str(i)        # 记录每个股票交易的buy_hold
        stock_port_id = 'stock_port_'+str(i)    # 记录每个股票交易的portfolio
        stock_action_id = 'stock_action_' + str(i)  # 记录每个股票交易的action
        flow_L_id = 'stock_flow_' + str(i)      # 记录每个股票的流水

        stock_bh_dt = []
        stock_port_dt = []
        action_policy_dt = []
        flow_L_dt = []

        day = 0
        while True:
            action = model.predict(state)
            next_state, reward, done, info = env.step(action[0])
            state = next_state
            # print("trying:",day,"reward:", reward,"now profit:",env.profit)   # 测试每一步的交易policy
            stock_bh_dt.append(env.buy_hold)
            stock_port_dt.append(env.Portfolio_unit)
            action_policy_dt.append(action[0][0])  # 用于记录policy
            flow_L_dt.append(env.flow)
            day+=1
            if done:
                print('stock: {}, total profit: {:.2f}%, buy hold: {:.2f}%, sp: {:.4f}, mdd: {:.2f}%, romad: {:.4f}'
                      .format(i, env.profit*100, env.buy_hold*100, env.sp, env.mdd*100, env.romad))
                # 交易完后记录：股票ID，利润（单位100%），buy_hold（单位100%），夏普率，最大回撤率（单位100%），romad
                result=pd.DataFrame([[i,env.profit*100,env.buy_hold*100,env.sp,env.mdd*100,env.romad]])
                break

        # trade_dt_stock = pd.DataFrame({stock_bh_id: stock_bh_dt, stock_port_id: stock_port_dt}) # 支股票的交易数据
        trade_dt_stock = pd.DataFrame({stock_port_id: stock_port_dt,
                                       stock_bh_id: stock_bh_dt,
                                       stock_action_id: action_policy_dt,
                                       flow_L_id: flow_L_dt})  # 支股票的交易数据

        trade_dt = pd.concat([trade_dt, trade_dt_stock], axis=1)    # 所有股票交易数据合并（加行）
        result_dt = pd.concat([result_dt,result],axis=0)            # 所有股票结果数据合并（加列）

    result_dt.columns = ['stock_id','prfit(100%)','buy_hold(100%)','sp','mdd(100%)','romad']
    trade_dt.to_csv('out_dt/trade_'+MODEL_PATH+'.csv',index=False)
    result_dt.to_csv('out_dt/result_'+MODEL_PATH+'.csv',index=False)

# 全局参数：根据不同的测试任务进行修改
TEST_STOCK_NUM = 453             # 测试多少股票（zz500共有453支）
MODEL_PATH = 'ddpg_200w_2019tr' # 保存模型名称，最新命名方式：算法+参数（迭代次数）+用哪年训练的
TRAIN_OR_NOT = True            # False代表只测试现有模型，True要训练并测试新的模型
PARAM = {
    'total_time_step': 4000000,
    'learning_starts': 2000000,
    'batch_size': 2048,
    'seed': 1,
}

if __name__ == '__main__':
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    if TRAIN_OR_NOT == True:
        train()
    test()
