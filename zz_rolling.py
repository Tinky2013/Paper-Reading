import warnings
from stable_baselines3 import DDPG, TD3, PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

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

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: The environment used for initialization
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,    # 有最佳新模型时，回调触发器
        n_eval_episodes: int = 10,           # 每次验证跑多少个episode
        eval_freq: int = 10000,             # 多少个timestep验证一次
        save_freq: int = 100000,            # 多少个timestep保存一次
        log_path: str = None,               # 存储evaluation.npz的路径
        best_model_save_path: str = None,   # 存储best model的路径
        deterministic: bool = True,         # 使用确定性策略还是随机策略来evaluation
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.best_mean_reward_overall = None
        self.best_mean_reward_timestep = None

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        info = locals_["info"]
        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        # n_calls -> timesteps
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                # print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                # self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            # self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            # self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    # self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.model.save(self.best_model_save_path)
                self.best_mean_reward = mean_reward
                self.best_mean_reward_overall, self.best_mean_reward_timestep = self.best_mean_reward, self.n_calls
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        if self.n_calls % self.save_freq == 0:
            self.model.save('model_save/'+MODEL_PATH+'/'+MODEL_PATH+'_'+str(self.n_calls)+'_timesteps')

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

def train():
    best_reward, best_reward_timesteps = None, None
    save_path = "model_save/"+MODEL_PATH+"/"
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # log_dir = f"model_save/"
    log_dir = save_path
    env, env_eval = ENV(util='train', par=PARAM, dt=DT), ENV(util='val', par=PARAM, dt=DT)
    env, env_eval = Monitor(env, log_dir), Monitor(env_eval, log_dir)
    env, env_eval = DummyVecEnv([lambda: env]), DummyVecEnv([lambda: env_eval])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                clip_obs=10.)

    if PARAM['algo']=='td3':
        model = TD3('MlpPolicy', env, verbose=1, batch_size=PARAM['batch_size'], seed=PARAM['seed'],
                    learning_starts=PARAM['learning_starts'])
    elif PARAM['algo']=='ddpg':
        model = DDPG('MlpPolicy', env, verbose=1, batch_size=PARAM['batch_size'], seed=PARAM['seed'],
                     learning_starts=PARAM['learning_starts'])
    elif PARAM['algo']=='ppo':
        model = PPO('MlpPolicy', env, verbose=1, batch_size=PARAM['batch_size'], seed=PARAM['seed'])

    eval_callback = EvalCallback(env_eval, best_model_save_path=save_path+MODEL_PATH+'_best_model',
                                 log_path=log_dir, eval_freq=PARAM['eval_freq'], save_freq=PARAM['save_freq'],
                                 deterministic=True, render=False)

    model.learn(total_timesteps=int(PARAM['total_time_step']), callback=eval_callback, log_interval = 500)
    print("best mean reward:", eval_callback.best_mean_reward_overall, "timesteps:", eval_callback.best_mean_reward_timestep)
    model.save(save_path+MODEL_PATH+'_final_timesteps')

def test(MODEL_TEST):
    log_dir = "model_save/" + MODEL_PATH + "/" + MODEL_PATH + MODEL_TEST

    env = ENV(util='test', par=PARAM, dt=DT)
    env.render = True
    env = Monitor(env, log_dir)

    if PARAM['algo']=='td3':
        model = TD3.load(log_dir)
    elif PARAM['algo']=='ddpg':
        model = DDPG.load(log_dir)
    elif PARAM['algo']=='ppo':
        model = PPO.load(log_dir)

    # plot_results(f"model_save/")
    trade_dt = pd.DataFrame([])     # trade_dt: 所有股票的交易数据
    result_dt = pd.DataFrame([])    # result_dt: 所有股票一年测试结果数据
    for i in range(TEST_STOCK_NUM):
        state = env.reset()
        stock_bh_id = 'stock_bh_'+str(i)            # 记录每个股票交易的buy_hold
        stock_port_id = 'stock_port_'+str(i)        # 记录每个股票交易的portfolio
        stock_action_id = 'stock_action_' + str(i)  # 记录每个股票交易的action
        flow_L_id = 'stock_flow_' + str(i)          # 记录每个股票的流水
        stock_bh_dt, stock_port_dt, action_policy_dt, flow_L_dt = [], [], [], []
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
TEST_STOCK_NUM = 15             # 测试多少股票（zz500共有453支）
MODEL_PATH = 'td3_test'        # 保存模型名称，最新命名方式：算法+参数（迭代次数）+用哪年训练的
MODEL_TEST = '_best_model'      # 想要测试哪个模型，可选'_best_model', '_final_timesteps', '_14400_timesteps'
TRAIN_OR_NOT = True    # False代表只测试现有模型，True要训练并测试新的模型
DT = {
    'train': 'local',
    'val': 'local',
    'test': 'local',
}
PARAM = {
    'algo': 'td3',      # 'ppo', 'td3', 'ddpg'
    'total_time_step': 40000,
    'learning_starts': 20000,
    'batch_size': 2048,
    'seed': 1,
    'seq_time': 48,     # 每个episode跑多少步（每个episode有多少个timestep）
    'eval_freq': 480,   # 多少个timestep测一次
    'save_freq': 4800,  # 多少个timestep保存一次模型
}

if __name__ == '__main__':
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    if TRAIN_OR_NOT:
        train()
    test(MODEL_TEST)
