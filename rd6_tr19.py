import numpy as np
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import time
import os
import h5py
from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.disable_v2_behavior()    # compact 2.0版本
# check CPU or GPU version
print(device_lib.list_local_devices())
print(tf.__version__)
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

# M2是多个股票预测和训练
sns.set()

class Model:
    def __init__(self, input_size1,output_size, addstate_size, learning_rate):
        self.X = tf.placeholder(tf.float32, (None, input_size1))    # 输入数据
        self.Y = tf.placeholder(tf.float32, (None, output_size))    # action（每个action一个Qvalue）
        self.S = tf.placeholder(tf.float32, (None, addstate_size))  # 增加维度（Portfolio_unit和Rest_unit）

        # tensor_action = tf.concat([self.X, self.S], 1)
        self.logits = tf.layers.dense(self.X, output_size)  # 输出（只有一个全连接层）
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


class Agent:
    def __init__(self, window_size, batch_size, pretime, stock_list, earning, state_data,
                 train_index, valid_index, test_index, close, output_graph=True):
        self.sell_window = window_size // 2

        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index

        self.state_data = state_data
        self.data_train = state_data[train_index]
        self.data_valid = state_data[valid_index]
        self.data_test = state_data[test_index]

        self.earning = earning
        self.earning_train = earning[train_index]
        self.earning_valid = earning[valid_index]
        self.earning_test = earning[test_index]

        self.stock_list = stock_list
        self.stock_train = stock_list[train_index]
        self.stock_valid = stock_list[valid_index]
        self.stock_test = stock_list[test_index]

        self.close =close
        self.close_train = close[train_index]
        self.close_valid = close[valid_index]
        self.close_test = close[test_index]



        self.action_size = 5
        self.memory = deque()
        self.memory_size = 50000

        self.batch_size = batch_size
        self.step = 50
        self.gamma = 0.99
        self.lr = 0.001
        # self.copy = 100
        # self.T_copy = 0
        self.best_reward = 0
        self.highest_win_rate = 0

        self.epsilon = 0.8
        self.epsilon_min = 0.5
        self.decay_rate = 0.005
        self.epsilon_decay = 0.9999

        self.seq_time = 180
        self.back_time = state_data.shape[1]
        self.layer_size = 8
        self.pretime = pretime

        self.add_state = 2
        self.state_size = state_data[:self.batch_size].shape
        self.state_size1 = state_data.shape[1]
        # self.state_size1 = close.shape[1]
        self.dim = state_data.shape[2]
        # print(self.state_size1)

        # hidden_layer的初始值吧v
        self.initial_value = np.zeros((1, 2 * self.layer_size))
        tf.reset_default_graph()
        tf.set_random_seed(1)
        # self.model = Model(self.state_size1, self.action_size,  self.add_state, self.lr)
        # self.saver = tf.train.Saver(max_to_keep=30)

        ###############__MODEL__##################
        self.X = tf.placeholder(tf.float32, (None, self.state_size1, self.dim))
        self.Y = tf.placeholder(tf.float32, (None, self.action_size))
        self.S = tf.placeholder(tf.float32, (None, self.add_state))
        self.P = tf.placeholder(tf.float32, (None, 1))

        with tf.variable_scope('DL'):
            cell = tf.nn.rnn_cell.LSTMCell(self.layer_size, state_is_tuple=False)
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * self.layer_size))
            # init_state = cell.zero_state(256, dtype=tf.float32)
            self.rnn, self.last_state = tf.nn.dynamic_rnn(inputs=self.X, cell=cell,
                                                          dtype=tf.float32,
                                                          initial_state=self.hidden_layer)
            self.dense = tf.layers.dense(self.rnn[:, -1], 1)

        with tf.variable_scope('RL'):
            # tf.split把一个张量划分成几个子张量,
            # num_or_size_splits：准备切成几份，axis : 准备在第几个维度上进行切割
            # 这里把lstm输出的所有ht直接硬分两半，我这里让靠近日子的ht去做action，靠远的ht做validation
            tensor_validation, tensor_action = tf.split(self.rnn[:, -1], 2, 1)
            # 像库存之类的额外state需要并人
            tensor_action = tf.concat([tensor_action, self.S], 1)
            feed_action = tf.layers.dense(tensor_action, self.action_size)
            feed_validation = tf.layers.dense(tensor_validation, 1)
            self.logits = feed_validation + tf.subtract(feed_action,
                                                        tf.reduce_mean(feed_action, axis=1, keep_dims=True))

        dl_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DL')
        rl_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RL')
        self.lose = tf.reduce_mean(tf.square(self.P - self.dense))
        self.optimizer_p = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.lose, var_list=dl_params)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits)) # Y是每个action的Q value
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        self.saver1 = tf.train.Saver(var_list=dl_params, max_to_keep=30)
        self.saver2 = tf.train.Saver(max_to_keep=30)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # tf.trainable_variables()用于查看可训练的变量
        self.trainable = tf.trainable_variables()

    def get_reward(self,profit):
        reward = 0
        if 0<profit<=0.1:
            reward = 1
        if 0.1<profit<=0.2:
            reward = 2
        if 0.2<=profit:
            reward = 4
        if -0.1<=profit<0:
            reward = -1
        if -0.2<=profit<0.1:
            reward = -2
        if profit<-0.2:
            reward = -4
        return reward

    def buy(self, initial_money, thscode, Test=True):

        stock_index_test = np.argwhere(self.stock_test == thscode)
        stock_index_test = list(stock_index_test.reshape(stock_index_test.shape[0]))

        stock_index_valid = np.argwhere(self.stock_valid == thscode)
        stock_index_valid = list(stock_index_valid.reshape(stock_index_valid.shape[0]))

        if Test == True:
            dt = self.data_test[stock_index_test]
            close = self.close_test[stock_index_test]
            close1 = list(close.reshape(close.shape[0]))
        else:
            dt = self.data_valid[stock_index_valid]
            close = self.close_valid[stock_index_valid]
            close1 = list(close.reshape(close.shape[0]))

        total_money = initial_money
        states_sell = []
        states_buy = []
        inventory = 0
        Portfolio_unit = 1
        Rest_unit = 1
        add_state = np.array([Portfolio_unit, Rest_unit])
        # 因为每个state已经会看过40天了，这里self.initial_value 都是0，这样不会让40天前的日子影响模型判断（predict日子==训练日子）
        self.initial_value = np.zeros((1, 2 * self.layer_size))

        # TODO
        # self.saver1.restore(self.sess, 'Model/new-ddr2/predict_6_30model1600.ckpt')
        # self.saver2.restore(self.sess, 'Model/double-duel-rnn/BEST_train_duel_rnn_model.ckpt')
        # self.saver2.restore(self.sess, 'Model/double-duel-rnn/Best_valid_duel_rnn_model.ckpt')
        # self.saver2.restore(self.sess, 'Model/double-duel-rnn/duel_rnn_model29999.ckpt')

        for t in range(0, len(close1)-1):
            Q, last_state = self.sess.run([self.logits, self.last_state],
                                          feed_dict={self.X: [dt[t]],
                                                     self.S: [add_state],
                                                     self.hidden_layer: self.initial_value})
            action = np.argmax(Q[0])

            if action == 0 and total_money* 0.2 >= close1[t] * 100:
                # 买20%仓位
                L = total_money * 0.2 // (close1[t] * 100)
                inventory += L
                total_money -= close1[t] * 100 * L
                Portfolio_unit = (total_money + close1[t] * 100 * inventory) / initial_money
                Rest_unit = total_money / initial_money
                states_buy.append(t)
                print(
                    'day %d: buy %f unit at price %f, total portfolio %f' % (t, 100 * L, close1[t], Portfolio_unit*initial_money))

            if action == 1 and total_money* 0.4 >= close1[t] * 100:
                # 买40%仓位
                L = total_money * 0.4 // (close1[t] * 100)
                inventory += L
                total_money -= close1[t] * 100 * L
                Portfolio_unit = (total_money + close1[t] * 100 * inventory) / initial_money
                Rest_unit = total_money / initial_money
                states_buy.append(t)
                print(
                    'day %d: buy %f unit at price %f, total portfolio %f' % (
                    t, 100 * L, close1[t], Portfolio_unit * initial_money))

            if action == 2 and total_money >= close1[t] * 100 and inventory// 5 > 0:
                # 卖20%仓位
                L = inventory // 5
                inventory -= L
                total_money += close1[t] * 100 * L
                Portfolio_unit = (total_money + close1[t] * 100 * inventory) / initial_money
                Rest_unit = total_money / initial_money
                states_sell.append(t)
                print('day %d, sell %f unit at price %f, total portfolio %f,'
                      % (t, 100 * L , close1[t], Portfolio_unit*initial_money))

            if action == 3 and total_money >= close1[t] * 100 and inventory* 2 // 5 > 0:
                # 卖40%仓位
                L = inventory * 2 // 5
                inventory -= L
                total_money += close1[t] * 100 * L
                Portfolio_unit = (total_money + close1[t] * 100 * inventory) / initial_money
                Rest_unit = total_money / initial_money
                states_sell.append(t)
                print('day %d, sell %f unit at price %f, total portfolio %f,'
                      % (t, 100 * L, close1[t], Portfolio_unit * initial_money))

            if action == 4 and total_money >= close1[t] * 100 and inventory > 0:
                # 全卖
                L = inventory
                inventory -= L
                total_money += close1[t] * 100 * L
                Portfolio_unit = (total_money + close1[t] * 100 * inventory) / initial_money
                Rest_unit = total_money / initial_money
                states_sell.append(t)
                print('day %d, sell %f unit at price %f, total portfolio %f,'
                      % (t, 100 * L, close1[t], Portfolio_unit * initial_money))

            add_state = np.array([Portfolio_unit, Rest_unit])

        total_profit = (total_money + close1[-1] * 100 * inventory) - initial_money
        total_earning = (total_profit / initial_money) * 100

        return close1, states_buy, states_sell, total_profit, total_earning

    def Train(self, iterations, checkpoint, initial_money, thscode):


        stock_index_train = np.argwhere(self.stock_train== thscode)
        stock_index_train = list(stock_index_train.reshape(stock_index_train.shape[0]))
        #print(stock_index_train)

        dt = self.data_train[stock_index_train]
        close= self.close_train[stock_index_train]
        close1 = list(close.reshape(close.shape[0]))
        #print(len(close1))
        total_reward = []
        total_cost = []
        valid_earning = 0
        #self.saver1.restore(self.sess, 'Model/new-ddr2/predict_57_30model1600.ckpt')
        # RL部分的训练核心意义应该在于加入了手数/仓位概念
        np.random.seed(1)
        for i in range(iterations):
            inventory = 0
            total_money = initial_money
            trade_date = np.random.randint(0, len(close1) - self.seq_time)
            #print(trade_date)
            #print(len(close1))
            Portfolio_unit = 1
            Rest_unit = 1
            add_state = np.array([Portfolio_unit, Rest_unit])
            self.initial_value = np.zeros((1, 2 * self.layer_size))

            global state1
            global Q1
            global action1
            global RNN_MEMORY

            buy = 0
            sell = 0

            for t in range(0, self.seq_time):

                # Q = self.sess.run(self.model.logits, feed_dict={self.model.X: [close[trade_date + t +1]],
                #                                          self.model.S: [add_state]})
                Q, last_state = self.sess.run([self.logits, self.last_state],
                                              feed_dict={self.X: [dt[trade_date + t]],
                                                         self.S: [add_state],
                                                         self.hidden_layer: self.initial_value})
                action = np.argmax(Q[0])

                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_size)

                if t == 0:
                    state1 = dt[trade_date + t]
                    Q1 = Q
                    action1 = action

                if action == 0 and total_money* 0.2 >= close1[trade_date + t] * 100:
                    # 买20%仓位
                    L = total_money * 0.2 // (close1[trade_date + t] * 100)
                    inventory += L
                    total_money -= close1[trade_date + t] * 100 * L
                    Portfolio_unit = (total_money+close1[trade_date + t] * 100 * inventory) / initial_money
                    Rest_unit = total_money/initial_money
                    buy +=1

                if action == 1 and total_money* 0.4 >= close1[trade_date + t] * 100:
                    # 买40%仓位
                    L = total_money * 0.4 // (close1[trade_date + t] * 100)
                    inventory += L
                    total_money -= close1[trade_date + t] * 100 * L
                    Portfolio_unit = (total_money+close1[trade_date + t] * 100 * inventory) / initial_money
                    Rest_unit = total_money/initial_money
                    buy +=1

                if action == 2 and total_money >= close1[trade_date + t] * 100 and inventory//5 > 0:
                    # 卖20%仓位
                    L = inventory //5
                    inventory -= L
                    total_money += close1[trade_date + t] * 100 * L
                    Portfolio_unit = (total_money+close1[trade_date + t] * 100 * inventory) / initial_money
                    Rest_unit = total_money/initial_money
                    sell +=1

                if action == 3 and total_money >= close1[trade_date + t] * 100 and inventory*2 //5 > 0:
                    # 卖40%仓位
                    L = inventory *2 //5
                    inventory -= L
                    total_money += close1[trade_date + t] * 100 * L
                    Portfolio_unit = (total_money+close1[trade_date + t] * 100 * inventory) / initial_money
                    Rest_unit = total_money/initial_money
                    sell +=1

                if action == 4 and total_money >= close1[trade_date + t] * 100 and inventory > 0:
                    # 全卖
                    L = inventory
                    inventory -= L
                    total_money += close1[trade_date + t] * 100 * L
                    Portfolio_unit = (total_money+close1[trade_date + t] * 100 * inventory) / initial_money
                    Rest_unit = total_money/initial_money
                    sell +=1

                add_state = np.array([Portfolio_unit, Rest_unit])
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            total_profit = (total_money + close1[trade_date+self.seq_time-1] * 100 * inventory) - initial_money
            reward =self.get_reward(total_profit/initial_money)
            total_reward.append(total_profit/initial_money)
            target = Q1
            target[:,action1] = reward
            cost, _ = self.sess.run([self.cost, self.optimizer],
                                    feed_dict={self.X: [state1], self.Y: target, self.S: [add_state], self.hidden_layer: self.initial_value})
            total_cost.append(cost)

            if total_profit > self.best_reward and i > 20000:
                self.saver2.save(self.sess, 'Model/double-duel-rnn/BEST_train_duel_rnn_model.ckpt')
                self.best_reward = total_profit
                print('new_reward_record at iteration '+str(i)+'is'+str(self.best_reward))

            if (i + 1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f, cost: %f， buy: %f, sell:%f' % (
                i + 1, total_profit, cost, buy, sell))
                close_v, states_buy, states_sell, total_profit_v, total_earning_v = self.buy(initial_money, thscode, False)
                print('valid total earning: %f'%(total_earning_v))
                if total_earning_v>valid_earning and i > 20000:
                    self.saver2.save(self.sess, 'Model/double-duel-rnn/Best_valid_duel_rnn_model.ckpt')
                    valid_earning = total_earning_v

            if (i + 1) % 50000 == 0:
                self.saver2.save(self.sess, 'Model/double-duel-rnn/duel_rnn_model' + str(i) + '.ckpt')
        return total_reward, total_cost

start = time.time()
# 先把train和test集分好
sdata1 = '2019-01-01'
edata1 = '2019-08-31'
sdata2 = '2019-09-01'
edata2 = '2019-10-31'
sdata3 = '2019-11-01'
edata3 = '2019-12-31'

f_name = "hd52/1020_57dim_40back_30pre.hdf5"

f18 = "hd5_sz50/2018_sz5031dim_120back_120pre.hdf5"
f19 = "hd5_sz50/201931dim_120back_120pre.hdf5"
f20 = "hd5_sz50/2020_sz5031dim_120back_120pre.hdf5"

with h5py.File(f_name, "r") as f:
    print(f.keys)
    # print(f)                              # hdf5格式（以1020_57dim_40back_30pre.hdf5为例）
    stock_list = f.get('stock')[:]          # 股票代码列表 185044
    date_index = f.get('date_index')[:]     # 日期列表 185044
    earning = f.get('earning')[:]           # earning那一列 185044
    state_new = f.get('state')[:]           # 三维列表（185044,40,57）
    close = f.get('close')[:]               # close那一列

date = []
for i in date_index:
    date.append(i.decode())
stock = []
for j in stock_list:
    stock.append(j.decode())

date = np.array(date)
train_index = np.argwhere((date >= sdata1) & (date <= edata1))
train_index = list(train_index.reshape(train_index.shape[0]))
valid_index = np.argwhere((date >= sdata2) & (date <= edata2))
valid_index = list(valid_index.reshape(valid_index.shape[0]))
test_index = np.argwhere((date >= sdata3) & (date <= edata3))
test_index = list(test_index.reshape(test_index.shape[0]))

earning = np.array(earning)
earning = earning.reshape((-1, 1))
stock =np.array(stock)

initial_money = 1000000
window_size = 4
batch_size = 256
pretime = int(f_name[-10:-8])
print(pretime)


agent = Agent(window_size=window_size, batch_size=batch_size, pretime=pretime,stock_list=stock, earning=earning,
              state_data=state_new,train_index=train_index, valid_index=valid_index, test_index=test_index, close=close, output_graph=False)
total_reward, total_cost = agent.Train(40000, 1000, initial_money, '600030.SH') # 进行训练
print(np.mean(total_reward))
#close1, states_buy, states_sell, total_profit, total_earning = agent.buy(initial_money, '600030.SH', True)  # 进行测试

end = time.time()
print(end-start)
print("complete training!")

# fig = plt.figure(figsize = (15,5))
# plt.plot(close1, color='r', lw=2.)
# plt.plot(close1, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
# plt.plot(close1, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
# plt.title('total gains %f, total investment %f%%'%(total_profit, total_earning))
# plt.legend()
# plt.show()