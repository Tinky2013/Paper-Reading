import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 全局参数：根据不同的测试任务进行修改
MODEL_PATH = 'ppo_test'  # 要测试哪个模型
TEST_STOCK_NUM = 15     # 要测试多少支股票
ST = '2019' # 从哪年的第一个交易日开始测试
ED = '2020' # 到哪年的第一个交易日结束测试
# 设定要画什么图
VIS_DT = {
    'ave_bh_and_port': True,    # 所有股票测试时期buy_hold和portfolio对比曲线图
    'ave_zr_and_port': False,    # 所有股票测试时期portfolio和zz500指数对比图
    'port_and_bh_dist': False,   # 股票buy_hold和portfolio分布对比图
    'port_minus_bh_dist': False, # 股票portfolio与buy_hold差值对比图
}


trade_dt = pd.read_csv(r'out_dt/trade_'+MODEL_PATH+'.csv')     # 交易总数据
result_dt = pd.read_csv(r'out_dt/result_'+MODEL_PATH+'.csv')   # 测试的结果数据（最后的各种指标）

# trade_dayend_dt = pd.read_csv(r'out_dt/trade_'+MODEL_PATH+'_dayend.csv')    # 每日收盘的数据（为了和zz500指数比较）
# zz500r_dt = pd.read_csv(r'out_dt/zz500_ratio.csv')  # zz500指数（单日涨跌幅）

bh_col = ['stock_bh_'+str(i) for i in range(TEST_STOCK_NUM)]
port_col = ['stock_port_'+str(i) for i in range(TEST_STOCK_NUM)]

# 将zz500指数的每日涨跌幅换为净值
# zz500_ratio = zz500r_dt.loc[(zz500r_dt['zz']>=(ST+'-01-02'))&(zz500r_dt['zz']<(ED+'-01-02'))]['zz500ratio']
# zz500_total = [1]
# for i in range(len(zz500_ratio)):
#     zz500_total.append((zz500_total[i]*(1+zz500_ratio.iloc[i]/100)))

# 所有测试股票buy_hold的平均值
bh_mean = trade_dt[bh_col].mean(axis=1)
for i in range(len(bh_mean)):
    bh_mean[i]=bh_mean[i]+1

port_mean = trade_dt[port_col].mean(axis=1) # 所有测试股票portfolio的平均值
# port_dayend_mean = trade_dayend_dt[port_col].mean(axis=1)   # 每天收盘时候的portfolio（为了和zz500指数比较）
go_over_buy_hold = result_dt['prfit(100%)']-result_dt['buy_hold(100%)'] # 最后超过了buy_hold多少

if VIS_DT['ave_bh_and_port']==True:
    err_dt = 1        # TODO 后面数据录入不全（待处理）
    # 绘制交易中平均股票buyhold-资产图
    t=[i for i in range(len(port_mean))]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t[:-err_dt],bh_mean[:-err_dt],"b-",linewidth=2,label="Buy hold")
    ax1.set_ylim((0.99,1.02))
    ax1.set_ylabel('Buy hold')
    ax1.set_title('Average buy-hold and portfolio')
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(t[:-err_dt],port_mean[:-err_dt],"g-",linewidth=2,label="Portfolio")
    # ax2.set_xlim([0, np.e])
    ax2.set_ylim((0.99,1.02))
    ax2.set_ylabel('Portfolio')
    ax2.set_xlabel('Trading time')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.grid()
    #plt.show()
    plt.savefig('img/bh_port_'+MODEL_PATH+'.jpg')
    print("Finish ploting bh port!")

# if VIS_DT['ave_zr_and_port']==True:
#     err_dt = 2          # TODO 后面数据录入不全（待处理）
#     # 绘制交易中资产-zz500指数图
#     t=[i for i in range(len(zz500_ratio)-1)]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.plot(t[:-err_dt],zz500_total[:-err_dt],"r-",linewidth=2,label="zz500 ratio")
#     ax1.set_ylim((0.5,2.5))
#     ax1.set_ylabel('zz500 ratio')
#     ax1.set_title('zz500 ratio and portfolio')
#     ax2 = ax1.twinx()  # this is the important function
#     ax2.plot(t[:-2],port_dayend_mean[:-2],"g-",linewidth=2,label="Portfolio")
#     # ax2.set_xlim([0, np.e])
#     ax2.set_ylim((0.5,2.5))
#     ax2.set_ylabel('Portfolio')
#     ax2.set_xlabel('Trading time')
#     ax1.legend(loc='upper right')
#     ax2.legend(loc='upper left')
#     plt.grid()
#     #plt.show()
#     plt.savefig('img/zzratio_port_'+MODEL_PATH+'.jpg')
#     print("Finish ploting zr port!")

if VIS_DT['port_and_bh_dist']==True:
    # 资产收益与buyhold分布图
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(result_dt['prfit(100%)'], bins=100, facecolor="blue", alpha=0.5, label='profit')
    ax1.set_ylabel('Stock Profit Distribution')
    ax1.set_title('Stock profit and buy-hold')
    ax2 = ax1.twinx()  # this is the important function
    ax2.hist(result_dt['buy_hold(100%)'], bins=100, facecolor="red", alpha=0.5, label='buy_hold')
    plt.xlabel("return (100%)")
    plt.ylabel("frequency")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.grid()
    #plt.show()
    plt.savefig('img/profit_bh_dis_'+MODEL_PATH+'.jpg')
    print("Finish ploting port bh dist!")

if VIS_DT['port_minus_bh_dist']==True:
    fig = plt.figure()
    title = 'Average of (profit - buyhold):'+str(go_over_buy_hold.mean())
    plt.hist(go_over_buy_hold, bins=100, facecolor="blue", alpha=0.5)
    plt.title(title)
    plt.xlabel("profit-buyhold")
    plt.ylabel("frequency")
    plt.grid()
    #plt.show()
    plt.savefig('img/ave_profit_bh_'+MODEL_PATH+'.jpg')
    print("Finish ploting port-bh dist!")
