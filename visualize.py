import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'td3_sp2'

trade_dt = pd.read_csv(r'out_dt/trade_'+MODEL_PATH+'.csv')
result_dt = pd.read_csv(r'out_dt/result_'+MODEL_PATH+'.csv')
trade_dayend_dt = pd.read_csv(r'out_dt/trade_'+MODEL_PATH+'_dayend.csv')
zz500r_dt = pd.read_csv(r'out_dt/zz500_ratio.csv')

bh_col = ['stock_bh_'+str(i) for i in range(10)]
port_col = ['stock_port_'+str(i) for i in range(10)]

zz500_ratio = zz500r_dt.loc[(zz500r_dt['zz']>='2019-01-02')&(zz500r_dt['zz']<'2020-01-02')]['zz500ratio']
bh_mean = trade_dt[bh_col].mean(axis=1)
for i in range(len(bh_mean)):
    bh_mean[i]=bh_mean[i]+1
zz500_total = [1]
for i in range(len(zz500_ratio)):
    zz500_total.append((zz500_total[i]*(1+zz500_ratio.iloc[i]/100)))
port_mean = trade_dt[port_col].mean(axis=1)
port_dayend_mean = trade_dayend_dt[port_col].mean(axis=1)
go_over_buy_hold = result_dt['prfit(100%)']-result_dt['buy_hold(100%)']

# 绘制交易中平均股票buyhold-资产图
t=[i for i in range(len(port_mean))]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t[:-400],bh_mean[:-400],"b-",linewidth=2,label="Buy hold")
ax1.set_ylim((0.5,2.5))
ax1.set_ylabel('Buy hold')
ax1.set_title('Average buy-hold and portfolio')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(t[:-400],port_mean[:-400],"g-",linewidth=2,label="Portfolio")
# ax2.set_xlim([0, np.e])
ax2.set_ylim((0.5,2.5))
ax2.set_ylabel('Portfolio')
ax2.set_xlabel('Trading time')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.grid()
#plt.show()
plt.savefig('img/bh_port_'+MODEL_PATH+'.jpg')

# 绘制交易中资产-zz500指数图
t=[i for i in range(len(zz500_ratio))]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t[:-1],zz500_total[:-2],"r-",linewidth=2,label="zz500 ratio")
ax1.set_ylim((0.5,2.5))
ax1.set_ylabel('zz500 ratio')
ax1.set_title('zz500 ratio and portfolio')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(t[:-1],port_dayend_mean[:-2],"g-",linewidth=2,label="Portfolio")
# ax2.set_xlim([0, np.e])
ax2.set_ylim((0.5,2.5))
ax2.set_ylabel('Portfolio')
ax2.set_xlabel('Trading time')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.grid()
#plt.show()
plt.savefig('img/zzratio_port_'+MODEL_PATH+'.jpg')

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

fig = plt.figure()
title = 'Average of (profit - buyhold):'+str(go_over_buy_hold.mean())
plt.hist(go_over_buy_hold, bins=100, facecolor="blue", alpha=0.5)
plt.title(title)
plt.xlabel("profit-buyhold")
plt.ylabel("frequency")
plt.grid()
#plt.show()
plt.savefig('img/ave_profit_bh_'+MODEL_PATH+'.jpg')
