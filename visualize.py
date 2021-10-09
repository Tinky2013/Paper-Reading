import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'ppo_sp2'

trade_dt = pd.read_csv(r'out_dt/trade_'+MODEL_PATH+'.csv')
result_dt = pd.read_csv(r'out_dt/result_'+MODEL_PATH+'.csv')

price_col = ['stock_price_'+str(i) for i in range(453)]
port_col = ['stock_port_'+str(i) for i in range(453)]

price_mean = trade_dt[price_col].mean(axis=1)
port_mean = trade_dt[port_col].mean(axis=1)
go_over_buy_hold = result_dt['prfit(100%)']-result_dt['buy_hold(100%)']

# 绘制交易中平均股票价格-资产图
t=[i for i in range(len(price_mean))]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(t,price_mean,"b-",linewidth=2,label="Stock Price")
ax1.set_ylabel('Stock Price')
ax1.set_title('Average stock price and portfolio')
ax2 = ax1.twinx()  # this is the important function
ax2.plot(t,port_mean,"g-",linewidth=2,label="Portfolio")
# ax2.set_xlim([0, np.e])
ax2.set_ylabel('Portfolio')
ax2.set_xlabel('Trading time')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.grid()
#plt.show()
plt.savefig('img/price_port_'+MODEL_PATH+'.jpg')


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
