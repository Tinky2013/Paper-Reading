import pandas as pd
import numpy as np

#df = pd.read_csv(r'2019_sz5.csv')
df = pd.read_csv(r'hd5_sz50/2019_sz50n.csv')

nor_dt = df.drop(['time','thscode','amount','CLOSE_AFTER'], axis=1)
id_dt = df[['time','thscode','amount','CLOSE_AFTER']]

# nor_dt = nor_dt.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

nor_dt = nor_dt.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

data_normalized = pd.concat([id_dt,nor_dt],axis=1)

#data_normalized.to_csv('2019_sz5s.csv',index=False)
data_normalized.to_csv('hd5_sz50/2019_sz50s.csv',index=False)