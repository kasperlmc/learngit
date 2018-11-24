'''
a=False
b=True
if not b :
    print("1")

import numpy as np
a=[1,2,3,4,5]
a=[3,1,2]
a=np.array(a)
print(np.sort(a))
print(np.argsort(a))

n=115
s=48
print(np.random.rand(s))
rnd_num = np.random.rand(s).argsort()
print(rnd_num)
print(np.sort(rnd_num[:24]))

'''

import sys

sys.path.append('..')
import pandas as pd
import numpy as np
import copy
from lib.myfun import *
# coding=utf-8


'''
def resample_1(df, rule):
    df = df.resample(rule, on='date_time').apply({'open': 'first',
                                                  'high': 'max',
                                                  'low': 'min',
                                                  'close': 'last',
                                                  'volume': 'sum'})
    df = df.reset_index()
    nan_idx = df[df.isnull().any(axis=1)].index
    for idx in nan_idx:
        if idx > 0:
            df.loc[idx, ['open', 'high', 'low', 'close']] = df.loc[idx-1]['close']

    return df
exchange = 'BITMEX'
symbols = ['.bxbt']
#data=read_data(exchange, symbols[0], '4h', "2017-01-01", "2018-10-01")

data=pd.read_csv("BITMEX_xbtusd_4h_2017-01-01_2018-10-01.csv",index_col=0)

print(data)
#data.to_csv("BITMEX_xbtusd_4h_2017-01-01_2018-10-01.csv")
'''
'''
from sklearn.model_selection import ParameterGrid

param_grid={"N1":[10,15,20,25,30],"N2":[0.8,0.85,0.9,0.95],"N2-N3":[0.025],"N4":[1.025]}
print(list(ParameterGrid(param_grid)))

for x in list(ParameterGrid(param_grid)):
    print(x)
    print(x["N1"])
    break

a=1
b=1
d=0
c=0
if not ((a and c) or b):
    print("end")
'''

# '''
data["date_time"] = pd.to_datetime(data["date"])
data_test=copy.deepcopy(data)
#print(data_test)
df = data_test.resample(rule="4h", on='date_time').apply({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
#print(data_test["high"].resample(rule="4h",how="max",on="date_time"))

df["amount"]=df["volume"]
print(df)
df.to_csv("BITMEX_xbtusd_4h_2017-01-01_2018-10-01.csv")
# '''










