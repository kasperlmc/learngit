import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import talib as ta
import matplotlib.pyplot as plt
from sklearn import linear_model

ss = StandardScaler()
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

alphas=["Alpha.alpha002_gtja","Alpha.alpha049_gtja","Alpha.alpha050_gtja","Alpha.alpha051_gtja","Alpha.alpha112_gtja"]
params=["Alpha.alpha002","Alpha.alpha049","Alpha.alpha050","Alpha.alpha051","Alpha.alpha112"]
for i in range(len(alphas)):
    df = pd.read_csv('../factor_writedb/btcusdt_' + alphas[i] + '.csv', index_col=0)
    factor=df[[params[i]]]
    if i==0:
        df_factors=factor
    else:
        df_factors=pd.concat([df_factors,factor],axis=1)

'''
columns_list=df_factors.columns
for i in range(len(columns_list)):
    df_factors[columns_list[i]]=ta.EMA(df_factors[columns_list[i]].values, timeperiod=5)
'''



df_factors["forward_ret3"]=df["close"].values
#print(df_factors["forward_ret3"])
df_factors=df_factors.dropna()
#print(df_factors)
columns_list=df_factors.columns
df_factors[columns_list]=ss.fit_transform(df_factors[columns_list])
#print(df_factors)

class BoostModel:
    def __init__(self,max_depth=3,subsample=0.95, num_round=1000):
        self.params = {'max_depth': max_depth, 'eta': 0.1, 'silent': 0, 'alpha': 0.5, 'lambda': 0.5,'eval_metric': 'rmse', 'subsample': subsample, 'objective': 'reg:linear'}
        self.num_round = num_round

    def fit(self, train_data, train_label):
        '''
        训练模型
        参数:
            train_data, train_label：分别对应着训练特征数据，训练标签
        返回:
            boost_model, 训练后的xgboost模型
        '''

        dtrain = xgb.DMatrix(train_data, label=train_label)

        boost_model = xgb.train(self.params, dtrain, num_boost_round=self.num_round)
        self.boost_model = boost_model

        return boost_model

    def get_resid(self, test_data, test_label):
        '''
        预测
        参数:
            test_data：待预测的特征数据
        返回:
            resid: 测试集的残差部分
        '''

        dtest = xgb.DMatrix(test_data)
        predict_score = self.boost_model.predict(dtest)

        resid = predict_score
        r2 = r2_score(test_label, predict_score)

        return resid, r2

y=df_factors["forward_ret3"].values
x=df_factors.iloc[:,:-1].values
#print(y)



'''
predict_value=np.zeros((2434,))

params = {'max_depth': 3, 'eta': 0.1, 'silent': 0, 'alpha': 0.5, 'lambda': 0.5,'eval_metric': 'rmse', 'objective': 'reg:linear', 'nthread': 32}
kf = KFold(n_splits=20,shuffle=True)
for train_index, test_index in kf.split(x):
    train_x=x[train_index]
    train_y=y[train_index]
    dtrain = xgb.DMatrix(train_x, label=train_y)
    boost_model=xgb.train(params,dtrain,num_boost_round=1000)
    test_x=x[test_index]
    dtest=xgb.DMatrix(test_x)
    predict_value[test_index]=boost_model.predict(dtest)

df_factors["predic_value"]=predict_value
'''

boost_model = BoostModel()
xgt_model = boost_model.fit(x, y)
xgt_resid, xgt_r2 = boost_model.get_resid(x, y)
print(xgt_r2)
df_factors["resid"]=xgt_resid
df_factors[["resid","forward_ret3"]].plot()
plt.show()




























