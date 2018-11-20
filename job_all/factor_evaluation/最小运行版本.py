import sys

sys.path.append('..')
import pandas as pd
import copy
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def ts_sum(df ,window=10): #移动求和
    return df.rolling(window).sum()

def max_s(x,y): #从两个序列中选择取值较大的那个构成合成一个序列
    value_list=[a if a>b else b for a,b in zip(x,y)]
    return pd.Series(value_list,name="max")

def min_s(x,y): #从两个序列中选择取值较小的那个构成合成一个序列
    value_list = [a if a < b else b for a, b in zip(x, y)]
    return pd.Series(value_list,name="min")

def sma(df ,window=10): #移动平均
    return df.rolling(window).mean()

def stddev(df ,window=10):  #求移动标准差
    return df.rolling(window).std()

def correlation(x ,y ,window=10):  #求移动相关系数
    return x.rolling(window).corr(y)

def covariance(x ,y ,window=10):  #求移动协方差
    return x.rolling(window).cov(y)

def rolling_rank(na): #对序列进行排序，得到最后一个值得排名
    return rankdata(na)[-1]

def ts_rank(df,window=10):  #利用上一个函数进行移动排序
    return window+1-df.rolling(window).apply(rolling_rank)

def rolling_prod(na):  #对序列进行累计求积
    return na.prod(na)

def product(df ,window=10): #利用上一个函数实现移动求积
    return df.rolling(window).apply(rolling_prod)

def ts_min(df ,window=10): #求移动最小值
    return df.rolling(window).min()

def ts_max(df ,window=10): #求移动最大值
    return df.rolling(window).max()

def ts_count(x,y,window=10):  #求移动y比x大的个数
    diff=y-x
    diff[diff<0]=np.nan
    result=diff.rolling(window).count()
    result[:window-1]=np.nan
    return result

def delta(df ,period=1): #求序列的差分
    return df.diff(period)

def delay(df ,period=1):
    return df.shift(period)

def ts_lowday(df,window=10): #计算序列前 n 期时间序列中最小值距离当前时点的间隔
    return (window-1)-df.rolling(window).apply(np.argmin)

def ts_highday(df,window=10): #计算序列前 n 期时间序列中最大值距离当前时点的间隔
    return (window-1)-df.rolling(window).apply(np.argmax)

#上面这部分是计算因子可能用到的一些函数

class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.volume = pn_data['volume']
        self.amount=pn_data['amount']
        self.returns = self.close-self.close.shift(1)

    def alpha059(self):
        data_mid1=[z if x>y else v for x,y,z,v in zip(self.close,delay(self.close),min_s(self.low,delay(self.close)),max_s(self.high,delay(self.close)))]
        data_mid1=np.array(data_mid1)
        data_mid1=self.close.values-data_mid1
        data_mid2=[0 if x==y else z for x,y,z in zip(self.close,delay(self.close),data_mid1)]
        data_mid2=pd.Series(data_mid2,name="values")
        return ts_sum(data_mid2,20)

    def alpha118(self):
        return ts_sum((self.high-self.open),20)/ts_sum((self.open-self.low),20)*100

#上面是两个因子的计算方式

def calc_alpha_signal(df, param, corr, lower_bound, upper_bound):
    '''
    :param df: 计算出来包含因子值得DataFrame
    :param param: DataFrame当中的因子值的名称，格式为字符串
    :param corr: 指该因子与未来收益的相关性，当corr大于0则因子值较大时做多反之做空；corr小于0时则反向操作，取值可为任意正数或负数
    :param lower_bound: 因子值的下界，也就是最小10%的边界，大于此边界则超出10%
    :param upper_bound: 因子值得上界，也就是最大10%的边界，小于此边界则低出10%
    :return: 返回计算后的DataFrame的最后一行，也就是最近的一个信号
    '''
    if corr > 0:
        df["open_long_signal"] = (df[param].shift(1) >= upper_bound*(1)) * 1
        df['close_long_signal'] = (df[param].shift(1) < upper_bound*(1)) * 1
        df["open_short_signal"] = (df[param].shift(1) <= lower_bound*(1)) * 1
        df["close_short_signal"] = (df[param].shift(1) > lower_bound*(1)) * 1
        df['buy_price'] = df['open']
        df['sell_price'] = df['open']
    else:
        df["open_long_signal"] = (df[param].shift(1) <= lower_bound*(1)) * 1
        df['close_long_signal'] = (df[param].shift(1) > lower_bound*(1)) * 1
        df["open_short_signal"] = (df[param].shift(1) >= upper_bound*(1)) * 1
        df["close_short_signal"] = (df[param].shift(1) < upper_bound*(1)) * 1
        df['buy_price'] = df['open']
        df['sell_price'] = df['open']
    return df.iloc[-1]
#上面是根据因子计算信号

'''
df=pd.read_csv("../factor_calculation/api_data/BITMEX_xbtusd_4h_2017-01-01_2018-10-01.csv")
#print(df)
'''
path="../factor_calculation/api_data/BITMEX_xbtusd_4h_2017-01-01_2018-10-01.csv"




start_day = pd.to_datetime('2017-08-18')
end_day = pd.to_datetime('2018-10-01')

def cal_factor(factor):
    '''

    :param factor: 输入字符串，其为因子的名称，如"alpha059"
    :return: 返回包含因子值、日期和高开低收的DataFrame
    '''
    dataf=pd.read_csv(path)
    Alpha = Alphas(dataf)
    factor = "Alpha."+factor
    df_m = copy.deepcopy(dataf)
    df_m[factor] = eval(factor)()
    return df_m[[factor, "date", "open", "close", "high", "low"]]

#print(cal_factor("alpha059"))
factor_1=cal_factor("alpha059")
factor_2=cal_factor("alpha118")
#print(factor_1)
factor_1["date"] = pd.to_datetime(factor_1["date"])
factor_1_cols=factor_1.columns

factor_2["date"] = pd.to_datetime(factor_2["date"])
factor_2_cols=factor_2.columns

'''
bin_alpha_1 = pd.qcut(factor_1[(factor_1['date'] >= start_day) & (factor_1['date'] < end_day)][factor_1_cols[0]], q=10, retbins=True,duplicates="drop")[1]
print(bin_alpha_1)
bin_alpha_2 = pd.qcut(factor_2[(factor_2['date'] >= start_day) & (factor_2['date'] < end_day)][factor_2_cols[0]], q=10, retbins=True,duplicates="drop")[1]
print(bin_alpha_2)
'''


print(calc_alpha_signal(factor_1,factor_1_cols[0],-1, -920.48, 1300.1))
print(calc_alpha_signal(factor_2, factor_2_cols[0], 1, 52.27, 166.45))



























