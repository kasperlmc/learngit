import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from functools import reduce
import warnings
import copy
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

# data=pd.read_csv("D:\\workdata\\data\\btcusdt_1d.csv",index_col="Unnamed: 0")
#print(data.head())


def ts_sum(df ,window=10):
    return df.rolling(window).sum()


def max_s(x,y):
    value_list=[a if a>b else b for a,b in zip(x,y)]
    return pd.Series(value_list,name="max")


def min_s(x,y):
    value_list = [a if a < b else b for a, b in zip(x, y)]
    return pd.Series(value_list,name="min")


def sma(df ,window=10):
    return df.rolling(window).mean()


def stddev(df ,window=10):
    return df.rolling(window).std()


def correlation(x ,y ,window=10):
    return x.rolling(window).corr(y)


def covariance(x ,y ,window=10):
    return x.rolling(window).cov(y)


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(df,window=10):
    return window+1-df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    return na.prod(na)


def product(df ,window=10):
    return df.rolling(window).apply(rolling_prod)


def ts_min(df ,window=10):
    return df.rolling(window).min()


def ts_max(df ,window=10):
    return df.rolling(window).max()


def ts_count(x,y,window=10):
    diff=y-x
    diff[diff<0]=np.nan
    result=diff.rolling(window).count()
    result[:window-1]=np.nan
    return result


def delta(df ,period=1):
    return df.diff(period)


def delay(df ,period=1):
    return df.shift(period)


def ranks(df):
    #print(df.rank(pct=True))
    return df.rank(pct=True).values[-1]


def rank(df,window=10):
    return df.rolling(window).apply(ranks,raw=False)


def scale(df ,k=1):
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df ,window=10):
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df ,window=10):
    return df.rolling(window).apply(np.argmin) + 1


def ts_lowday(df,window=10):
    return (window-1)-df.rolling(window).apply(np.argmin)


def ts_highday(df,window=10):
    return (window-1)-df.rolling(window).apply(np.argmax)


def SMA(vals, n, m):
    # 算法1
    return reduce(lambda x, y: ((n - m) * x + y * m) / n, vals)


def sma_list(df, n, m):
    result_list = [np.nan]
    for x in range(1, len(df)):
        if df.values[x-1]*0 == 0:
            value = SMA([df.values[x - 1], df.values[x]], n, m)
            result_list.append(value)
        elif df.values[x-1]*0 != 0:
            result_list.append(np.nan)
        elif df.values[x-2]*0 != 0:
            value = SMA([df.values[x - 1], df.values[x]], n, m)
            result_list.append(value)
        else:
            value = SMA([result_list[-1], df.values[x]], n, m)
            result_list.append(value)
    result_series = pd.Series(result_list, name="sma")
    return result_series


def decay_linear(df, period=10):
    if df.isnull().values.any():
        df.fillna(method='ffill' ,inplace=True)
        df.fillna(method='bfill' ,inplace=True)
        df.fillna(value=0 ,inplace=True)

    na_lwma=df.values
    y = list(range(1, period+1))
    y.reverse()
    y=np.array(y)
    y=y/y.sum()
    value_list=[np.nan]*(period-1)
    for pos in range(period,len(na_lwma)):
        value=na_lwma[pos-period:pos]
        value=value*y
        value_list.append(value.sum())
    return pd.Series(value_list,name="close")


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

    def alpha001(self):
        data_x=rank(delta(log(self.volume),1))
        data_y=rank(((self.close-self.open)/self.open))
        data=correlation(data_x,data_y,6)*-1
        return data

    def alpha002(self):
        data_m=delta((((self.close-self.low)-(self.high-self.close))/(self.high-self.low)),1)
        data_m=data_m*-1
        return data_m

    def alpha003(self):
        data_mid1=min_s(self.low,delay(self.close,1))
        data_mid2=max_s(self.high,delay(self.close,1))
        data_mid3=[z if x>y else v for x,y,z,v in zip(self.close,delay(self.close,1),data_mid1,data_mid2)]
        data_mid3=np.array(data_mid3)
        data_mid4=self.close-data_mid3
        data_mid5=[0 if x==y else z for x,y,z in zip(self.close,delay(self.close,1),data_mid4)]
        data_mid5=np.array(data_mid5)
        df=pd.Series(data_mid5,name="value")
        return ts_sum(df,6)

    def alpha004(self):
        data_mid1=self.volume/(sma(self.volume,20))
        data_mid2=[1 if x>=1 else -1 for x in data_mid1]
        data_mid3=[1 if x<y else z for x,y,z in zip((ts_sum(self.close,2)/2),((ts_sum(self.close,8)/8)-(stddev(self.close,8))),data_mid2)]
        data_mid4=[-1 if x<y else z for x,y,z in zip((ts_sum(self.close,8)/8+stddev(self.close,8)),(ts_sum(self.close,2)/2),(data_mid3))]
        return pd.Series(data_mid4,name="value")

    def alpha005(self):
        data_mid1=correlation(ts_rank(self.volume,5),ts_rank(self.high,5),5)
        return -1*ts_max(data_mid1)

    def alpha006(self):
        return -1*(rank(sign(delta((self.open*0.85+self.high*0.15),4))))

    def alpha009(self):
        data_mid1=((self.high+self.low)/2-(delay(self.high)+delay(self.low))/2)*(self.high-self.low)/self.volume
        return sma_list(data_mid1, 7, 2)

    def alpha011(self):
        data_mid1=((self.close-self.low)-(self.high-self.low))/(self.high-self.low)*self.volume
        return ts_sum(data_mid1, 6)

    def alpha014(self):
        return self.close-delay(self.close,5)

    def alpha015(self):
        return self.open/delay(self.close)-1

    def alpha018(self):
        return self.close/delay(self.close,5)

    def alpha019(self):
        data_mid1=[0 if x==y else z for x,y,z in zip((self.close),(delay(self.close,5)),(self.close-delay(self.close,5))/self.close)]
        data_mid2=[z if x<y else v for x,y,z,v in zip(self.close,delay(self.close,5),(self.close-delay(self.close,5))/delay(self.close,5),data_mid1)]
        return pd.Series(data_mid2,name="value")

    def alpha020(self):
        return (self.close-delay(self.close,6))/delay(self.close,6)*100

    def alpha024(self):
        data_mid = self.close-delay(self.close,5)
        return sma_list(data_mid, 5, 1)

    def alpha028(self):
        data_mid1 = (self.close-ts_min(self.low,9)/(ts_max(self.high,9)-ts_min(self.low,9))*100)
        data_mid1 = sma_list(data_mid1, 3, 1)
        data_mid2 = (self.close-ts_min(self.low,9))/(ts_max(self.high,9)-ts_max(self.low,9))*100
        data_mid2 = sma_list(data_mid2, 3, 1)
        data_mid3 = sma_list(data_mid2, 3, 1)
        return 3*data_mid1-2*data_mid3

    def alpha029(self):
        return (self.close-delay(self.close,6))/delay(self.close,6)*self.volume

    def alpha031(self):
        return (self.close-sma(self.close,12))/sma(self.close,12)*100

    def alpha032(self):
        return -1*ts_sum((rank(correlation(rank(self.high),rank(self.volume),3))),3)

    def alpha033(self):
        data_mid1=-1*ts_min(self.low,5)+delay(ts_min(self.low,5),5)
        data_mid2=rank((ts_sum(self.returns,240)-ts_sum(self.returns,20))/220)
        return data_mid1*data_mid2*ts_rank(self.volume,5)

    def alpha034(self):
        return sma(self.close,12)/self.close

    def alpha035(self):
        data_mid1=rank(decay_linear(delta(self.open),15))
        data_mid2=rank(decay_linear(correlation(self.volume,self.open,17),7))
        return min_s(data_mid1,data_mid2)*-1

    def alpha037(self):
        data_mid1=ts_sum(self.open,5)*ts_sum(self.returns,5)
        data_mid2=delay((ts_sum(self.open,5)*ts_sum(self.returns,5)),10)
        return rank(data_mid1-data_mid2)*-1

    def alpha038(self):
        data=[z if x<y else 0 for x,y,z in zip(ts_sum(self.high,20)/20,self.high,(-1*delta(self.high,2)))]
        return pd.Series(data,name="value")

    def alpha040(self):
        data_mid1=copy.deepcopy(self.volume)
        data_mid1=[0.001 if x<=y else z for x,y,z in zip(self.close,delay(self.close),data_mid1)]
        data_mid1=pd.Series(data_mid1,name="value")
        data_mid2=copy.deepcopy(self.volume)
        data_mid2=[0.001 if x>y else z for x,y,z in zip(self.close,delay(self.close),data_mid2)]
        data_mid2=pd.Series(data_mid2,name="value")
        return ts_sum(data_mid1,26)/ts_sum(data_mid2,26)

    def alpha042(self):
        return -1*rank(stddev(self.high,10))*correlation(self.high,self.volume,10)

    def alpha043(self):
        data_mid1=-1*copy.deepcopy(self.volume)
        data_mid1[self.close>=delay(self.close)]=0
        data_mid2=copy.deepcopy(self.volume)
        data_mid2[self.close<=delay(self.close)]=data_mid1
        return ts_sum(data_mid2,6)

    def alpha046(self):
        data_mid1=sma(self.close,3)+sma(self.close,6)+sma(self.close,12)+sma(self.close,24)
        return data_mid1/(4*self.close)

    def alpha047(self):
        data_mid1=(ts_max(self.high,6)-self.close)
        data_mid2=ts_max(self.high,6)-ts_min(self.low,6)
        data_mid3 = sma_list(data_mid1/data_mid2, 9, 1)
        return 100*data_mid3

    def alpha049(self):
        data_mid1=[0 if x>=y else z for x,y,z in zip((self.high+self.low),(delay(self.high)+delay(self.low)),(max_s((self.high-delay(self.high)).abs(),(self.low-delay(self.low)).abs())))]
        data_mid1=pd.Series(data_mid1,name="values")
        data_mid1=ts_sum(data_mid1,12)

        data_mid2=[0 if x<=y else z for x,y,z in zip((self.high+self.low),(delay(self.high)+delay(self.low)),(max_s((self.high-delay(self.high)).abs(),(self.low-delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)

        return data_mid1/(data_mid1+data_mid2)

    def alpha050(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid1 = ts_sum(data_mid1, 12)

        data_mid2 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)

        data_mid3=data_mid1/(data_mid1+data_mid2)

        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)

        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)

        data_mid6=data_mid4/(data_mid4+data_mid5)

        return data_mid6-data_mid3

    def alpha051(self):
        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)

        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)),(max_s((self.high - delay(self.high)).abs(),(self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)

        data_mid6 = data_mid4 / (data_mid4 + data_mid5)

        return data_mid6

    def alpha052(self):
        data_mid1 = self.high-delay((self.high+self.low+self.close)/3)
        data_mid1[data_mid1 < 0] = 0
        data_mid2=delay((self.high+self.low+self.close)/3)-self.low
        data_mid2[data_mid2 < 0] = 0
        return ts_sum(data_mid1, 26)/ts_sum(data_mid2, 26)*100

    def alpha053(self):
        data_mid1 = ts_count(delay(self.close), self.close, 12)
        return (data_mid1/12)*100

    def alpha055(self):
        data_mid1=(self.close-delay(self.close)+(self.close-self.open)/2+delay(self.close)-delay(self.open))*16

        data_mid_z=(self.high-delay(self.close)).abs()+(self.low-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4
        data_mid_vz=(self.low-delay(self.close)).abs()+(self.high-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4
        data_mid_vv=(self.high-delay(self.low)).abs()+(delay(self.close)-delay(self.open))/4

        data_mid_v=[vz if x1>y1 and x2>y2 else vv for x1,y1,x2,y2,vz,vv in zip((self.low-delay(self.close)).abs(),(self.high-delay(self.low)).abs(),(self.low-delay(self.close)).abs(),(self.high-delay(self.close)).abs(),data_mid_vz,data_mid_vv)]
        data_mid2=[z if x1>y1 and x2>y2 else v for x1,y1,x2,y2,z,v in zip((self.high-delay(self.close)).abs(),(self.low-delay(self.close)).abs(),(self.high-delay(self.close)).abs(),(self.high-delay(self.low)).abs(),data_mid_z,data_mid_v)]

        data_mid3=max_s((self.high-delay(self.close)).abs(),(self.low-delay(self.close)).abs())

        data_all=data_mid1/data_mid2*data_mid3

        return ts_sum(data_all,20)

    def alpha057(self):
        data_mid1=(self.close-ts_min(self.low,9))/(ts_max(self.high,9)-ts_min(self.low,9))*100
        return sma_list(data_mid1, 3, 1)

    def alpha058(self):
        data_mid1=ts_count(delay(self.close),self.close,20)
        return (data_mid1/20)*100

    def alpha059(self):
        data_mid1=[z if x>y else v for x,y,z,v in zip(self.close,delay(self.close),min_s(self.low,delay(self.close)),max_s(self.high,delay(self.close)))]
        data_mid1=np.array(data_mid1)
        data_mid1=self.close.values-data_mid1
        data_mid2=[0 if x==y else z for x,y,z in zip(self.close,delay(self.close),data_mid1)]
        data_mid2=pd.Series(data_mid2,name="values")
        return ts_sum(data_mid2,20)

    def alpha060(self):
        data_mid1=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        return ts_sum(data_mid1,20)

    def alpha063(self):
        data_mid1=[0 if x<=0 else x for x in (self.close-delay(self.close))]
        data_mid1=pd.Series(data_mid1,name="values")
        data_mid2=(self.close-delay(self.close)).abs()
        return ((sma_list(data_mid1, 6, 1))/(sma_list(data_mid2, 6, 1)))*100

    def alpha065(self):
        return sma(self.close,6)/self.close

    def alpha066(self):
        return (self.close-sma(self.close,6))/sma(self.close,6)*100

    def alpha067(self):
        data_mid1 = [0 if x <= 0 else x for x in (self.close - delay(self.close))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid2 = (self.close - delay(self.close)).abs()
        return ((sma_list(data_mid1, 24, 1)) / (sma_list(data_mid2, 24, 1))) * 100

    def alpha068(self):
        data_mid1=((self.high+self.low)/2-(delay(self.high)+delay(self.low))/2)*(self.high-self.low)/self.volume
        return sma_list(data_mid1, 15, 2)

    def alpha069(self):
        dtm=[0 if x<=y else z for x,y,z in zip(self.open,delay(self.open),max_s((self.high-self.open),(self.open-delay(self.open))))]
        dbm=[0 if x>=y else z for x,y,z in zip(self.open,delay(self.open),max_s((self.open-self.low),(self.open-delay(self.open))))]
        dtm=pd.Series(dtm,name="dtm")
        dbm=pd.Series(dbm,name="dbm")
        data_mid_z=(ts_sum(dtm,20)-ts_sum(dbm,20))/ts_sum(dtm,20)
        data_mid_vz=(ts_sum(dtm,20)-ts_sum(dbm,20))/ts_sum(dbm,20)

        data_mid_v=[0 if x==y else z for x,y,z in zip(ts_sum(dtm,20),ts_sum(dbm,20),data_mid_vz)]
        data_mid=[z if x>y else v for x,y,z,v in zip(ts_sum(dtm,20),ts_sum(dbm,20),data_mid_z,data_mid_v)]

        return pd.Series(data_mid,name="values")

    def alpha070(self):
        return stddev(self.amount,6)

    def alpha071(self):
        return (self.close-sma(self.close, 24))/sma(self.close, 24)*100

    def alpha072(self):
        data_mid1=(ts_max(self.high,6)-self.close)/(ts_max(self.high,6)-ts_min(self.low,6))*100
        return sma_list(data_mid1, 15, 1)

    def alpha076(self):
        data_mid1=stddev((self.close/delay(self.close)-1).abs()/self.volume,20)
        data_mid2=sma((self.close/delay(self.close)-1).abs()/self.volume,20)
        return data_mid1/data_mid2

    def alpha078(self):
        data_mid1=(self.high+self.low+self.close)/3+sma((self.high+self.low+self.close)/3,12)
        data_mid2=0.015*sma((self.close-sma((self.high+self.low+self.close)/3,12)).abs(),12)

        return data_mid1/data_mid2

    def alpha079(self):
        data_mid1=self.close-delay(self.close)
        data_mid1[data_mid1<0]=0
        data_mid2 = (self.close-delay(self.close)).abs()
        return (sma_list(data_mid1, 12, 1))/(sma_list(data_mid2, 12, 1))*100

    def alpha080(self):
        return (self.volume-delay(self.volume,5))/delay(self.volume,5)*100

    def alpha081(self):
        return sma_list(self.volume, 21, 2)

    def alpha082(self):
        data_mid1=ts_max(self.high,6)-self.close
        data_mid2=ts_max(self.high,6)-ts_min(self.low,6)
        return sma_list(data_mid1/data_mid2*100, 20, 1)

    def alpha084(self):
        data_mid_v=[-z if x<y else 0 for x,y,z in zip(self.close,delay(self.close),self.volume)]
        data_mid2=[z if x>y else v for x,y,z,v in zip(self.close,delay(self.close),self.volume,data_mid_v)]
        data_mid2=pd.Series(data_mid2,name="values")
        return ts_sum(data_mid2,20)

    def alpha085(self):
        data_mid1=ts_rank((self.volume/sma(self.volume,20)),20)
        data_mid2=ts_rank((-1*delta(self.close,7)),8)
        return data_mid1*data_mid2

    def alpha086(self):
        data_yx=(delay(self.close,20)-delay(self.close,10))/10-(delay(self.close,10)-self.close)/10
        data_y=[1 if x<0 else y for x,y in zip(data_yx,-1*(self.close-delay(self.close)))]
        data=[-1 if x>0.25 else y for x,y in zip(data_yx,data_y)]
        data=pd.Series(data,name="values")
        return data

    def alpha088(self):
        return (self.close-delay(self.close,20))/delay(self.close,20)*100

    def alpha089(self):
        data_mid1 = sma_list(self.close, 13, 2)-sma_list(self.close, 27, 2)-sma_list((sma_list(self.close, 13, 2)-sma_list(self.close, 27, 2)), 10, 2)
        return data_mid1*2

    def alpha093(self):
        data_mid1=[0 if x>=y else z for x,y,z in zip(self.open,delay(self.open),max_s((self.open-self.low),(self.open-delay(self.open))))]
        data_mid1=pd.Series(data_mid1,name="values")
        return ts_sum(data_mid1,20)

    def alpha094(self):
        data_mid_v=[-z if x<y else 0 for x,y,z in zip(self.close,delay(self.close),self.volume)]
        data=[z if x>y else v for x,y,z,v in zip(self.close,delay(self.close),self.volume,data_mid_v)]
        data=pd.Series(data,name="values")
        return ts_sum(data,30)/self.volume

    def alpha095(self):
        return stddev(self.amount,20)

    def alpha096(self):
        return sma_list((sma_list(((self.close-ts_min(self.low,9))/(ts_max(self.high,9)-ts_min(self.low,9))*100), 3, 1)), 3, 1)

    def alpha097(self):
        return stddev(self.volume,10)

    def alpha098(self):
        data_mid=[y if x<=0.05 else z for x,y,z in zip((delta((ts_sum(self.close,100)/100),100)/delay((self.close),100)),(-1*(self.close-ts_min(self.close,100))),(-1*(delta(self.close,3))))]
        return pd.Series(data_mid,name="values")

    def alpha100(self):
        return stddev(self.volume,20)

    def alpha102(self):
        data_mid=self.volume-delay(self.volume)
        data_mid[data_mid<0]=0
        data_mid2 = (self.volume-delay(self.volume)).abs()
        return (sma_list(data_mid, 6, 1))/(sma_list(data_mid2, 6, 1))*100

    def alpha103(self):
        return ((20-ts_lowday(self.low,20))/20)*100

    def alpha106(self):
        return self.close-delay(self.close,20)

    def alpha109(self):
        data_mid1 = sma_list(self.high-self.low, 10, 2)
        return data_mid1/sma_list(data_mid1, 10, 2)

    def alpha110(self):
        data_mid1=self.high-delay(self.close)
        data_mid1[data_mid1<0]=0
        data_mid2=delay(self.close)-self.low
        data_mid2[data_mid2<0]=0
        return (ts_sum(data_mid1,20))/(ts_sum(data_mid2,20))*100

    def alpha111(self):
        data_mid1 = ((2*self.close-self.low-self.high)/(self.high-self.low))*self.volume
        return sma_list(data_mid1, 11, 2)-sma_list(data_mid1, 4, 2)

    def alpha112(self):
        data_mid1=self.close-delay(self.close)
        data_mid1[data_mid1<0]=0
        data_mid2=self.close-delay(self.close)
        data_mid2[data_mid2>0]=0
        data_mid2=data_mid2.abs()
        return (ts_sum((data_mid1),12)-ts_sum((data_mid2),12))/(ts_sum(data_mid1,12)+ts_sum(data_mid2,12))*100

    def alpha117(self):
        return (ts_rank(self.volume,32)*(1-ts_rank((self.close+self.high-self.low),16)))*(1-ts_rank(self.returns,32))

    def alpha118(self):
        return ts_sum((self.high-self.open),20)/ts_sum((self.open-self.low),20)*100

    def alpha122(self):
        data_mid1 = sma_list((sma_list((sma_list((self.close.map(np.log)), 13, 2)), 13, 2)), 13, 2)
        return (data_mid1-delay(data_mid1))/(delay(data_mid1))

    def alpha126(self):
        return (self.close+self.high+self.low)/3

    def alpha128(self):
        data_mid1=(self.high+self.low+self.close)/3*self.volume
        data_mid1[(self.high+self.low+self.close)/3<=delay((self.high+self.low+self.close)/3)]=0
        data_mid2=(self.high+self.low+self.close)/3*self.volume
        data_mid2[(self.high+self.low+self.close)/3>=delay((self.high+self.low+self.close)/3)]=0
        return 100-(100/(1+ts_sum((data_mid1),14)/ts_sum((data_mid2),14)))

    def alpha129(self):
        data_mid1=(self.close-delay(self.close)).abs()/delay(self.close)
        data_mid1[self.close>=delay(self.close)]=0
        return ts_sum(data_mid1,12)

    def alpha132(self):
        return sma(self.amount,20)

    def alpha133(self):
        return ((20-ts_highday(self.high,20))/20)*100-((20-ts_lowday(self.low,20))/20)*100

    def alpha134(self):
        return (self.close-delay(self.close,12))/delay(self.close,12)

    def alpha135(self):
        data_mid1 = delay(self.close/delay(self.close, 20))
        return sma_list(data_mid1, 20, 1)

    def alpha137(self):
        data_mid1=(self.high-delay(self.low)).abs()-(delay(self.close)-delay(self.open)).abs()/4
        data_mid1[((self.low-delay(self.close)).abs()>(self.high-delay(self.low)).abs()) & ((self.low-delay(self.close)).abs()>(self.high-delay(self.close)).abs())]=(self.low-delay(self.close)).abs()+(self.high-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4
        data_mid1[((self.high-delay(self.close)).abs()>(self.low-delay(self.close)).abs()) & ((self.high-delay(self.close)).abs()>(self.high-delay(self.low)).abs())]=(self.high-delay(self.close)).abs()+(self.low-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4

        return 16*(self.close-delay(self.close)+(self.close-self.open)/2+delay(self.close)-delay(self.open))/data_mid1*max_s((self.high-delay(self.close)).abs(),(self.low-delay(self.close)).abs())

    def alpha139(self):
        return -1*correlation(self.open,self.volume,10)

    def alpha145(self):
        return (sma(self.volume,9)-sma(self.volume,26))/sma(self.volume,12)*100

    def alpha146(self):
        data_mid1=sma(((self.close-delay(self.close))/delay(self.close)-sma_list(((self.close-delay(self.close))/delay(self.close)), 61, 2)),20)
        data_mid2=(self.close-delay(self.close))/delay(self.close)-sma_list(((self.close-delay(self.close))/delay(self.close)), 61, 2)
        data_mid3=(((self.close-delay(self.close))/delay(self.close)-((self.close-delay(self.close))/delay(self.close)-((self.close-delay(self.close))/delay(self.close)-sma_list(((self.close-delay(self.close))/delay(self.close)), 61, 2))))**2)
        data_mid3 = sma_list(data_mid3, 61, 2)
        return data_mid1*data_mid2/data_mid3

    def alpha150(self):
        return (self.close+self.high+self.low)/(3*self.close)

    def alpha151(self):
        return sma_list((self.close-delay(self.close,20)), 20, 1)

    def alpha152(self):
        data_mid1=sma((delay(sma_list((delay(self.close/delay(self.close,9))), 9, 1))),12)
        data_mid2=sma((delay(sma_list((delay(self.close/delay(self.close,9))), 9, 1))),26)
        return sma_list(data_mid1-data_mid2, 9, 1)

    def alpha153(self):
        return (sma(self.close,3)+sma(self.close,6)+sma(self.close,12)+sma(self.close,24))/4

    def alpha155(self):
        return sma_list(self.volume, 13, 2)-sma_list(self.volume, 27, 2)-sma_list((sma_list(self.volume, 13, 2)-sma_list(self.volume, 27, 2)), 10, 2)

    def alpha158(self):
        return ((self.high-sma_list(self.close, 15, 2))-(self.low-sma_list(self.close, 15, 2)))/self.close

    def alpha159(self):
        data_mid1=(self.close-ts_sum((min_s(self.low,delay(self.close))),6))/(ts_sum(max_s(self.high,delay(self.close))-min_s(self.low,delay(self.close)),6))*12*24
        data_mid2=(self.close-ts_sum((min_s(self.low,delay(self.close))),12))/(ts_sum(max_s(self.high,delay(self.close))-min_s(self.low,delay(self.close)),12))*6*24
        data_mid3=(self.close-ts_sum((min_s(self.low,delay(self.close))),24))/(ts_sum(max_s(self.high,delay(self.close))-min_s(self.low,delay(self.close)),24))*6*24
        return (data_mid1+data_mid2+data_mid3)*100/(6*12+6*24+12*24)

    def alpha160(self):
        data_mid=stddev(self.close,20)
        data_mid[self.close>delay(self.close)]=0
        return sma_list(data_mid, 20, 1)

    def alpha161(self):
        return sma(max_s((max_s((self.high-self.low),(delay(self.close)-self.high).abs())),(delay(self.close)-self.low).abs()),12)

    def alpha162(self):
        data_mid1=self.close-delay(self.close)
        data_mid1[data_mid1<0]=0
        data_mid2=sma_list(data_mid1, 12, 1)/sma_list((self.close-delay(self.close)), 12, 1)*100
        data_mid3=copy.deepcopy(data_mid2)
        data_mid3[data_mid3>12]=12
        data_mid4=copy.deepcopy(data_mid2)
        data_mid4[data_mid4<12]=12
        return (data_mid2-data_mid3)/(data_mid4-data_mid3)

    def alpha164(self):
        data_mid1=1/(self.close-delay(self.close))
        data_mid1[self.close<=delay(self.close)]=1

        data_mid2=copy.deepcopy(data_mid1)
        data_mid2[data_mid2>12]=12

        return sma_list(((data_mid1-data_mid2)/(self.high-self.low)*100), 13, 2)

    def alpha167(self):
        data_mid=self.close-delay(self.close)
        data_mid[self.close<=delay(self.close)]=0
        return ts_sum(data_mid,12)/self.close

    def alpha168(self):
        return -1*self.volume/sma(self.volume,20)

    def alpha169(self):
        data_mid=delay(sma_list((self.close-delay(self.close)), 9, 1))
        return sma_list((sma(data_mid,12)-sma(data_mid,26)), 10, 1)

    def alpha171(self):
        return (-1*(self.low-self.close)*(self.open**5))/((self.close-self.high)*(self.close**5))

    def alpha172(self):
        tr=max_s(max_s((self.high-self.low),(self.high-delay(self.close)).abs()),(self.low-delay(self.close)).abs())
        hd=self.high-delay(self.high)
        ld=delay(self.low)-self.low

        data_mid1=copy.deepcopy(ld)
        data_mid1[(ld<=0) | (ld<=hd)]=0
        data_mid1=100*ts_sum(data_mid1,14)/ts_sum(tr,14)

        data_mid2=copy.deepcopy(hd)
        data_mid2[(hd<=0) | (hd<=ld)]=0
        data_mid2=100*ts_sum(data_mid2,14)/ts_sum(tr,14)

        data_mid3=(data_mid1-data_mid2).abs()/(data_mid1+data_mid2)*100

        return sma(data_mid3,6)

    def alpha173(self):
        return sma_list(self.close, 13, 2)*3-2*sma_list((sma_list(self.close, 13, 2)), 13, 2)+sma_list((sma_list((sma_list((log(self.close)), 13, 2)), 13, 2)), 13, 2)

    def alpha174(self):
        data_mid1=stddev(self.close,20)
        data_mid1[self.close<=delay(self.close)] = 0
        return sma_list(data_mid1, 20, 1)

    def alpha175(self):
        return sma(max_s(max_s((self.high-self.low),(delay(self.close)-self.high).abs()),(delay(self.close)-self.low).abs()),6)/self.close

    def alpha177(self):
        return 100*((20-ts_highday(self.high,20))/20)

    def alpha178(self):
        return self.volume*(self.close-delay(self.close))/delay(self.close)

    def alpha180(self):
        data_mid1=(-1*ts_rank((delta(self.close,7)).abs(),60))*sign(delta(self.close,7))
        data_mid1[sma(self.volume,20)>=self.volume]=-1*self.volume
        return data_mid1

    def alpha186(self):
        tr = max_s(max_s((self.high - self.low), (self.high - delay(self.close)).abs()),(self.low - delay(self.close)).abs())
        hd = self.high - delay(self.high)
        ld = delay(self.low) - self.low

        data_mid1 = copy.deepcopy(ld)
        data_mid1[(ld <= 0) | (ld <= hd)] = 0
        data_mid1 = 100 * ts_sum(data_mid1, 14) / ts_sum(tr, 14)

        data_mid2 = copy.deepcopy(hd)
        data_mid2[(hd <= 0) | (hd <= ld)] = 0
        data_mid2 = 100 * ts_sum(data_mid2, 14) / ts_sum(tr, 14)

        data_mid3 = (data_mid1 - data_mid2).abs() / (data_mid1 + data_mid2) * 100
        data_mid3_sma=sma(data_mid3,6)

        return (data_mid3_sma+delay(data_mid3_sma,6))/2

    def alpha187(self):
        data_mid1=max_s((self.high-self.open),(self.open-delay(self.open)))
        #data_mid1[self.open<=delay(self.open)]=0
        data_mid1=[0 if x<=y else z  for x,y,z in zip(self.open,delay(self.open),data_mid1)]
        data_mid1=pd.Series(data_mid1,name="value")
        return ts_sum(data_mid1,20)/self.close

    def alpha188(self):
        return ((self.high-self.low-sma_list((self.high-self.low), 11, 2))/(sma_list((self.high-self.low), 11, 2)))*100

    def alpha189(self):
        return sma((self.close-sma(self.close,6)).abs(),6)

    def alpha191(self):
        return correlation(sma(self.volume,20),self.low,5)+(self.high+self.low)/2-self.close

    def alpha192(self):
        data_mid1=-1*delta(self.close)
        data_mid1[ts_max(delta(self.close),5)<0]=delta(self.close)
        data_mid1[ts_min(delta(self.close),5)>0]=delta(self.close)
        return data_mid1

    def alpha193(self):
        return -1*(correlation(self.open,self.volume))

    def alpha194(self):
        return np.sign(delta(self.volume))*(-1)*(delta(self.close))

    def alpha195(self):
        data_mid1=[z if x<y else 0 for x,y,z in zip((ts_sum(self.high,20)/20),self.high,(-1*delta(self.high,2)))]
        data_mid1=pd.Series(data_mid1,name="values")
        return data_mid1

    def alpha196(self):
        data_mid1=-1*(self.close-delay(self.close))
        data_mid2=((delay(self.close,20)-delay(self.close,10))/10)-((delay(self.close,10)-self.close)/10)
        data_mid1[data_mid2<0]=1
        data_mid1[data_mid2>0.25]=-1
        return data_mid1

    def alpha197(self):
        data_mid1 = -1 * (self.close - delay(self.close))
        data_mid2 = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        data_mid1[data_mid2 < -0.1] = 1
        return data_mid1

    def alpha198(self):
        data_mid1 = -1 * (self.close - delay(self.close))
        data_mid2 = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        data_mid1[data_mid2 < -0.05] = 1
        return data_mid1

    def alpha199(self):
        data_mid1=((self.close-self.low)-(self.high-self.close))/(self.close-self.low)
        return -1*delta(data_mid1,9)

    def alpha200(self):
        data_mid1=-1*(self.low-self.close)*(self.open**5)
        data_mid2=(self.low-self.high)*(self.close**5)
        return data_mid1/data_mid2

    def alpha201(self):
        return (self.close-self.open)/((self.high-self.low)+0.001)


#Alpha=Alphas(data)
#alpha_use=Alpha.alpha162()
#print(alpha_use)
'''
a=list(range(1,192))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))
#print(alpha_test)
#print(alpha_use)
i=0
for func in alpha_test:
    print(func)
    try:
        eval(func)()
        i+=1
        print(i)
    except AttributeError:
        pass
'''























