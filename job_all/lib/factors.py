import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import attr
import warnings
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv("D:\\workdata\\data\\btcusdt_1d.csv",index_col="Unnamed: 0")

def ts_sum(df ,window=10):
    return df.rolling(window).sum()

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
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    return na.prod(na)

def product(df ,window=10):
    return df.rolling(window).apply(rolling_prod)

def ts_min(df ,window=10):
    return df.rolling(window).min()

def ts_max(df ,window=10):
    return df.rolling(window).max()

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
    return df.rolling(window).apply(np.argmax ) +1

def ts_argmin(df ,window=10):
    return df.rolling(window).apply(np.argmin ) +1

def decay_linear(df ,period=10):
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
        self.returns = self.close-self.close.shift(1)

    #   每个因子的计算公式：
    #   alpha001:(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)

    def alpha001(self,n=10):
        inner = self.close
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5),n)

    #  alpha002:(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))

    def alpha002(self,n=10):
        df_1 = -1 * correlation(rank(delta(log(self.volume), 2),n), rank((self.close - self.open) / self.open,n), 6)
        return df_1.replace([-np.inf, np.inf], 0)

    # alpha003:(-1 * correlation(rank(open), rank(volume), 10))

    def alpha003(self,n=10):
        df = -1 * correlation(rank(self.open,n), rank(self.volume,n), 10)
        return df.replace([-np.inf, np.inf], 0)

    # alpha004: (-1 * Ts_Rank(rank(low), 9))

    def alpha004(self,n=10):
        return -1 * ts_rank(rank(self.low,n), 9)

    #  alpha006: (-1 * correlation(open, volume, 10))

    def alpha006(self,n=10):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0)

    def alpha007(self,n=10):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    # alpha008: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))

    def alpha008(self,n=10):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)),n))

    # alpha009:((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))

    def alpha009(self,n=10):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    # alpha010: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))

    def alpha010(self,n=10):
        data_x=[y if x<0 else z for x,y,z in zip((ts_max(delta(self.close),4)),(delta(self.close)),(-1*delta(self.close)))]
        data_y=[y if x>0 else z for x,y,z in zip((ts_min(delta(self.close),4)),(delta(self.close)),(data_x))]
        df=pd.Series(data_y,name="value")
        return rank(df,n)

    #  alpha012:(sign(delta(volume, 1)) * (-1 * delta(close, 1)))

    def alpha012(self,n=10):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # alpha013:(-1 * rank(covariance(rank(close), rank(volume), 5)))

    def alpha013(self,n=10):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    #  alpha014:((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))

    def alpha014(self,n=10):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * rank(delta(self.returns, 3),n) * df

    # alpha015:(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))

    def alpha015(self,n=10):
        df = correlation(rank(self.high,n), rank(self.volume,n), 3)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * ts_sum(rank(df,n), 3)

    #  alpha016:(-1 * rank(covariance(rank(high), rank(volume), 5)))

    def alpha016(self,n=10):
        return -1 * rank(covariance(rank(self.high,n), rank(self.volume,n), 5),n)

    # alpha017: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))

    def alpha017(self,n=10):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10),n) *
                     rank(delta(delta(self.close, 1), 1),n) *
                     rank(ts_rank((self.volume / adv20), 5),n))

    # alpha018: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))

    def alpha018(self,n=10):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) +
                          df,n))

    #  alpha019:((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))

    def alpha019(self,n=10):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250),n)))

    # alpha020: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))

    def alpha020(self,n=10):
        return -1 * (rank(self.open - delay(self.high, 1),n) *
                     rank(self.open - delay(self.close, 1),n) *
                     rank(self.open - delay(self.low, 1),n))

    # alpha012: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))

    def alpha021(self,n=10):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    # alpha022:(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))

    def alpha022(self,n=10):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20),n)

    # alpha023: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

    def alpha023(self,n=10):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index,columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2)
        return alpha

    # alpha024: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))

    def alpha024(self,n=10):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    #   alpha026:(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self,n=10):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * ts_max(df, 3)

    # alpha028:scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    def alpha028(self,n=10):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # alpha029:(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self,n=10):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5),n),n),n), 2))),n),n), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    # alpha0230:(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

    def alpha030(self,n=10):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner,n)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    # alpha031:((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self,n=10):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12)
        df = df.replace([-np.inf, np.inf], 0)
        return ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10),n),n)), 10),n),n),n) +rank((-1 * delta(self.close, 3)),n)) + sign(scale(df)))

    # alpha033: rank((-1 * ((1 - (open / close))^1)))

    def alpha033(self,n=10):
        return rank(-1 + (self.open / self.close),n)

    # alpha034: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

    def alpha034(self,n=10):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1)
        return rank(2 - rank(inner,n) - rank(delta(self.close, 1),n),n)

    # alpha035:((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))

    def alpha035(self,n=10):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    # alpha037:(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

    def alpha037(self,n=10):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200),n) + rank(self.open - self.close,n)

    # alpha038: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

    def alpha038(self,n=10):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1)
        return -1 * rank(ts_rank(self.open, 10),n) * rank(inner,n)

    # alpha039:((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))

    def alpha039(self,n=10):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decay_linear(self.volume / adv20, period=9),n)),n)) *(1 + rank(ts_sum(self.returns, 250),n)))

    # alpha040: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

    def alpha040(self,n=10):
        return -1 * rank(stddev(self.high, 10),n) * correlation(self.high, self.volume, 10)

    # alpha43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

    def alpha043(self,n=10):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    # alpha04: (-1 * correlation(high, rank(volume), 5))

    def alpha044(self,n=10):
        df = correlation(self.high, rank(self.volume,n), 5)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * df

    # alpha045: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))

    def alpha045(self,n=10):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * (rank(sma(delay(self.close, 5), 20),n) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2),n))

    # alpha046: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))

    def alpha046(self,n=10):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # alpha049:(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha049(self,n=10):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    # alpha051:(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    def alpha051(self,n=10):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    # alpha052: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))

    def alpha052(self,n=10):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220),n)) * ts_rank(self.volume, 5))

    # alpha053:(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))

    def alpha053(self,n=10):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # alpha054:((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    def alpha054(self,n=10):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # alpha055: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))

    def alpha055(self,n=10):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / (divisor)
        df = correlation(rank(inner,n), rank(self.volume,n), 6)
        return -1 * df.replace([-np.inf, np.inf], 0)

    # alpha060: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))

    def alpha060(self,n=10):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner,n))) - scale(rank(ts_argmax(self.close, 10),n)))
'''
a=list(range(1,61))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    else:
        alpha_test.append("Alpha.alpha0"+str(x))
#print(alpha_test)
#print(alpha_use)
for func in alpha_test:
    print(func)
    try:
        eval(func)()
    except AttributeError:
        pass
'''













