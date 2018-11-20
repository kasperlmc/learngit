import sys
import pandas as pd
import numpy as np
sys.path.append('..')
from lib.myfun import *
import copy


def PBOTest(rets, SampleTimes, block_len):
    '''
        Probability of Overfitting Test Based On Return Matrix and Sharpe Ratio
        Inputs:
            rets: Return Matrix with shape of TxN indicating N backtesting result using different params
            SampleTimes: Overfitting Indicator Sample Size
            block_len: Bootstrap Block Length
        Outputs:
            PBO: Probability of Overffiting
            POS: Probability of Getting Loss OutOfSample
            rd: Overfitting Indicator Logit
            ratios: Sharpe Ratios of Optimal InSample and of respective OutOfSample
    '''

    # Compute Through the Framework
    N = rets.shape[1]
    S = len(rets) // block_len
    M = rets[:(block_len * S)]
    Ms = M.reshape((S, block_len, N))
    training_part_block = S // 2 + S % 2
    rd = np.zeros(SampleTimes)
    ratios = np.zeros((SampleTimes, 2))
    for i in range(SampleTimes):
        rnd_num = np.random.rand(S).argsort()
        TPartOrder = np.sort(rnd_num[:training_part_block])
        VPartOrder = np.sort(rnd_num[training_part_block:])
        TJ = Ms[TPartOrder].reshape((training_part_block * block_len, N))
        VJ = Ms[VPartOrder].reshape(((S - training_part_block) * block_len, N))
        TSharpe = np.mean(TJ, axis=0) / np.std(TJ, axis=0)
        VSharpe = np.mean(VJ, axis=0) / np.std(VJ, axis=0)
        Vrelative_rank = (1 + VSharpe.argsort().argsort()) / (N + 1)
        # Compute rd
        T_n_best = np.argmax(TSharpe)
        VT_n_best_rank = Vrelative_rank[T_n_best]
        logit = np.log(VT_n_best_rank / (1 - VT_n_best_rank))
        rd[i] = logit
        # Compute Ratio
        ratios[i, 0] = TSharpe[T_n_best]
        ratios[i, 1] = VSharpe[T_n_best]

    PBO = np.sum(rd <= 0) / SampleTimes
    POS = np.sum(ratios[:, 1] <= 0) / SampleTimes

    return PBO, POS, rd, ratios


def MA(close, num=5):
    res = [close[0] * 1]
    for i in range(1, len(close)):
        if i >= num:
            res.append((res[-1] * num - close[i - num] + close[i]) / num)
        else:
            res.append((res[-1] * i + close[i]) / (i + 1))
    res = np.array(res)
    return res


def backtest(data, params=[], num=100, money=1e6):
    Fastlen, Slowlen, thdAB, stoploss = params

    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    hlen = len(close)

    profit = np.zeros(hlen)
    Ama = MA(close, Fastlen)
    Bma = MA(close, Slowlen)
    opened = False
    long = False
    opened_price = 0

    for i in range(hlen):
        if not opened:
            if Ama[i] / Bma[i] - 1 > thdAB:
                opened = True
                long = True
                opened_price = close[i]
                profit[i] = -close[i] * 0.0002 * num
            elif Ama[i] / Bma[i] - 1 < -thdAB:
                opened = True
                long = False
                opened_price = close[i]
                profit[i] = -close[i] * 0.0012 * num
        else:
            if long:
                profit[i] = (close[i] - close[i - 1]) * num
                if Ama[i] < Bma[i] or low[i] / opened_price - 1 < -stoploss:
                    opened = False
                    profit[i] = profit[i] - close[i] * num * 0.0012
            else:
                profit[i] = (close[i - 1] - close[i]) * num
                if Ama[i] > Bma[i] or high[i] / opened_price - 1 > stoploss:
                    opened = False
                    profit[i] = profit[i] - close[i] * num * 0.0002

    rets = profit / money
    equity = np.cumprod(1 + rets)
    return rets, equity
exchange = 'BIAN'
symbols = ['btcusdt']
dataf = read_data(exchange, symbols[0], '4h', "2017-01-01", "2018-10-01")
#print(dataf)
print(len(dataf))
X = np.arange(5,120)
print(X)
rets = np.array([backtest(dataf,[x,2*x,0,1],100)[0] for x in X]).T
np.random.seed(0)
PBO,POS,rd,ratios = PBOTest(rets,SampleTimes = 1000,block_len = 50)
#print(PBO,POS,rd,ratios)