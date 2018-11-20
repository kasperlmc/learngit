# coding=utf-8

import sys

sys.path.append('..')
from backtest import *
import matplotlib.pyplot as plt
from lib import dataapi
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def read_data(exchange, symbol, period, start_day, end_day):
    fname = '%s_%s_%s_%s_%s' % (exchange, symbol, period, start_day, end_day)
    fname = fname.replace('/', '_')
    fpath = mkfpath('.\\api_data', fname + '.csv')
    print('reading and writing %s...' % fname)

    # 如果没有文件
    if not os.path.exists(fpath):
        ohlc = dataapi.get_exsymbol_kline(exchange, symbol, period, start_day, end_day)[2]
        ohlc.reset_index(inplace=True)
        # 节省空间写入数据
        if not ohlc.empty:
            ohlc.to_csv(fpath)
    else:
        ohlc = pd.read_csv(fpath, index_col=[0])
    ohlc['date_time'] = pd.to_datetime(ohlc['date'])

    # 观察数据缺失情况
    count = ohlc['date_time'].diff().value_counts()
    if len(count) > 1:
        print('Warning: discontinuous data')
        print(count.head(), '\n')

    return ohlc


def calc_alpha_signal(df, param, start_day, end_day, corr):
    if corr > 0:
        df["date_time"] = pd.to_datetime(df["date"])
        factor_df = df[(df['date_time'] >= start_day) & (df['date_time'] < end_day)]
        factor = factor_df[param]
        bin_alpha = pd.qcut(factor, q=10, retbins=True,duplicates="drop")[1]
        df["open_long_signal"] = (df[param].shift(1) >= bin_alpha[-2]) * 1
        df['close_long_signal'] = (df[param].shift(1) < bin_alpha[-2]) * 1
        df["open_short_signal"] = (df[param].shift(1) <= bin_alpha[1]) * 1
        df["close_short_signal"] = (df[param].shift(1) > bin_alpha[1]) * 1
        df['buy_price'] = df['open']
        df['sell_price'] = df['open']
    else:
        df["date_time"] = pd.to_datetime(df["date"])
        factor_df = df[(df['date_time'] >= start_day) & (df['date_time'] < end_day)]
        factor = factor_df[param]
        bin_alpha = pd.qcut(factor, q=10, retbins=True,duplicates="drop")[1]
        df["open_long_signal"] = (df[param].shift(1) <= bin_alpha[1]) * 1
        df['close_long_signal'] = (df[param].shift(1) > bin_alpha[1]) * 1
        df["open_short_signal"] = (df[param].shift(1) >= bin_alpha[-2]) * 1
        df["close_short_signal"] = (df[param].shift(1) < bin_alpha[-2]) * 1
        df['buy_price'] = df['open']
        df['sell_price'] = df['open']
    return df


def main_strategy(df, param, ini_capital):
    # initial capital: [tradecoin, basecoin]
    # 净值以basecoin计算
    capital = ini_capital
    fee = 0.00075
    slippage = 0.00025

    tradecoin_amt = []
    basecoin_amt = []
    signal = []
    stop_loss_buy=0
    stop_loss_sell=0

    for rid, row in df.iterrows():

        pos_tmp = (capital[0] * row.open) / (capital[0] * row.open + capital[1])
        # 空仓时，关注做多做空信号
        if (-0.2 < pos_tmp < 0.2):
            if row.open_long_signal and not row.close_long_signal:
                # 全仓买入 tradecoin
                b_price = row.buy_price * (1 + slippage)
                stop_loss_buy=row.buy_price*0.95
                capital, delta = order_pct_to(1, capital, b_price, fee)
                # print("全仓买入", row["date"],capital)
                signal.append([row.date_time, 'b0', b_price / (1 - fee), 1])
            elif row.open_short_signal and not row.close_short_signal:
                # 1倍杠杆做空 tradecoin
                s_price = row.sell_price * (1 - slippage)
                stop_loss_sell=row.sell_price*1.05
                capital, delta = order_pct_to(-1, capital, s_price, fee)
                # print("杠杆做空", row["date"], capital)
                signal.append([row.date_time, 's1', s_price * (1 - fee), -1])
        # 满仓时，关注平仓信号
        elif pos_tmp >= 0.2:
            if row.low<stop_loss_buy:
                s_price=stop_loss_buy*(1-slippage)
                capital, delta = order_pct_to(0, capital, s_price, fee)
                signal.append([row.date_time, 's0', s_price * (1 - fee), 0])
            elif row.close_long_signal:
                s_price = row.sell_price * (1 - slippage)
                capital, delta = order_pct_to(0, capital, s_price, fee)
            # print("平多仓", row["date"],capital)
                signal.append([row.date_time, 's0', s_price * (1 - fee), 0])
        elif pos_tmp <= -0.2:
            if row.high>stop_loss_sell:
                b_price=stop_loss_sell*(1+slippage)
                capital, delta = order_pct_to(0, capital, b_price, fee)
                # print("平空仓", row["date"],capital)
                signal.append([row.date_time, 'b1', b_price / (1 - fee), 0])
            elif row.close_short_signal:
                b_price = row.buy_price * (1 + slippage)
                capital, delta = order_pct_to(0, capital, b_price, fee)
                # print("平空仓", row["date"],capital)
                signal.append([row.date_time, 'b1', b_price / (1 - fee), 0])

        # 记录持币数量
        tradecoin_amt = np.append(tradecoin_amt, capital[0])
        basecoin_amt = np.append(basecoin_amt, capital[1])

    # 计算净值
    tradecoin_net = tradecoin_amt + basecoin_amt / df['close']
    basecoin_net = tradecoin_amt * df['close'] + basecoin_amt

    pos = tradecoin_amt / tradecoin_net

    end_capital = capital
    net_df = pd.DataFrame({'date_time': df['date_time'],
                           'close': df['close'],
                           'net': basecoin_net,
                           'pos': pos,
                           'tradecoin': tradecoin_amt,
                           'basecoin': basecoin_amt})

    signal_df = pd.DataFrame(signal, columns=['date_time', 'signal', 'price', 'pos'])

    df = df.merge(signal_df, how='outer', on='date_time')
    df = df.merge(net_df, how='outer', on='date_time')
    return net_df, signal_df, end_capital


def long_short_ret(net_df):
    net_df = net_df.copy()
    net_df['ret'] = net_df['net'] / net_df['net'].shift(1)
    long_ret = np.prod(net_df[net_df['pos'] > 0]['ret']) - 1
    short_ret = np.prod(net_df[net_df['pos'] < 0]['ret']) - 1
    return long_ret, short_ret


def month_profit(net_df):
    net_df = net_df.set_index('date_time')
    ret = []
    for gid, group in net_df.groupby(pd.Grouper(freq='M')):
        gid = group.index[0]
        long_ret, short_ret = long_short_ret(group)
        ret.append([gid, long_ret, short_ret])

    month_ret = pd.DataFrame(ret, columns=['date_time', 'long_return', 'short_return'])
    return month_ret


def summary_net(net_df, plot_in_loops,alphas):
    month_ret = month_profit(net_df)
    # 转换成日净值
    net_df.set_index('date_time', inplace=True)
    net_df = net_df.resample('1D').asfreq()
    net_df.reset_index(inplace=True)
    net_df.dropna(inplace=True)

    # 计算汇总
    net = net_df['net']
    date_time = net_df['date_time']
    base = net_df['close']
    tot_ret = total_ret(net)
    ann_ret = annual_ret(date_time, net)
    sharpe = sharpe_ratio(net)
    annualVolatility = AnnualVolatility(net)
    drawdown = max_drawdown(net.values)
    alpha, beta = alpha_beta(date_time, base, net)
    ir = infomation_ratio(date_time, base, net)
    ret_r = ann_ret / drawdown

    result = [tot_ret, ann_ret, sharpe, annualVolatility,
              drawdown, alpha, beta, ret_r, ir,
              net_df['date_time'].iloc[0], net_df['date_time'].iloc[-1]]
    cols = ['tot_ret', 'ann_ret', 'sharpe', 'annualVolatility', 'max_drawdown', 'alpha', 'beta', 'ret_ratio', 'ir', 'start_time', 'end_time']

    if plot_in_loops:
        param_str="multiple"+alphas

        net_df['close'] = net_df['close'] / net_df['close'].iloc[0]
        net_df['net'] = net_df['net'] / net_df['net'].iloc[0]

        fpath = mkfpath('api_figure_1', param_str + '.png')

        fig, ax = plt.subplots(2)
        net_df.plot(x='date_time', y=['close', 'net'],
                    title=param_str, grid=True, ax=ax[0])
        ax[0].set_xlabel('')
        month_ret['month'] = month_ret['date_time'].dt.strftime('%Y-%m')
        month_ret.plot(kind='bar', x='month',
                       y=['long_return', 'short_return'],
                       color=['r', 'b'], grid=True, ax=ax[1])
        plt.tight_layout()
        plt.savefig(fpath)

    return result, cols

def summary_signal(signal_df):
    win_r, profit_r = win_profit_ratio(signal_df)
    result=[win_r,profit_r]
    cols=['win_r', 'profit_r']
    return result,cols

def do_backtest(df, param, start_day, end_day):
    ini_capital = [0, 1]
    print('backtesting %s %s %s...' % (param, start_day, end_day))

    # 回测时间段
    df["date_time"] = pd.to_datetime(df["date"])
    df = df[(df['date_time'] >= start_day) & (df['date_time'] < end_day)]

    net_df, signal_df, end_capital = main_strategy(df, param, ini_capital)
    return net_df, signal_df, end_capital