# -*- coding: UTF-8 -*-
# 接口为临时性接口，数据不保存
import gzip
import json
import time
import traceback
import urllib

import pandas as pd
import requests
from websocket import create_connection

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 100)
# 币安 restapi
urlfmt_binance = 'https://api.binance.com/api/v1/klines?symbol=%s&interval=%s&startTime=%d'
# 火币网ws请求地址
huobi_history_kline_wss = "wss://api.huobi.pro/ws"
# bitfinex restapi
urlfmt_bitfinex = 'https://api.bitfinex.com/v2/candles/trade:%s:t%s/hist?limit=1000&start=%s&sort=1'


def huobi_kline_req(fromTime, toTime, symbol_kline, period_KLine):
    trader = """{
                  "req": "market.""" + str(symbol_kline) + """.kline.""" + period_KLine + """",
                  "id": "history_kline_""" + str(symbol_kline) + """",
                  "from":""" + str(fromTime) + """,""" + """
                  "to":""" + str(toTime) + """
                }"""
    while 1:
        try:
            ws = create_connection(huobi_history_kline_wss)
            ws.send(trader)
            compressData = ws.recv()
            break
        except:
            # print('connect ws error,retry...')
            pass
    # print(trader)
    # print(trader)

    # print("websocket success connect")
    result = gzip.decompress(compressData).decode('utf-8')
    while 1:
        if result[:7] == '{"ping"':
            ts = result[8:21]
            pong = '{"pong":' + ts + '}'
            ws.send(pong)
            # print("rev ping ,again req")
            # print(trader)
            ws.send(trader)
            # time.sleep(1)
            compressData = ws.recv()
            result = gzip.decompress(compressData).decode('utf-8')
        else:
            # print(result)
            return result
    compressData = ws.recv()
    result = gzip.decompress(compressData).decode('utf-8')
    return result


def http_get_request(url, params=None, add_to_headers=None):
    """
    请求接口
    :param url:
    :param params:
    :param add_to_headers:
    :return:
    """
    headers = {
        'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0"
    }
    if params is None:
        params = {}
    if add_to_headers:
        headers.update(add_to_headers)
    postdata = urllib.parse.urlencode(params)
    response = requests.get(url, postdata, headers=headers, timeout=60)
    time.sleep(1)
    try:

        if response.status_code == 200:
            return response.text
        else:
            return
    except BaseException as e:
        # print("httpGet failed, detail is:%s,%s" % (response.text, e))
        return


def time_check(t_time, len_time=13):
    if len_time == 13:
        lt = 1000
    else:
        lt = 1
    if isinstance(t_time, str):
        for f in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y-%m-%d %H']:
            try:
                t_time_tmp = time.strptime(t_time, f)
                t_time = time.mktime(t_time_tmp) * lt
            except:
                continue
    elif isinstance(t_time, int) and len(str(t_time)) == 13:
        pass
    elif isinstance(t_time, int) and len(str(t_time)) == 10:
        t_time = t_time * lt
    else:
        raise IOError("invalid params: start_time or end_time")
    return int(t_time)


def data_clean_binance(res_tmp, period, symbole):
    """
    数据清洗 币安
    :param res_tmp:
    :param period:
    :param symbole:
    :return:
    """
    res = json.loads(res_tmp)
    list_res = []
    symbole = symbole.lower()
    for r in res:
        tickid = int(r[0]) // 1000
        open = r[1]
        high = r[2]
        low = r[3]
        close = r[4]
        volume = r[5]  # 成交量
        amount = r[7]  # 成交额
        list_res.append(['BIAN', period, symbole, tickid, open, high, low, close, volume, amount])
    return list_res


def data_clean_huobi(data, period, symbole):
    """
    数据清洗 火币
    :param data:
    :param period:
    :param symbole:
    :return:
    """
    res = json.loads(data)
    list_res = []
    symbole = symbole.lower()
    for di in res['data']:
        tickid = di['id']
        open = di['open']
        close = di['close']
        low = di['low']
        high = di['high']
        amount = di['vol']  # 成交额
        volume = di['amount']  # 成交量
        list_res.append(['HUOBI', period, symbole, tickid, open, high, low, close, volume, amount])
    return list_res


def data_clean_bitfinex(data, period, symbole):
    res = json.loads(data)
    list_res = []
    symbole = symbole.lower()
    for di in res:
        tickid = di[0] // 1000
        open = di[1]
        close = di[2]
        high = di[3]
        low = di[4]
        volume = di[5]  # 成交量
        amount = None   # 成交额
        list_res.append(['BITFINEX', period, symbole, tickid, open, high, low, close, volume, amount])
    return list_res


def data_tran(l, rule):
    """
    火币、bitfinex 4h数据合成
    :param l:
    :param rule:
    :return:
    """
    f_data = l[0][3]
    l_data = l[-1][3]
    f_data_hour = time.localtime(f_data).tm_hour
    l_data_hour = time.localtime(l_data).tm_hour
    df = pd.DataFrame(l, columns=['exchange', 'period', 'symbol', 'tickid', 'open', 'high', 'low', 'close', 'volume',
                                  'amount'])
    df['date'] = df['tickid'].map(timestamp2str)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('date')))
    df = df.ix[:, cols]
    # date列变成datetime格式并用作索引
    df['date'] = pd.to_datetime(df['date'])  # todo 将 timestamp去掉
    df.set_index("date", inplace=True)
    df = df.resample(rule).apply({
                                  'exchange': 'first',
                                  'period': 'first',
                                  'symbol': 'first',
                                  'tickid': 'first',
                                  'open': 'first',
                                  'high': 'max',
                                  'low': 'min',
                                  'close': 'last',
                                  'volume': 'sum',
                                  'amount': 'sum'})
    df = df.reset_index()
    df['period'] = '4h'
    # print(len(df))
    # print(df)
    count = len(df)
    # print("第一条数据小时数：" + str(f_data_hour))
    if f_data_hour % 4 != 0:
        df.drop(labels=[0], inplace=True)
    # print("最后一条数据小时数：" + str(l_data_hour))
    if l_data_hour % 4 != 3:
        df.drop(labels=[count - 1], inplace=True)
    del df['date']
    # print(df)
    train_data = pd.np.array(df)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    return train_x_list


def timestamp2str(ts):
    tmp = time.localtime(ts)
    # print tmp
    return time.strftime("%Y-%m-%d %H:%M:%S", tmp)


def realtime_kline_BIAN(periods, symboles, start_time, end_time=None):
    """
    获取实时Kline： BINANCE交易所
    :param periods: list或str ['4h','1d']或'4h'
    :param symboles: list或str： ['EOSBTC', 'BNBBTC']或'BNBBTC'
    :param start_time: 时间戳或者时间字符串：1543593600000  1543593600  '2018-12-01 00:00:00
    :param end_time: 时间戳或者时间字符串或者不写
    :return:list : [[exchange, period, symbole, tickid, open, high, low, close, volume, amount]]
    """
    try:
        if isinstance(periods, list):
            pass
        elif isinstance(periods, str):
            periods = [periods]
        else:
            raise IOError("invalid params: periods")

        if isinstance(symboles, list):
            pass
        elif isinstance(symboles, str):
            symboles = [symboles]
        else:
            raise IOError("invalid params: symboles")

        start_time = time_check(start_time)
        lists_datas = []
        s_time_cp = start_time
        if end_time is None:
            end_time = int(time.time()) * 1000
        else:
            end_time = time_check(end_time)
        for period in periods:
            if period[-1] == 'm':
                internal_time = int(period[0:-1]) * 60 * 1000
            elif period[-1] == 'h':
                internal_time = int(period[0:-1]) * 60 * 60 * 1000
            elif period[-1] == 'd':
                internal_time = int(period[0:-1]) * 60 * 60 * 24 * 1000
            else:
                raise IOError("unknow period")
            for symbole in symboles:
                while True:
                    # print(">>>>>>>>>period=%s,symbol=%s" % (str(period), str(symbole)))
                    url = urlfmt_binance % (str(symbole), str(period), start_time)
                    # print(">>>>>>>>>url=%s", str(url))
                    # print(int(start_time) + internal_time)
                    # print(end_time)
                    if start_time + internal_time >= end_time:
                        start_time = s_time_cp
                        # print("break while")
                        break
                    time.sleep(1)
                    res_tmp = http_get_request(url)
                    list_data = data_clean_binance(res_tmp, period, symbole)
                    lists_datas.extend(list_data)
                    # print(list_data)
                    start_time = int(list_data[-1][3]) * 1000
        return lists_datas
    except Exception as e:
        # print(traceback.format_exc())
        raise IOError('internal system error')


def realtime_kline_HUOBI(periods, symboles, start_time, end_time=None):
    try:
        if isinstance(periods, list):
            pass
        elif isinstance(periods, str):
            periods = [periods]
        else:
            raise IOError("invalid params: periods")

        if isinstance(symboles, list):
            pass
        elif isinstance(symboles, str):
            symboles = [symboles]
        else:
            raise IOError("invalid params: symboles")
        start_time = time_check(start_time, 10)
        start_time_cp = start_time
        lists_datas = []

        for period in periods:
            p_period = period
            tran_flag = False
            if period in ['1m', '5m', '15m', '30m']:
                period_time = 18000 * int(period[0:-1])
                period_req = period + "in"
            elif period in ['1h']:
                period_time = 18000 * 60
                period_req = '60min'
            elif period in ['4h']:
                period_time = 18000 * 60
                period_req = '60min'
                period = '1h'
                tran_flag = True
            elif period in ['1d']:
                period_time = 18000 * 60 * 24
                period_req = '1day'
            else:
                raise IOError("unknow period")
            for symbole in symboles:
                lists_datas_cp = []
                symbole = symbole.lower()
                # print(">>>>>>>>>period=%s,symbol=%s" % (str(p_period), str(symbole)))
                while 1:
                    stop_flag = False
                    to_time = start_time + period_time
                    if start_time > time.time():
                        # print(
                        #     ">>>>>>>>>>>>>start time greate then currency time,break;start_time=%s,currency_time=%s" % (
                        #         start_time, time.time()))
                        start_time = start_time_cp
                        break
                    if to_time > time.time():
                        stop_flag = True
                        to_time = int(time.time())
                        # print(">>>>>>>>>>>>>define to time eq currency time,to_time=%s, currency_time=%s" % (
                        #     to_time, time.time()))
                    res = huobi_kline_req(start_time, to_time, symbole, period_req)
                    res_list_cp = res_list = data_clean_huobi(res, period, symbole)
                    if tran_flag:
                        lists_datas_cp.extend(res_list_cp)
                    else:
                        lists_datas.extend(res_list)
                    start_time = to_time
                    if stop_flag:
                        # print("get all data and break")
                        start_time = start_time_cp
                        break
                if tran_flag:
                    list_tran_datas = data_tran(lists_datas_cp, '4h')
                    lists_datas.extend(list_tran_datas)
        return lists_datas
    except Exception as e:
        raise IOError('>>>>>>>>>>internal system error')


def realtime_kline_BITFINEX(periods, symboles, start_time, end_time=None):
    try:
        if isinstance(periods, list):
            pass
        elif isinstance(periods, str):
            periods = [periods]
        else:
            raise IOError("invalid params: periods")

        if isinstance(symboles, list):
            pass
        elif isinstance(symboles, str):
            symboles = [symboles]
        else:
            raise IOError("invalid params: symboles")

        start_time = time_check(start_time)
        lists_datas = []
        s_time_cp = start_time
        if end_time is None:
            end_time = int(time.time()) * 1000
        else:
            end_time = time_check(end_time)
        for period in periods:
            p_period = period
            tran_flag = False
            if period[-1] == 'm':
                internal_time = int(period[0:-1]) * 60 * 1000
            elif period[-1] == 'h':
                if period == '4h':
                    internal_time = int(1 * 60 * 60 * 1000)
                    period = '1h'
                    tran_flag = True
                else:
                    internal_time = int(period[0:-1]) * 60 * 60 * 1000
            elif period[-1] == 'd':
                internal_time = int(period[0:-1]) * 60 * 60 * 24 * 1000
            else:
                raise IOError("unknow period")
            for symbole in symboles:
                lists_datas_cp = []
                p_symbole = symbole.lower()
                while True:
                    # print(">>>>>>>>>period=%s,symbol=%s" % (str(p_period), str(p_symbole)))
                    url = urlfmt_bitfinex % (str(period), str(symbole), start_time)
                    # print(">>>>>>>>>url=%s", str(url))
                    # print(int(start_time) + internal_time)
                    # print(end_time)
                    if start_time + internal_time >= end_time:
                        start_time = s_time_cp
                        # print("break while")
                        break
                    time.sleep(1)
                    res_tmp = http_get_request(url)
                    res_list_cp = list_data = data_clean_bitfinex(res_tmp, period, symbole)
                    if tran_flag:
                        lists_datas_cp.extend(res_list_cp)
                    else:
                        lists_datas.extend(list_data)
                    # print(list_data)
                    start_time = int(list_data[-1][3]) * 1000
                if tran_flag:
                    list_tran_datas = data_tran(lists_datas_cp, '4h')
                    lists_datas.extend(list_tran_datas)
        return lists_datas
    except Exception as e:
        print(traceback.format_exc())
        raise IOError('internal system error')


def get_realtime_data(exchange, periods, symboles, start_time, end_time=None):
    """
    获取实时Kline： 交易所
    :param exchange: string：BIAN,HUOBI,BITFINEX
    :param periods: list或str ['4h','1d']或'4h'
    :param symboles: list或str： ['EOSBTC', 'BNBBTC']或'BNBBTC'
    :param start_time: 时间戳或者时间字符串：1543593600000  1543593600  '2018-12-01 00:00:00
    :param end_time: 时间戳或者时间字符串或者不写
    :return:list : [[exchange, period, symbole, tickid, open, high, low, close, volume, amount]]
    """
    if not exchange or not periods or not symboles or not start_time:
        raise IOError('parameters cannot be empty ')
    if exchange == 'BIAN':
        return realtime_kline_BIAN(periods, symboles, start_time)
    elif exchange == 'HUOBI':
        return realtime_kline_HUOBI(periods, symboles, start_time)
    elif exchange == 'BITFINEX':
        return realtime_kline_BITFINEX(periods, symboles, start_time)
    else:
        raise IOError('not support exchange: %s' % exchange)


if __name__ == '__main__':
    # 调用实例
    # r = realtime_kline_BIAN(['4h', '1d'], ['EOSBTC', 'BNBBTC'], 1543593600000)
    # r = realtime_kline_BIAN('4h', 'BNBBTC', 1543593600)
    # r = realtime_kline_BIAN('4h', 'BNBBTC', '2018-12-01 00:00:00')

    r = get_realtime_data('BIAN', '4h', 'ADABTC', '2018-01-23 00:00:00', end_time=None)
    # r = get_realtime_data('HUOBI', ['4h'], ['ethhusd', 'btcusdt'], '2018-12-01 00:00:00')
    # r = get_realtime_data('BITFINEX', ['4h'], ['BTCUSD', 'LTCUSD'], '2018-12-01 00:00:00')
    # print('333')
    cols = ["exchange", "period", "symbol", "tickid", "open", "high", "low", "close", "volume", "amount"]
    df = pd.DataFrame(r,columns=cols)
    # print(len(r))
    # print(df)
    # symbols=["btcusdt", "ethusdt", "xrpusdt", "trxusdt", "eosusdt", "zecusdt00", "ltcusdt",
    #          "etcusdt", "bchusdt", "iotausdt", "adausdt", "xmrusdt", "dashusdt", "htusdt",
    #          "omgusdt", "wavesusdt", "nanousdt", "btmusdt", "elausdt", "ontusdt", "iostusdt",
    #          "qtumusdt","dtausdt", "zilusdt", "elfusdt", "gntusdt"]

    # symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
    #            "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
    #            "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
    #            "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc"]

    # symbols = [x.upper() for x in symbols]
    # print(symbols)
    #
    # for i in range(len(symbols)):
    #     try:
    #         values_list = get_realtime_data('BIAN', '4h', symbols[i], start_time='2018-06-01 00:00:00', end_time=None)
    #         print(len(values_list))
    #         df = pd.DataFrame(values_list, columns=cols)
    #         print(symbols[i])
    #         print(df.tail())
    #         df.to_csv("/Users/wuyong/alldata/original_data/BIAN_"+symbols[i]+"_4h_2018-06-01_2018-12-27.csv")
    #     except OSError:
    #         print(symbols[i])














