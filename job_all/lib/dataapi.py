# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import json
import datetime, time
import socket

import requests
import pandas as pd



pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth',100)
print(socket.gethostname())
if socket.gethostname() == 'quan-data-jp':
    common_url = 'https://ds.goupupupup.com/api/'
elif socket.gethostname() in ['ooliuyuedeMacBook-Pro.local','localhost']:
    # common_url = 'http://127.0.0.1:9000/'
    common_url = 'http://ds.goupupupup.com/api/'
else:
    common_url = 'https://ds.goupupupup.com/api/'

retry_count = 3

exchange_url = common_url + 'exchanges?datatype=%d'
exsymbol_url = common_url + 'symbols?exname=%s&datatype=%d'
exkline_url = common_url + 'kline?exname=%s&symbol=%s&period=%s&starttime=%s&endtime=%s'
huobi_kline_url = 'https://mi.goupupupup.com/klines?symbol=%s&period=%s&starttime=%s&endtime=%s'
huobi_exchange_url = 'https://mi.goupupupup.com/klines/symbols'
spread_atr_url = common_url + 'huobi/fordaily/spread/atr?exchange=%s&symbol=%s&update_day_start=%s' \
                              '&update_day_end=%s '  # 生产环境时，将127.0.0.1改为47.74.16.216
daily_volume_url = common_url + 'huobi/fordaily/daily/volume?exchange=%s&symbol=%s&update_day_start=%s' \
                                '&update_day_end=%s '  # 生产环境时，将127.0.0.1改为47.74.16.216
funding_url = common_url + 'funding/bitmex?exchange=%s&symbol=%s&timestamp=%d'
position_url = common_url + 'position/bitmex?exchange=%s&symbol=%s&period=%s&timestamp=%d'
trades_url = common_url + 'trades/getTrades?exname=%s&symbol=%s&starttime=%d'


def get_trades_data(exchange, symbol, starttime):
    """
    获取指定条件的交易信息，starttime之后的100000条信息
    :param exchange: string 大写
    :param symbol:  string 小写
    :param starttime: long unix时间戳（13位）
    :return:
    """
    # gateio。poloniex otcbtc bitmex 的逐笔交易数据，时间戳是10位的，还没有统一
    if exchange in ['GATEIO', 'POLONIEX','OTCBTC', 'BITMEX']:
        starttime = int(str(starttime)[0:10])
        print(starttime)
    url = trades_url % (exchange, symbol, starttime)
    print(url)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = (l['description'])
            if errcode != 0:
                return errcode, errmsg, None
            df = pd.DataFrame(l['data']['items'],
                              columns=['exchange', 'symbol', 'dealid', 'amount', 'dealtime', 'dir', 'price'])
            cols = list(df)
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_position_data(exchange, symbol, period, timestamp):
    """
    获取持仓量，获取指定条件且timestamp值后的10000条数据
    :param exchange: string （大写）
    :param symbol: string(小写)
    :param period: string (1h)
    :param timestamp: long (unix 10位)
    :return:
    """
    url = position_url % (exchange, symbol, period, timestamp)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = (l['description'])
            if errcode != 0:
                return errcode, errmsg, None
            df = pd.DataFrame(l['data']['items'],
                              columns=['exchange', 'period', 'symbol', 'timestamp', 'dayturnover', 'expirydate', 'fundinginterval', 'fundingrate', 'openinterest', 'predictedrate', 'totalvolume', 'createtime'])
            cols = list(df)
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_funding_data(exchange, symbol, timestamp):
    """
    获取资金费率数据
    :param exchange:  string 交易所名称
    :param symbol:  string 币对名称
    :param timestamp:  long unix时间戳（10位）获取该时间以后的1000条数据
    :return:
    """
    url = funding_url % (exchange, symbol, timestamp)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = (l['description'])
            if errcode != 0:
                return errcode, errmsg, None
            df = pd.DataFrame(l['data']['items'],
                              columns=['exchange', 'symbol', 'timestamp', 'fundingInterval', 'fundingRate', 'fundingRateDaily'])
            cols = list(df)
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_daily_volume(exchange, symbol, start_day, end_day):
    """

    :param exchange: 交易所名称 string
    :param symbol: 币对名 string
    :param start_day: 更新开始时间 string  ，YYYY-MM-dd
    :param end_day:更新结束时间 string  ，YYYY-MM-dd
    :return:
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          result:查询结果
    """
    url = daily_volume_url % (exchange, symbol, start_day, end_day)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = (l['description'])
            if errcode != 0:
                return errcode, errmsg, None
            df = pd.DataFrame(l['data']['items'],
                              columns=['Exchange', 'Symbol', 'Basecoin', 'Quotecoin', 'DailyAmount', 'DailyVolume',
                                       'UpdateDay', 'StartDay', 'EndDay'])

            # df['date'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            # cols.insert(0, cols.pop(cols.index('date')))
            # use ix to reorder
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_spread_atr(exchange, symbol, start_day, end_day):
    """

    :param exchange: 交易所名称 string
    :param symbol: 币对名 string
    :param start_day: 更新开始时间 string  ，YYYY-MM-dd
    :param end_day:更新结束时间 string  ，YYYY-MM-dd
    :return:
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          result:查询结果
    """
    url = spread_atr_url % (exchange, symbol, start_day, end_day)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = (l['description'])
            if errcode != 0:
                return errcode, errmsg, None
            df = pd.DataFrame(l['data']['items'],columns=['Exchange', 'Symbol', 'Point', 'RelativeAtr', 'BaseSpread', 'UpdateDay','StartDay', 'EndDay'])

            # df['date'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            # cols.insert(0, cols.pop(cols.index('date')))
            # use ix to reorder
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_exchange(datatype=1):
    """
        获取交易所列表
        Parameters
        ------
          datatype:int
                    数据类型，取值  1-K线
        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          List:
              币对列表
    """

    for _ in range(retry_count):
        try:
            r = requests.get(exchange_url % datatype, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None
            return errcode, errmsg, l['data']['items']
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_exsymbol(exchange, datatype=1):
    """
        获取交易所币对
        Parameters
        ------
          exchange:string
                    交易所名称
          datatype:int
                    数据类型，取值  1-K线

        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          List:
              币对列表
    """
    url = exsymbol_url % (exchange, datatype)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None
            return errcode, errmsg, l['data']['items']
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def timestamp2str(ts):
    tmp = time.localtime(ts)
    # print tmp
    return time.strftime("%Y-%m-%d %H:%M:%S", tmp)


def get_exsymbol_kline_old(exchange, symbol, period, startstr, endstr):
    """
        获取交易所币对K线
        Parameters
        ------
          exchange:string
                      交易所名称
          symbol:string
                      币对名称
          period：string
                      周期，1m,5m,15m,30m,1h,4h,1d    m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
          startstr:string
                      开始日期 format：YYYY-MM-DD
          endstr:string
                      结束日期 format：YYYY-MM-DD

        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          DataFrame:
              属性:日期 ，开盘价， 最高价， 收盘价， 最低价， 成交量
    """
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d")
        start = int(tmp.strftime("%s"))
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    url = exkline_url % (exchange, symbol, period, start, end)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['tickid', 'open', 'high', 'low', 'close', 'volume'])

            df['date'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index('date')))
            # use ix to reorder
            df = df.ix[:, cols]

            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_exsymbol_kline(exchange, symbol, period, startstr, endstr):
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d")
        start = int(time.mktime(time.strptime(str(tmp), "%Y-%m-%d %H:%M:%S")))
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d")
        end = int(time.mktime(time.strptime(str(tmp), "%Y-%m-%d %H:%M:%S")))

    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    try:
        stop_flag = 0
        data_list = []
        while 1:
            ss_time = time.time()
            url = exkline_url % (exchange, symbol, period, start, end)
            print(url) # todo 注释掉
            r = requests.get(url, timeout=10, headers={"Accept-encoding": "gzip"})
            l = r.json()
            ee_time = time.time()
            print("url time :" + str(ee_time - ss_time))  # todo test expend time
            errcode = l['result']
            errmsg = l['description']
            if not l['data']['items'] and (data_list or len(data_list) == 0):
                break

            if errcode != 0:
                return errcode, errmsg, None
            # print(l['data']['items'][-1][0])
            if stop_flag == l['data']['items'][-1][0]:
                break
            else:
                ssss_time = time.time()
                # data_dict = {**data_dict, **l['data']}
                data_list.extend(l['data']['items'])
                start = l['data']['items'][-1][0]
                stop_flag = l['data']['items'][-1][0]
                eeee_time = time.time()
                # print("list:" + str(eeee_time - ssss_time))  # todo test expend time
        sss_time = time.time()
        df = pd.DataFrame(data_list, columns=['tickid', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        df['date'] = df['tickid'].map(timestamp2str)

        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('date')))
        df = df.ix[:, cols]
        eee_time = time.time()
        # print("df:" + str(eee_time-sss_time))  # todo test expend time
        return errcode, errmsg, df
    except Exception as e:
        print(e)

    raise IOError("无法连接")


def get_huobi_ontime_kline(symbol, period, startstr, endstr):
    """
        获取交易所币对K线
        Parameters
        ------
          symbol:string
                      币对名称
          period：string
                      周期，1min,5min,15min,30min,60min,4hour,1day    m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
          startstr:int
                      开始日期 format：YYYY-MM-DD HH:MM
          endstr:string
                      结束日期 format：YYYY-MM-DD HH:MM

        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息
          DataFrame:
              属性:日期 ，开盘价， 最高价， 收盘价， 最低价， 成交量
    """
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d %H:%M")
        start = int(time.mktime(time.strptime(str(tmp), "%Y-%m-%d %H:%M:%S")))
        print(start)
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d %H:%M")
        end = int(time.mktime(time.strptime(str(tmp), "%Y-%m-%d %H:%M:%S")))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    url = huobi_kline_url % (symbol, period, start, end)
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['tickid', 'open', 'high', 'low', 'close', 'volume'])

            df['date'] = df['tickid'].map(timestamp2str)

            # get a list of columns
            cols = list(df)
            # move the column to head of list using index, pop and insert
            cols.insert(0, cols.pop(cols.index('date')))
            # use ix to reorder
            df = df.ix[:, cols]

            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_huobi_exchange():
    """
        获取交易所币对

    """

    url = huobi_exchange_url
    for _ in range(retry_count):
        try:
            r = requests.get(url, timeout=10)
            l = r.json()
            errcode = l['result']
            errmsg = l['description']
            if errcode != 0:
                return errcode, errmsg, None

            df = pd.DataFrame(l['data']['items'], columns=['symbolname'])

            return errcode, errmsg, df.symbolname.tolist()
        except Exception as e:
            print(e)

    raise IOError("无法连接")


if __name__ == '__main__':
    # errcode, errmsg, result = get_exchange()
    # print(result)

    # errcode, errmsg, result = get_exsymbol("BIAN")
    # print(result)

    #s_time = time.time()
    errcode, errmsg, df = get_exsymbol("BITFINEX")
    print(errcode,errmsg)
    print(df)

    errcode, errmsg, df = get_exsymbol_kline("BITFINEX", "qtmusdt", "1h", "2017-01-01", "2018-10-29")
    print(df.head())
    #e_time = time.time()
    #print("total:"+str(e_time-s_time))
    #
    #errcode, errmsg, result = get_huobi_exchange()
    #print(result)

    # errcode, errmsg, result = get_huobi_ontime_kline('btcusdt', '15min', '2018-10-10 00:30', '2018-10-10 00:40')
    # print(result.head(30))
    #
    # errcode, errmsg, result = get_spread_atr('BIAN', 'adausdt', '2018-08-01', '2018-09-12')
    # print(result)
    #
    # errcode, errmsg, result = get_daily_volume('BIAN', 'adausdt', '2018-08-01', '2018-09-12')
    # print(result)

    # errcode, errmsg, result = get_funding_data('BITMEX', 'xbtusd', 1463198400)
    # print(result)

    # errcode, errmsg, result = get_position_data('BITMEX', 'xbtusd', '1h', 1538049179)
    # print(result)

    # errcode, errmsg, result = get_trades_data('BITMEX', 'xbtusd', 1463198400000)
    # print(result)
