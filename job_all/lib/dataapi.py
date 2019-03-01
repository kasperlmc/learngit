# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import json
import datetime, time
import socket

import requests
import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 100)
print(socket.gethostname())
if socket.gethostname() == 'quan-data-jp':
    common_url = 'https://ds.goupupupup.com/api/'
elif socket.gethostname() in ['ooliuyuedeMacBook-Pro.local', 'localhost']:
    # common_url = 'http://127.0.0.1:9000/'
    common_url = 'http://ds.goupupupup.com/api/'
else:
    common_url = 'https://ds.goupupupup.com/api/'

retry_count = 3

exchange_url = common_url + 'exchanges?datatype=%d'
exsymbol_url = common_url + 'symbols?exname=%s&datatype=%d'
exkline_url = common_url + 'kline?exname=%s&symbol=%s&period=%s&starttime=%s&endtime=%s'
exfuture_kline_url = common_url + 'future/kline?exname=%s&symbol=%s&period=%s&starttime=%s&endtime=%s&datatype=%d&symboltype=%d'

hb_debug = 0
if hb_debug == 1:
    migo = "http://127.0.0.1:8000"
else:
    migo = 'https://mi.goupupupup.com'
huobi_kline_url = migo + '/data/klines?symbol=%s&period=%s&starttime=%s&endtime=%s&exname=%s'
huobi_exchange_url = migo + '/data/symbols'

spread_atr_url = common_url + 'huobi/fordaily/spread/atr?exchange=%s&symbol=%s&update_day_start=%s' \
                              '&update_day_end=%s '  # 生产环境时，将127.0.0.1改为47.74.16.216
daily_volume_url = common_url + 'huobi/fordaily/daily/volume?exchange=%s&symbol=%s&update_day_start=%s' \
                                '&update_day_end=%s '  # 生产环境时，将127.0.0.1改为47.74.16.216
funding_url = common_url + 'funding/bitmex?exchange=%s&symbol=%s&timestamp=%d'
position_url = common_url + 'position/bitmex?exchange=%s&symbol=%s&period=%s&timestamp=%d'
trades_url = common_url + 'trades/getTrades?exname=%s&symbol=%s&starttime=%d'


def get_addrs_tx_history_data(symbol, deleflag, address=None, hash=None, role=None, timeStr=None):
    """
    获取链上地址转账历史数据
    :param timeStr: string yyyy-mm-dd(也可精确到时分秒) 以后的100000条数据
    :param symbol: string
    :param deleflag: int  0 或1
    :param address: string
    :param hash: string
    :param role: int 0：发送方；1 接收方
    :return:
    """
    if timeStr is not None:
        timeStamp = str_time_to_timestamp(timeStr, 10)
    else:
        timeStamp = None
    # http: // localhost: 9000 / chaindata / history / getAddsHistoryData?deleflag = 0 & symbol = btc & role = 1 & timeStamp = 1540400230
    url = common_url + '/chaindata/history/getAddsHistoryData?symbol=%s&deleflag=%d' % (symbol.lower(), deleflag)
    if address is not None:
        url = url + '&address=' + address
    if hash is not None:
        url = url + '&hash=' + hash
    if role is not None:
        url = url + '&role=' + str(role)
    if timeStamp is not None:
        url = url + '&timeStamp=' + str(timeStamp)
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
                              columns=['hash', 'blockNumber', 'address', 'timeStamp', 'role', 'symbol', 'from', 'to',
                                       'value', 'isError', 'contractAddress', 'blockHash', 'source'])
            df['date'] = df['timeStamp'].map(timestamp2str)
            cols = list(df)
            cols.insert(0, cols.pop(cols.index('date')))
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_trades_data(exchange, symbol, starttime):
    """
    获取指定条件的交易信息，starttime之后的100000条信息(降序排列)
    :param exchange: string 大写
    :param symbol:  string 小写
    :param starttime: string '2018-08-02 16:26'
    :return:
    """
    timestamp = str_time_to_timestamp(starttime, 13)
    url = trades_url % (exchange, symbol, timestamp)
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


def get_position_data(exchange, symbol, period, str):
    """
    获取持仓量，获取指定条件且string值后的10000条数据
    :param exchange: string （大写）
    :param symbol: string(小写)
    :param period: string (1h)
    :param str: string '2018-08-02 16:26'
    :return:
    """
    timestamp = str_time_to_timestamp(str, 10)
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
                              columns=['exchange', 'period', 'symbol', 'timestamp', 'dayturnover', 'expirydate',
                                       'fundinginterval', 'fundingrate', 'openinterest', 'predictedrate', 'totalvolume',
                                       'createtime'])
            cols = list(df)
            df = df.ix[:, cols]
            return errcode, errmsg, df
        except Exception as e:
            print(e)

    raise IOError("无法连接")


def get_funding_data(exchange, symbol, str):
    """
    获取资金费率数据
    :param exchange:  string 交易所名称
    :param symbol:  string 币对名称
    :param str:  string，'2018-08-02 16:26' 获取该时间以后的1000条数据
    :return:
    """
    timestamp = str_time_to_timestamp(str, 10)
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
                              columns=['exchange', 'symbol', 'timestamp', 'fundingInterval', 'fundingRate',
                                       'fundingRateDaily'])
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
            df = pd.DataFrame(l['data']['items'],
                              columns=['Exchange', 'Symbol', 'Point', 'RelativeAtr', 'BaseSpread', 'UpdateDay',
                                       'StartDay', 'EndDay'])

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
                    数据类型，取值  1-K线, 3-合约k线
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
                    数据类型，取值  1-K线；3-合约k线

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
    """
    获取现货历史kline线
    :param exchange: string 交易所名称
    :param symbol: string 币对名称
    :param period: string 时间级别
    :param startstr: string 起始时间
    :param endstr: string 结束时间
    :return:
    """
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d")
        start = int(tmp.strftime("%s"))
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    try:
        stop_flag = 0
        data_list = []
        while 1:
            ss_time = time.time()
            url = exkline_url % (exchange, symbol, period, start, end)
            print(url)  # 注释掉
            r = requests.get(url, timeout=10, headers={"Accept-encoding": "gzip"})
            l = r.json()
            ee_time = time.time()
            print("url time :" + str(ee_time - ss_time))  # test expend time
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
                # print("list:" + str(eeee_time - ssss_time))  # test expend time
        sss_time = time.time()
        df = pd.DataFrame(data_list, columns=['tickid', 'open', 'high', 'low', 'close', 'volume', 'amount'])

        df['date'] = df['tickid'].map(timestamp2str)

        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('date')))
        df = df.ix[:, cols]
        eee_time = time.time()
        # print("df:" + str(eee_time-sss_time))  # test expend time
        return errcode, errmsg, df
    except Exception as e:
        print(e)

    raise IOError("无法连接")


def get_exsymbol_future_kline(exchange, symbol, period, startstr, endstr, datatype=0, symboltype=0):
    """
    获取合约Kline线
    :param symboltype: 查询用，例：0对应xrp，将取出所有符合条件的xrp数据(数据已拼接)；1对应xrph19，只取出该币对的数据
    :param exchange: string 交易所名称
    :param symbol: string 币对名称
    :param period: string 时间级别
    :param startstr: string 起始时间
    :param endstr: string 结束时间
    :param datatype: 默认0：不区分合约类型：获取当前币对所有；1：当周；2：次周；3：季度；4：永续，5：半年
    :return:
    """
    try:
        tmp = datetime.datetime.strptime(startstr, "%Y-%m-%d")
        start = int(tmp.strftime("%s"))
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    try:
        stop_flag = 0
        data_list = []
        while 1:
            ss_time = time.time()
            url = exfuture_kline_url % (exchange, symbol, period, start, end, datatype, symboltype)
            print(url)  # 注释掉
            r = requests.get(url, timeout=10, headers={"Accept-encoding": "gzip"})
            l = r.json()
            ee_time = time.time()
            print("url time :" + str(ee_time - ss_time))  # test expend time
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
                # print("list:" + str(eeee_time - ssss_time))  # test expend time
        sss_time = time.time()
        df = pd.DataFrame(data_list, columns=['tickid', 'open', 'high', 'low', 'close', 'volume', 'amount', 'type'])

        df['date'] = df['tickid'].map(timestamp2str)

        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('date')))
        df = df.ix[:, cols]
        eee_time = time.time()
        # print("df:" + str(eee_time-sss_time))  # test expend time
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
        start = int(tmp.strftime("%s"))
        print(tmp)
        print(start)
        tmp = datetime.datetime.strptime(endstr, "%Y-%m-%d %H:%M")
        end = int(tmp.strftime("%s"))
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')

    url = huobi_kline_url % (symbol, period, start, end, 'HUOBI')
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


def str_time_to_timestamp(str, len):
    """
    字符日期转换为10位或13位时间戳
    :param str: string '2018-08-18 00:30'
    :param len: int 10或13
    :return:
    """
    try:
        for time_str in ['%Y-%m-%d %H:%M:%S', "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d"]:
            try:
                tmp = datetime.datetime.strptime(str, time_str)
            except:
                continue
        if len == 13:
            timestamp = int(tmp.strftime("%s")) * 1000
        elif len == 10:
            timestamp = int(tmp.strftime("%s"))
        else:
            raise TypeError('invalid params.')
        print(tmp)
        print(timestamp)
        return timestamp
    except Exception as e:
        print(e)
        raise TypeError('ktype input error.')


if __name__ == '__main__':
    # errcode, errmsg, result = get_exchange(datatype=3)
    # print(result)
    #
    # errcode, errmsg, result = get_exsymbol("BITMEX", datatype=3)
    # print(result)

    # s_time = time.time()
    # errcode, errmsg, df = get_exsymbol_kline("BIAN", "ethbtc", "1h", "2019-01-04", "2019-12-19")
    errcode, errmsg, df = get_exsymbol_kline("BIAN", "ethbtc", "1m", "2019-02-01", "2019-02-02")
    # # errcode, errmsg, df = get_exsymbol_kline("BIAN", "neobtc", "1d", "2015-05-15", "2018-10-29")
    print(str(len(df)))
    print(df.head())
    print(df.tail())
    print(type(df["date"].values[0]))
    print(df[(df["date"] > "2019-02-01 00:03:00") & (df["date"] < "2019-02-01 01:03:00")])
    # print(str(len(df)), df[0:21350])
    # e_time = time.time()
    # print("total:"+str(e_time-s_time))
    # print(len(df))
    # print(type(df))
    # print(df["date"].values[0], type(df["date"].values[0]))
    # df["date"] = [datetime.datetime.strptime(pub_time, "%Y-%m-%d %H:%M:%S") for pub_time in df["date"].values]
    # time_delta = datetime.timedelta(hours=8)
    # print(type(time_delta))
    # print(df.head())
    # print(df.tail())
    # df["date"] = df["date"]-time_delta
    # print(df.head())
    # print(df.tail())
    # df.to_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_1h_all.csv")

    # s_time = time.time()
    # errcode, errmsg, df = get_exsymbol_future_kline("BITMEX", "xrpu18", "1m", "2017-01-01", "2019-02-26", 0, 1)
    # errcode, errmsg, df = get_exsymbol_future_kline("OKEX", "bch-usd-190118", "1m", "2018-01-20", "2019-02-26", 0)
    # df.to_csv('xrpu18.csv')
    # e_time = time.time()
    # print("total:" + str(e_time - s_time))

    # errcode, errmsg, result = get_huobi_exchange()
    # print(result)

    # errcode, errmsg, result = get_huobi_ontime_kline('btcusdt', '4hour', '2018-12-10 00:30', '2018-12-25 00:30')
    # print(result[0:10])
    #
    # errcode, errmsg, result = get_spread_atr('BIAN', 'adausdt', '2018-08-01', '2018-09-12')
    # print(result)
    #
    # errcode, errmsg, result = get_daily_volume('BIAN', 'adausdt', '2018-08-01', '2018-09-12')
    # print(result)

    # errcode, errmsg, result = get_funding_data('BITMEX', 'xbtusd', '2018-08-02 16:26')
    # print(result)

    # errcode, errmsg, result = get_position_data('BITMEX', 'xbtusd', '1h', '2018-08-02 16:26')
    # print(result)

    # errcode, errmsg, result = get_trades_data('BIAN', 'btcusdt', '2018-12-01 01:00')
    # # errcode, errmsg, result = get_trades_data('POLONIEX', 'usdt_btc', '2018-09-02 16:26')
    # print(result)

    # print(str_time_to_timestamp('2018-08-18 00:30', 13))

    # errcode, errmsg, result = get_addrs_tx_history_data('btc', 0, timeStr='2017-01-01')
    # print(result)

    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(1545646500)))
    #
    # print(pd.to_datetime(1545646500,unit='s') + pd.Timedelta(hours=8))
