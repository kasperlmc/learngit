import matplotlib.pyplot as plt
import numpy as np
import time
import os
import mpl_finance as mpf


def save_trade_fig(tuple_list, df_k, save_path, filename, n_seconds):
    """

    :param tuple_list: 元组列表，元组为（买入点的时间戳，买入时的市场价格）
    :param df_k: K线数据
    :param save_path: 保村本地的本地文件夹名
    :param filename: 保存本地时本地的文件名
    :return: 买入时的K线走势图
    """
    for i in range(len(tuple_list)):
        try:
            print("saving pic:%s" % i)
            buy_time,buy_price = tuple_list[i]
            data_k = df_k[(df_k["tickid"] >= buy_time-80*n_seconds) & (df_k["tickid"] <= buy_time+60*n_seconds)]
            candleData = np.column_stack([list(range(len(data_k))), data_k[["open", "high", "low", "close"]]])
            fig = plt.figure(figsize=(30, 12))
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
            mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
            lns1 = ax.plot(data_k["date"].values, data_k["ma20"], label="ma20")
            lns2 = ax.plot(data_k["date"].values, data_k["ma60"], label="ma60")
            lns3 = ax.plot(data_k["date"].values, data_k["ma120"], label="ma120")
            ax.annotate('buy_point', (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(buy_time)), buy_price), xytext=(0.8, 0.9), textcoords='axes fraction', arrowprops=dict(facecolor='grey', color='grey'))
            plt.grid(True)
            plt.xticks(list(range(len(data_k))), list(data_k["date"].values))
            plt.xticks(rotation=85)
            plt.tick_params(labelsize=10)
            ax2 = ax.twinx()
            lns4 = ax2.plot(data_k["date"].values, data_k["asset"], label="asset", color="#8B0000")
            lns = lns1 + lns2 + lns3 + lns4
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)
            plt.savefig("/Users/wuyong/alldata/original_data/trade_fig_save/"+save_path+"/"+filename + "_" + str(i) + ".png")
        except FileNotFoundError:
            os.mkdir("/Users/wuyong/alldata/original_data/trade_fig_save/"+save_path)




















