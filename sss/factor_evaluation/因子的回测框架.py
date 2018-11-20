import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

factor_df = pd.read_csv('../factor_writedb/btcusdt_Alpha.alpha002.csv', index_col=0)
factor_df_1=pd.read_csv('../factor_writedb/btcusdt_Alpha.alpha040.csv', index_col=0)
#print("y")

factor_df["add_factor"]=factor_df_1["Alpha.alpha040_rank_5"]
#print(factor_df)


'''
#print(factor_df.tail(50))
factor_df=factor_df.dropna()
factor_0=factor_df["Alpha.alpha002_rank_20"][2:]
factor_df_1=factor_df_1.dropna()
factor_1=factor_df_1['Alpha.alpha040_rank_5']
#print(len(factor_0),len(factor_1))
factor=pd.Series(factor_0.values+factor_1.values,name="factor")
print(factor)
'''


factor_df=factor_df.dropna()
factor=factor_df[["add_factor","Alpha.alpha002_rank_20"]]




#print(factor)
print(pd.qcut(factor["add_factor"],q=10,retbins=True)[1])

#'''
print(factor_df.ix[2441])
index_list=factor.index

cash_value=1000000
value_list=[1000000]
position_status=0
hold_num=0
bin_alpha=pd.qcut(factor,q=10,retbins=True)[1]
for x in index_list[1:]:
    data_today=factor_df.ix[x]
    data_yesterday=factor_df.ix[x-1]
    alpha_yesterday=data_yesterday["sum_factor"]
    if alpha_yesterday<bin_alpha[1]:
        if position_status==0:
            hold_num+=cash_value*0.5/data_today["open"]
            cash_value=cash_value*0.5
            position_status += 1
            print("B",data_today["date"],position_status,hold_num,data_today["open"])
            print(cash_value+hold_num*position_status*data_today["open"])
        else:
            pass
    elif alpha_yesterday>=bin_alpha[-2]:
        if position_status==0:
            hold_num=cash_value*0.5/data_today["open"]
            cash_value=cash_value+cash_value*0.5
            position_status = position_status - 1
            print("S",data_today["date"],position_status,hold_num,data_today["open"])
            print(cash_value+hold_num*position_status*data_today["open"])
        else:
            pass
    else:
        if position_status==0:
            pass
        elif position_status==1:
            cash_value=cash_value+hold_num*data_today["open"]
            position_status=position_status-1
            hold_num=0
            print("PL",data_today["date"],position_status,data_today["open"])
            print(cash_value+hold_num*position_status*data_today["open"])
        else:
            cash_value=cash_value-hold_num*data_today["open"]
            position_status=position_status+1
            hold_num=0
            print("PS",data_today["date"],position_status,data_today["open"])
            print(cash_value+hold_num*position_status*data_today["open"])
    value_list.append(cash_value+hold_num*position_status*data_today["open"])

print(cash_value)
print(len(value_list),len(factor_df))
df=pd.DataFrame({"value":value_list},index=factor_df["date"].values)
df.plot()
plt.show()
#'''
