from pymongo import MongoClient
import pandas as pd


client = MongoClient()
db = client.pythondb
data=pd.read_csv('D:\\workdata\\new\\trade_day.csv',encoding='gbk',index_col='Unnamed: 0')
date_values=data['date'].values
print(date_values)
posts=db.posts

for i in range(len(date_values)):
    print(date_values[i])
    post=posts.find({"日期" : date_values[i]})
    df=pd.DataFrame(columns=['_id','日期','股票代码','名称','收盘价','最高价','最低价','开盘价','前收盘','涨跌额','涨跌幅','换手率','成交量','成交金额','总市值','流通市值'])
    for x in post:
        df=df.append(x,ignore_index=True)
    print(len(df))
    df['volatility']=pd.DataFrame({'volatility':(df['最高价']-df['最低价'])/df['前收盘']},index=df.index)
    df=df.sort_values(by='volatility',ascending=False)
    stock_values=df['股票代码'].values
    if i==0:
        df_all=pd.DataFrame({date_values[i]:stock_values},index=list(range(len(stock_values))))
        df_100=pd.DataFrame({date_values[i]:stock_values[:100]},index=list(range(len(stock_values[:100]))))
    else:
        df3=pd.DataFrame({date_values[i]:stock_values[:100]},index=list(range(len(stock_values[:100]))))
        df4=pd.DataFrame({date_values[i]:stock_values},index=list(range(len(stock_values))))
        df_all=pd.concat([df_all,df4],axis=1)
        df_100=pd.concat([df_100,df3],axis=1)
        df_all.to_csv('D:\\workdata\\new\\stock_list_all.csv')
        df_100.to_csv('D:\\workdata\\new\\stock_list_100.csv')
