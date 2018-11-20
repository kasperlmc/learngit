from pymongo import MongoClient
import pandas as pd


client = MongoClient()
db = client.pythondb
data=pd.read_csv('D:\\workdata\\new\\volatility_data_1.csv',index_col='Unnamed: 0')
print(data)
posts=db.posts
post = posts.find({"日期" : "1992-10-20"})
for x in post:
    print(x)
#data.index=data['secID'].values
#print(data.ix['600237.XSHG'])
#print(len(data))
#db.users.ensureIndex({"username": 1})
'''
df=pd.DataFrame(columns=['_id','日期','股票代码','名称','收盘价','最高价','最低价','开盘价','前收盘','涨跌额','涨跌幅','换手率','成交量','成交金额','总市值','流通市值'])
for x in post:
    df=df.append(x,ignore_index=True)
df['volatility']=pd.DataFrame({'volatility':(df['最高价']-df['最低价'])/df['前收盘']},index=df.index)
df=df.sort_values(by='volatility',ascending=False)
stock_values=df['股票代码'].values[:100]
print(stock_values)


'''


