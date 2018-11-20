from pymongo import MongoClient
import pandas as pd
from pymongo import MongoClient
from datetime import datetime


client = MongoClient()
db = client.pythondb
industry_code=db.industry_code
func=lambda x:datetime.strptime(x, "%Y-%m-%d")
data=pd.read_csv('D:\\workdata\\data\\stock_industry.csv',index_col='Unnamed: 0')

data["intoDate"]=data["intoDate"].apply(func)
data["outDate"]=data["outDate"].apply(func)
dict_list = data.to_dict('records')
industry_code.insert_many(dict_list)
print(end)

'''

dict_list = data.to_dict('records')
print(dict_list[:5])
explain_factors.insert_many(dict_list)
print('end')

'''
