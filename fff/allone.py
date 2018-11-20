import pandas as pd
import numpy as np
import csv
import xlwt
data=pd.read_excel('D:\\workdata\\allone.xlsx')
print(data)
print(data.columns)
l=list(data.columns)
print(l)
data1=pd.DataFrame()
for x in l:
    if np.array(data[x][data[x]<0]).size!=0:
        print(data[x][data[x]<0])
        print((data[x].max() - data[x])/(data[x].max() - data[x].min()))
        data1[x]=(data[x].max() - data[x])/(data[x].max() - data[x].min())
    else:
        print((data[x] - data[x].min())/(data[x].max() - data[x].min()))
        data1[x]=(data[x] - data[x].min())/(data[x].max() - data[x].min())
print(data1)

data1.to_csv('D:\\workdata\\allone1.csv')

