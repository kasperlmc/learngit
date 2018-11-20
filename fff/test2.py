import os
import pandas as pd
print(os.path.exists('C:\\Users\\lmc\\Downloads\\000001.csv'))
if os.path.exists('C:\\Users\\lmc\\Downloads\\000001.csv') :
    print('y')
data=pd.read_csv('C:\\Users\\lmc\\Downloads\\000001.csv',encoding='gbk')
print(data)







