import numpy as np
from numpy import abs
import pandas as pd
from functools import reduce


a=list(range(1,10))
b=[4,5,6,7,8,9,10]
# a=np.array(a)
# b=np.array(b)
a=pd.Series(a,name="a")
# b=pd.Series(b,name="b")
#
# print((a-b).abs()/2)


# a="aa"
# l=["ab","bb","cc"]
# if a in l:
#     print("y")
# else:
#     print("n")

print(a)


def SMA(vals, n, m):
    # 算法1
    return reduce(lambda x, y: ((n - m) * x + y * m) / n, vals)

# print(a.rolling(window=2).apply(SMA))


result_list = [np.nan]
for x in range(1,len(a)):
    if x == 1:
        aa = SMA(a[:2], 3, 4)
        result_list.append(aa)
    else:
        aa = SMA([result_list[-1], a[x]],3,4)
        result_list.append(aa)
print(result_list)
print(SMA(a,3,4))


def sma_list(df, n, m):
    result_list = [np.nan]
    for x in range(1, len(df)):
        if x == 1:
            value = SMA(df.values[:2], n, m)
            result_list.append(value)
        else:
            value = SMA([result_list[-1], df.values[x]], n, m)
            result_list.append(value)
    result_series = pd.Series(result_list,name="sma")
    return result_series


print(sma_list(a, 3, 4))


a = ["c", "b", "a"]
b = ["a", "c", "b"]

print(sorted(a) == sorted(b))

a = {"a": 1, "b": 5, "c": 10}
print(a.keys())
print(list(a.keys()))
print(a["a"])

for x in range(0):
    print("no")














