import numpy as np
from numpy import abs
import pandas as pd


a=[1,2,3,4,5,6,7]
b=[4,5,6,7,8,9,10]
a=np.array(a)
b=np.array(b)
a=pd.Series(a,name="a")
b=pd.Series(b,name="b")

print((a-b).abs()/2)