import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample=50
groups=np.zeros(nsample,int)
#print(groups)
groups[20:40]=1
groups[40:]=2
dummy=sm.categorical(groups,drop=True)
#print(dummy)
x=np.linspace(0,20,nsample)
X=np.column_stack((x,dummy))
X=sm.add_constant(X)
beta=[10,1,1,3,8]
beta=np.array(beta)
e=np.random.normal()
y=np.dot(X,beta)+e
result=sm.OLS(y,X).fit()
print(result.summary())

