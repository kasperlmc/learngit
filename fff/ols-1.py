import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample=100
x=np.linspace(0,10,nsample)
#print(x)
X=sm.add_constant(x)
print(X)
beta=np.array([1,10])
e=np.random.normal(size=nsample)
y=np.dot(X,beta)+e
model=sm.OLS(y,x)
result=model.fit()
print(result.summary())
y_fitted=result.fittedvalues
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(x,y,'o',label='data')
ax.plot(x,y_fitted,'r--.',label='ols')
ax.legend(loc='best')
ax.axis((0,2,-1,25))
plt.show()
