import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample=100
x=np.linspace(0,10,nsample)
X=np.column_stack((x,x**2))
X=sm.add_constant(X)
#print(x)
beta=np.array([1,0.1,10])
e=np.random.normal(size=nsample)
y=np.dot(X,beta)+e
print(y)
model=sm.OLS(y,X)
result=model.fit()
print(result.summary())
y_fitted=result.fittedvalues
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(x,y,'o',label='data')
ax.plot(x,y_fitted,'r--.',label='ols')
ax.legend(loc='best')
ax.axis((0,2,0,50))
plt.show()



