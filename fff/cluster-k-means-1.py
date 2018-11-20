from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)
digits=load_digits()
data=scale(digits.data)
n_samples,n_features=data.shape
print(n_samples,n_features)
n_digits=len(np.unique(digits.target))
labels=digits.target
#print(digits.target.size)
sample_size=300
print('*'*82)
model=KMeans(init='k-means++',n_clusters=n_digits,n_init=10)
model.fit(data)
print(metrics.adjusted_rand_score(labels,model.labels_))
pca=PCA(n_components=n_digits).fit(data)
model1=KMeans(init=pca.components_,n_clusters=n_digits,n_init=1)
model1.fit(data)
print(metrics.adjusted_rand_score(labels,model1.labels_))
reduced_data=PCA(n_components=2).fit_transform(data)
model2=KMeans(init='k-means++',n_clusters=n_digits,n_init=10)
model2.fit(reduced_data)
print(metrics.adjusted_rand_score(labels,model2.labels_))
print(reduced_data.shape)