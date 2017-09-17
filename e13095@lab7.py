import pandas as pd

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df=pd.read_csv('breaset-cancer.csv',names=['country','income per person','alc consumption','armed force rate','breast cancer per 100th person','co2 emmision','female employee rate','hivrate','internet use rate','lifeexpentancy','oil per person','polity score','Relectric per person','suicide 100th person','employee rate','urban rate'])
df.shape
df=df.fillna(0)
df= df.ix[1:] #removing first row
country=df['country']
del df['country']
#random state in kmeans makes it reproducable so that you can conduct the test several times and get the optimal result.
#if random state is not used in kmeans you may get differnt results(meaning differnt columns) at different times so its not compareable.
X=df.as_matrix(columns=None)
print X
z=X[:,:3]
est=KMeans(n_clusters=3,random_state=0).fit(z)
labels = est.labels_

x = df.ix[:, 1]
y = df.ix[:, 2]
z = df.ix[:, 3]


plt.subplot(1,3,1)
plt.title(" X vs Y")
plt.scatter(x , y , edgecolors ='k', c=labels)

plt.subplot(1,3,2)
plt.title(" X vs Z")
plt.scatter(x , z , edgecolors ='k', c=labels)

plt.subplot(1,3,3)
plt.title(" X vs W")
plt.scatter(y , z , edgecolors ='k', c=labels)

plt.show()
Z = linkage(X, 'average')
plt.figure(figsize=(25, 10))
dendrogram( Z, leaf_rotation=90,leaf_font_size=8)
dendrogram( Z,truncate_mode='lastp',p=12,show_leaf_counts=False,leaf_rotation=90,leaf_font_size=12,show_contracted=True)
plt.axhline(y=10, c='k')
plt.show()


