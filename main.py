import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('players_22.csv', low_memory=False)


data = df[['overall', 'value_eur', 'potential', 'wage_eur', 'age']].dropna().values


pca = PCA(n_components=2)
pcaData = pca.fit_transform(data)


model = KMeans(n_clusters=5)
model.fit(pcaData)

centros = model.cluster_centers_


plt.scatter(pcaData[:,0], pcaData[:,1], c=model.labels_.astype(float))
plt.scatter(centros[:,0], centros[:,1], s=200, marker='X', c='red')
plt.xlabel("PCA Componente 1")
plt.ylabel("PCA Componente 2")
plt.title("K-Means FIFA Players Clustering (5 variables → PCA)")
plt.show()