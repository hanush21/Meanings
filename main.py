import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("players_22.csv", low_memory=False)


df_clean = df[['sofifa_id', 'overall', 'value_eur', 'potential', 'wage_eur', 'age', 'short_name']].dropna()

ids = df_clean['sofifa_id'].values
data = df_clean[['overall', 'value_eur', 'potential', 'wage_eur', 'age']].values
nombres = df_clean['short_name'].values

model = KMeans(n_clusters=3)
model.fit(data)

centros = model.cluster_centers_
labels = model.labels_


result = pd.DataFrame({"ID": ids, "Cluster": labels, "Nombre": nombres})
print(result.head())


plt.scatter(data[:,0], data[:,2], c=labels.astype(float))
plt.scatter(centros[:,0], centros[:,2], s=200, marker='X', c='red')
plt.xlabel("Overall")
plt.ylabel("Potential")
plt.title("K-Means FIFA Players (sin PCA)")
plt.show()