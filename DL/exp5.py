import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("Mall_Customers.csv")

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.figure(figsize=(10, 6))
dendrogram(linkage(X, method='ward'))
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

hc = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)

clusters = hc.fit_predict(X)

print("Cluster Labels:")
print(clusters)

data['Cluster'] = clusters

print("\nClustered Data:")
print(data.head(10))

print("\nNumber of customers in each cluster:")
print(data['Cluster'].value_counts().sort_index())