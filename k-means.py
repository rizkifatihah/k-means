import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

obejct = pd.read_csv("name_file.csv")
obejct.head()

plt.scatter(obejct.x, obejct.y, s =10, c = "c", marker = "o", alpha = 1)
plt.show()
x_array =  np.array(obejct)
print(x_array)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled
kmeans = KMeans(n_clusters = 3, random_state=123)
kmeans.fit(x_scaled)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
obejct["cluster"] = kmeans.labels_
output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = obejct.cluster, marker = "o", alpha = 1, )
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=1 , marker="s")
plt.title("Cluster Object")
plt.colorbar (output)
plt.show()
df = pd.DataFrame(obejct, columns = ['x', 'y','kluster'])
df.to_excel('name_file.xlsx')