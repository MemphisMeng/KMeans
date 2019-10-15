import pandas as pd
from sklearn.decomposition import PCA
from Kmeans import Kmeans
import matplotlib.pyplot as plt
from PreProcessing import normalize

file = pd.read_csv('C:\\Users\pc1\preprocessed_data.csv')
file['total_votes'] = file['vote_average'] * file['vote_count']
df = file.nlargest(250, 'total_votes')
df = normalize(df)

km = Kmeans(n_clusters=20, init="random")

pre = km.fit_predict(df.values)
pca = PCA(n_components=2)
pca = pca.fit_transform(df)
L1 = [n[0] for n in pca]
L2 = [n[1] for n in pca]

plt.figure(figsize=(10, 10))
plt.title("Prior Component Analysis")
plt.scatter(L1, L2, c=pre[0], marker='s')
plt.xlabel("Prior Component 1")
plt.ylabel("Prior Component 2")
plt.show()

