import pandas as pd
from Kmeans import Kmeans
import matplotlib.pyplot as plt
from PreProcessing import normalize

file = pd.read_csv('C:\\Users\pc1\preprocessed_data.csv')
file['budget'].loc[file['budget'].isna()] = 0
file['popularity'].loc[file['popularity'].isna()] = 0
file['revenue'].loc[file['revenue'].isna()] = 0
file['runtime'].loc[file['runtime'].isna()] = 0
file['vote_average'].loc[file['vote_average'].isna()] = 0
file['vote_count'].loc[file['vote_count'].isna()] = 0

file = normalize(file)

K = range(1, 21)
# K = range(1, 21)
sum_E1 = []
sum_E2 = []
for k in K:
    km1 = Kmeans(n_clusters=k, init="random")
    km2 = Kmeans(n_clusters=k, init="k-means++")
    labels1, errors1 = km1.fit_predict(file.values)
    labels2, errors2 = km2.fit_predict(file.values)
    sum_E1.append(errors1)
    sum_E2.append(errors2)

plt.plot(K, sum_E1, 'bx-', label='k-means')
plt.plot(K, sum_E2, 'rx', label='k-means++')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Errors')
plt.title('Elbow Method For Optimal K')
plt.legend()
plt.show()
