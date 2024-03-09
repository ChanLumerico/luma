from luma.clustering.kmeans import KMeansClustering, KMediansClustering
from luma.visual.evaluation import DecisionRegion

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=500, 
                  centers=5, 
                  cluster_std=1.0, 
                  random_state=10)

kmed = KMediansClustering(n_clusters=5, max_iter=100)
kmed.fit(X)

kmp = KMeansClustering(n_clusters=5, max_iter=100)
kmp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec_kmed = DecisionRegion(kmed, X, kmed.labels)
dec_kmed.plot(ax=ax1)

dec_kmp = DecisionRegion(kmp, X, kmp.labels)
dec_kmp.plot(ax=ax2, show=True)