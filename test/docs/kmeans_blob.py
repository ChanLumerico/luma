import __local__
from luma.clustering.kmeans import KMeansClustering
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=300, 
                  centers=4, 
                  cluster_std=1.2, 
                  random_state=10)

kmeans = KMeansClustering(n_clusters=4,
                          max_iter=100)

kmeans.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(kmeans, X)
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, kmeans.labels)
sil.plot(ax=ax2, show=True)