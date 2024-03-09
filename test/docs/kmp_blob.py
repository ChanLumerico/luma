import __local__
from luma.clustering.kmeans import KMeansClusteringPlus
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=500, 
                  centers=7, 
                  cluster_std=1.2, 
                  random_state=10)

kmp = KMeansClusteringPlus(n_clusters=7, max_iter=300)
kmp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(kmp, X)
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, kmp.labels)
sil.plot(ax=ax2, show=True)