import __local__
from luma.clustering.spectral import SpectralClustering
from luma.metric.clustering import SilhouetteCoefficient
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=300, 
                  centers=3, 
                  cluster_std=1.5, 
                  random_state=10)

sp = SpectralClustering(n_clusters=3, gamma=1.0)
sp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(sp, X, cmap='viridis')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, sp.labels)
sil.plot(ax=ax2, show=True)
