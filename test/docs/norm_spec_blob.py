import __local__
from luma.clustering.spectral import NormalizedSpectralClustering
from luma.metric.clustering import SilhouetteCoefficient
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=400, 
                  centers=4, 
                  cluster_std=1.0, 
                  random_state=10)

X[:, 1] *= 5

nsp = NormalizedSpectralClustering(n_clusters=4, 
                                   gamma=1.0, 
                                   strategy='symmetric')

nsp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(nsp, X, cmap='coolwarm')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, nsp.labels)
sil.plot(ax=ax2, show=True)
