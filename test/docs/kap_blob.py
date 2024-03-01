import __local__
from luma.clustering.affinity import KernelAffinityPropagation
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=400, 
                  cluster_std=1.2, 
                  centers=4, 
                  random_state=10)

kap = KernelAffinityPropagation(max_iter=100,
                               damping=0.7,
                               gamma=0.01,
                               preference='min',
                               kernel='rbf')

kap.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(kap, X, cmap='Spectral')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, kap.labels)
sil.plot(ax=ax2, show=True)
