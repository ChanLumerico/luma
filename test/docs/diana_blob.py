import __local__
from luma.clustering.hierarchy import DivisiveClustering
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=300,
                  centers=3,
                  cluster_std=1.5,
                  random_state=10)

div = DivisiveClustering(n_clusters=3)
div.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(div, X, cmap='spring')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, div.labels)
sil.plot(ax=ax2, show=True)
