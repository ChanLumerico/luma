import __local__
from luma.clustering.hierarchy import AgglomerativeClustering
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=300,
                  centers=5,
                  cluster_std=1.0,
                  random_state=10)

agg = AgglomerativeClustering(n_clusters=5, linkage='average')
agg.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(agg, X)
clst.plot(ax=ax1)

agg.plot_dendrogram(ax=ax2, hide_indices=True, show=True)
