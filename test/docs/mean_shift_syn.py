import __local__
from luma.clustering.density import MeanShiftClustering
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=600, 
                  centers=3, 
                  cluster_std=2.0, 
                  random_state=42)

mshift = MeanShiftClustering(bandwidth=5,
                             max_iter=300,
                             tol=0.001)

mshift.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(mshift, X)
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, mshift.labels)
sil.plot(ax=ax2, show=True)
