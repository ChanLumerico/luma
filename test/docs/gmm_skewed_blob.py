import __local__
from luma.clustering.mixture import GaussianMixture
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


X, y = make_blobs(n_samples=300, 
                  centers=3, 
                  cluster_std=2.0, 
                  random_state=100)

trans_mat = np.array([[1, 2],
                      [0, 1]])

X_skew = X @ trans_mat

gmm = GaussianMixture(n_clusters=3, max_iter=100)
gmm.fit(X_skew)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(gmm, X_skew, cmap='viridis')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X_skew, gmm.labels)
sil.plot(ax=ax2, show=True)
