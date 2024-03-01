import __local__
from luma.clustering.density import DENCLUE
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
import numpy as np


num = 100
moons, _ = make_moons(n_samples=num, noise=0.05)

blobs, _ = make_blobs(n_samples=num, 
                      centers=[(-0.75,2.25), (1.0, -2.0)], 
                      cluster_std=0.3)

blobs2, _ = make_blobs(n_samples=num, 
                       centers=[(2,2.25), (-1, -2.0)], 
                       cluster_std=0.3)

X = np.vstack([moons, blobs, blobs2])

den = DENCLUE(h='auto',
              tol=0.01,
              max_climb=100,
              min_density=0.01)

den.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(den, X, cmap='plasma')
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, den.labels)
sil.plot(ax=ax2, show=True)
