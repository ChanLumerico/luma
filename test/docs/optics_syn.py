import __local__
from luma.clustering.density import OPTICS
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np


num = 200
moons, _ = make_moons(n_samples=num, noise=0.05)

blobs, _ = make_blobs(n_samples=num, 
                      centers=[(-0.75,2.25), (1.0, -2.0)], 
                      cluster_std=0.3)

blobs2, _ = make_blobs(n_samples=num, 
                       centers=[(2,2.25), (-1, -2.0)], 
                       cluster_std=0.3)

X = np.vstack([moons, blobs, blobs2])

opt = OPTICS(epsilon=0.3,
             min_points=5,
             threshold=1.5)

opt.fit(X)
opt.plot_reachability()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(opt, X)
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, opt.labels)
sil.plot(ax=ax2, show=True)
