import __local__
from luma.clustering.affinity import AdaptiveAffinityPropagation
from luma.visual.evaluation import ClusterPlot
from luma.metric.clustering import SilhouetteCoefficient

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=500, 
                  cluster_std=1.3, 
                  centers=5, 
                  random_state=10)

aap = AdaptiveAffinityPropagation(max_iter=100,
                                  damping=0.7,
                                  lambda_param=0.5,
                                  preference='min')

aap.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(aap, X)
clst.plot(ax=ax1)

sil = SilhouetteCoefficient(X, aap.labels)
sil.plot(ax=ax2, show=True)
