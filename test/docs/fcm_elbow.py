import __local__
from luma.clustering.kmeans import FuzzyCMeansClustering
from luma.metric.clustering import Inertia
from luma.visual.evaluation import ClusterPlot, InertiaPlot

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=400, 
                  centers=4, 
                  cluster_std=1.7, 
                  random_state=10)

scores, labels, models = [], [], []
for i in range(2, 10):
    fcm = FuzzyCMeansClustering(n_clusters=i,
                               max_iter=100,
                               m=2.0,
                               random_state=42)
    fcm.fit(X)    
    scores.append(Inertia.score(X, fcm.centers))
    labels.append(fcm.labels)
    models.append(fcm)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(models[2], X)
clst.plot(ax=ax1)

inp = InertiaPlot(scores, list(range(2, 10)))
inp.plot(ax=ax2, show=True)
