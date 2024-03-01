import __local__
from luma.clustering.density import DBSCAN
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt


X1, y1 = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
X2, y2 = make_moons(n_samples=500, noise=0.05, random_state=42)

db1 = DBSCAN(epsilon=0.2,
             min_points=5,
             metric='euclidean')

db2 = DBSCAN(epsilon=0.2,
             min_points=5,
             metric='euclidean')

db1.fit(X1)
db2.fit(X2)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst1 = ClusterPlot(db1, X1, cmap='rainbow')
clst1.plot(ax=ax1)

clst2 = ClusterPlot(db2, X2, cmap='summer')
clst2.plot(ax=ax2, show=True)
