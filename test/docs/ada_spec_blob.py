import __local__
from luma.clustering.spectral import AdaptiveSpectralClustering
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=400, 
                  centers=5,  
                  cluster_std=1.5, 
                  random_state=10)

asp = AdaptiveSpectralClustering(gamma=0.1, max_clusters=10)
asp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(asp, X, cmap='viridis')
clst.plot(ax=ax1)

im = ax2.imshow(asp.W, cmap='inferno', vmin=-0.1, vmax=1.0)
ax2.set_title('Adapted Similarity Matrix')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Samples')

plt.colorbar(im, fraction=0.05)
plt.tight_layout()
plt.show()
