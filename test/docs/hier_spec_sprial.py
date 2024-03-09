import __local__
from luma.clustering.spectral import HierarchicalSpectralClustering
from luma.visual.evaluation import ClusterPlot

import numpy as np
import matplotlib.pyplot as plt


def generate_spiral_dataset(points_per_spiral=100, noise=0.7):
    n = points_per_spiral
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * noise

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * noise

    X = np.concatenate((x_a, x_b))
    y = np.concatenate((np.zeros(n), np.ones(n)))

    return X, y


X, y = generate_spiral_dataset()

hsp = HierarchicalSpectralClustering(n_clusters=2,
                                     method='agglomerative',
                                     linkage='single',
                                     gamma=0.5)

hsp.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(hsp, X)
clst.plot(ax=ax1)

hsp.plot_dendrogram(ax=ax2)
ax2.set_yscale('log')
ax2.set_ylabel('Distance (log)')
ax2.set_ylim(1e-5, 1e-0)

plt.tight_layout()
plt.show()
