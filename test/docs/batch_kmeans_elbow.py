import __local__
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.linear import PCA
from luma.clustering.kmeans import MiniBatchKMeansClustering
from luma.metric.clustering import Inertia
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np


X, y = load_wine(return_X_y=True)

sc = StandardScaler()
X_std = sc.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

scores, labels, models = [], [], []
for i in range(2, 10):
    km = MiniBatchKMeansClustering(n_clusters=i,
                                   batch_size=100,
                                   max_iter=100)
    km.fit(X_pca)    
    scores.append(Inertia.score(X_pca, km.centroids))
    labels.append(km.labels)
    models.append(km)


def derivate_inertia(scores: list) -> list:
    diff = [0.0]
    for i in range(1, len(scores) - 1):
        diff.append(scores[i - 1] - scores[i + 1] / 2)
    
    diff.append(0.0)
    return diff


d_scores = derivate_inertia(scores)
best_n = np.argmax(d_scores)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(models[best_n], X_pca, cmap='viridis')
clst.plot(ax=ax1)

ax2.plot(range(2, 10), scores, marker='o', label='Inertia')
ax2.plot(range(2, 10), d_scores, marker='o', label='Absolute Derivative')

ax2.set_title('Inertia Plot')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Inertia')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
