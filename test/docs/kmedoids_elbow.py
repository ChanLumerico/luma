import __local__
from luma.reduction.linear import PCA
from luma.clustering.kmeans import KMedoidsClustering
from luma.metric.clustering import Inertia
from luma.visual.evaluation import ClusterPlot

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


X, y = load_iris(return_X_y=True)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

scores, labels, models = [], [], []
for i in range(2, 10):
    km = KMedoidsClustering(n_clusters=i,
                            max_iter=300,
                            random_state=42)
    km.fit(X_pca)    
    scores.append(Inertia.score(X_pca, km.medoids))
    labels.append(km.labels)
    models.append(km)


def derivate_inertia(scores: list) -> list:
    diff = [0.0]
    for i in range(1, len(scores) - 1):
        diff.append(scores[i - 1] - scores[i + 1])
    
    diff.append(0.0)
    return diff


d_scores = derivate_inertia(scores)
best_n = np.argmax(d_scores)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

clst = ClusterPlot(models[best_n], X_pca)
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
