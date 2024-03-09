import __local__
from luma.clustering.mixture import MultinomialMixture
from luma.visual.evaluation import ConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np


def generate_multi_dataset(n_samples: int, 
                           component_probs: list, 
                           mixture_weights: list) -> tuple:
    n_components = len(component_probs)
    dataset, labels = [], []
    for _ in range(n_samples):
        component = np.random.choice(range(n_components), p=mixture_weights)
        sample = np.random.multinomial(1, component_probs[component])
        
        dataset.append(sample)
        labels.append(component)
        
    return np.array(dataset), np.array(labels)


X, y = generate_multi_dataset(n_samples=300,
                              component_probs=[[0.2, 0.8],
                                               [0.7, 0.3]],
                              mixture_weights=[0.5, 0.5])

mmm = MultinomialMixture(n_clusters=2, max_iter=1000)
mmm.fit(X)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

n_clusters = 2
bincounts = [np.bincount(X[y == i, 0]) for i in range(2)]

width = 0.2
ax1.bar(np.arange(n_clusters) - width / 2, bincounts[0], 
        width=width,
        label='Mixture 0')

ax1.bar(np.arange(n_clusters) + width / 2, bincounts[1], 
        width=width,
        label='Mixture 1')

ax1.set_xticks([0, 1])
ax1.set_ylabel('Conut')
ax1.set_title('Frequency Counts')
ax1.legend(loc='upper right')

conf = ConfusionMatrix(y_true=y, y_pred=mmm.labels)
conf.plot(ax=ax2, show=True)