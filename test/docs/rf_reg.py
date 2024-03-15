import __local__
from luma.ensemble.forest import RandomForestRegressor
from luma.preprocessing.scaler import StandardScaler
from luma.metric.regression import RootMeanSquaredError
from luma.visual.evaluation import ResidualPlot

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

X = np.linspace(-4, 4, 400).reshape(-1, 1)
y = (np.cos(2 * X) * np.sin(np.exp(-np.ceil(X / 2)))).flatten()
y += 0.2 * np.random.randn(400)

sc = StandardScaler()
y_trans = sc.fit_transform(y)

forest = RandomForestRegressor(n_trees=10,
                               max_depth=7,
                               bootstrap=True)

forest.fit(X, y_trans)

y_pred = forest.predict(X)
score = forest.score(X, y_trans, metric=RootMeanSquaredError)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.scatter(X, y_trans, 
            s=10, c='black', alpha=0.3, 
            label=r'$y=\cos{2x}\cdot$' + \
                r'$\sin{e^{-\left\lceil x/2\right\rceil}}+\epsilon$')

for tree in forest.trees:
    ax1.plot(X, tree.predict(X), c='violet', alpha=0.2)

ax1.plot(X, y_pred, lw=2, c='purple', label='Predicted Plot')
ax1.legend(loc='upper right')
ax1.set_xlabel('x')
ax1.set_ylabel('y (Standardized)')
ax1.set_title(f'RandomForestRegressor [RMSE: {score:.4f}]')

res = ResidualPlot(forest, X, y_trans)
res.plot(ax=ax2)

plt.tight_layout()
plt.show()