import __local__
from luma.ensemble.boost import GradientBoostingRegressor
from luma.preprocessing.scaler import StandardScaler
from luma.metric.regression import RootMeanSquaredError
from luma.visual.evaluation import ResidualPlot

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


X = np.linspace(-4, 4, 400).reshape(-1, 1)
y = (np.sin(3 * X) * np.cos(np.exp(X / 2))).flatten()
y += 0.15 * np.random.randn(400)

sc = StandardScaler()
y_trans = sc.fit_transform(y)

gb = GradientBoostingRegressor(n_estimators=50,
                               learning_rate=0.1,
                               subsample=1.0,
                               loss='mae',
                               max_depth=3)

gb.fit(X, y_trans)

y_pred = gb.predict(X)
score = gb.score(X, y_trans, metric=RootMeanSquaredError)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.scatter(X, y_trans, 
            s=10, c='black', alpha=0.3, 
            label=r'$y=\sin{3x}\cdot\cos{e^{x/2}}+\epsilon$')

for tree in gb:
    ax1.plot(X, tree.predict(X), c='pink', alpha=0.1)

ax1.plot(X, y_pred, lw=2, c='crimson', label='Predicted Plot')
ax1.legend(loc='upper right')
ax1.set_xlabel('x')
ax1.set_ylabel('y (Standardized)')
ax1.set_title(f'GradientBoostingRegressor [RMSE: {score:.4f}]')

res = ResidualPlot(gb, X, y_trans)
res.plot(ax=ax2)

plt.tight_layout()
plt.show()
