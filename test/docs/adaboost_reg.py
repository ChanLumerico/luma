import __local__
from luma.ensemble.boost import AdaBoostRegressor
from luma.preprocessing.scaler import StandardScaler
from luma.metric.regression import RootMeanSquaredError
from luma.visual.evaluation import ResidualPlot

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = (np.sin(5 * X) - X).flatten() + 0.15 * np.random.randn(200)

rand_idx = np.random.choice(200, size=80)
y[rand_idx] += 0.75 * np.random.randn(80)

sc = StandardScaler()
y_trans = sc.fit_transform(y)

ada = AdaBoostRegressor(n_estimators=50,
                        learning_rate=1.0,
                        loss='linear',
                        max_depth=5)

ada.fit(X, y_trans)

y_pred = ada.predict(X)
score = ada.score(X, y_trans, metric=RootMeanSquaredError)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.scatter(X, y_trans, s=10, c='black', alpha=0.3, label='Original Data')  
ax1.plot(X, y_pred, lw=2, c='blue', label='Predicted Plot')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y (Standardized)')
ax1.set_title(f'AdaBoost Regression [RMSE: {score:.4f}]')

res = ResidualPlot(ada, X, y_trans)
res.plot(ax=ax2)
ax2.set_ylim(y_trans.min(), y_trans.max())

plt.tight_layout()
plt.show()
