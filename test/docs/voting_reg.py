import __local__
from luma.ensemble.vote import VotingRegressor
from luma.regressor.neighbors import KNNRegressor
from luma.regressor.svm import KernelSVR
from luma.regressor.tree import DecisionTreeRegressor
from luma.regressor.poly import PolynomialRegressor
from luma.regressor.linear import KernelRidgeRegressor
from luma.preprocessing.scaler import StandardScaler
from luma.metric.regression import RootMeanSquaredError

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

X = np.linspace(-5, 5, 500).reshape(-1, 1)
y = (np.sin(2 * X) * np.cos(np.exp(X / 2))).flatten()
y += 0.2 * np.random.randn(500)

sc = StandardScaler()
y_trans = sc.fit_transform(y)

estimators = [
    PolynomialRegressor(deg=9, alpha=0.01, regularization='l2'),
    KNNRegressor(n_neighbors=10), 
    KernelSVR(C=1.0, gamma=3.0, kernel='rbf'),
    DecisionTreeRegressor(max_depth=7, random_state=42),
    KernelRidgeRegressor(alpha=1.0, gamma=1.0, kernel='rbf')
]

vote = VotingRegressor(estimators=estimators)
vote.fit(X, y_trans)

y_pred = vote.predict(X)
score = vote.score(X, y_trans, metric=RootMeanSquaredError)

fig = plt.figure(figsize=(10, 5))
plt.scatter(X, y_trans, 
            s=10, c='black', alpha=0.3, 
            label=r'$y=\sin{2x}\cdot\cos{e^{x/2}}+\epsilon$')

for est in vote:
    est_pred = est.predict(X)
    plt.plot(X, est_pred, 
             alpha=0.5, label=f'{type(est).__name__}')
    plt.fill_between(X.flatten(), est_pred, y_pred, alpha=0.05)

plt.plot(X, y_pred, lw=2, c='blue', label='Predicted Plot')
plt.legend()
plt.xlabel('x')
plt.ylabel('y (Standardized)')
plt.title(f'VotingRegressor [RMSE: {score:.4f}]')

plt.tight_layout()
plt.show()