import __local__
from luma.regressor.time_series import AutoRegressor

from matplotlib import pyplot as plt
import numpy as np


np.random.seed(42)
X = np.arange(200)
y = np.sqrt(X) * np.sin(X / 8)
y += np.random.normal(0, 5, 200)

p = 10

ar = AutoRegressor(p=p)
ar.fit(y)

in_pred = ar.predict(y)
next_pred = ar.predict(y, step=20)

X_ext = np.arange(len(y) + len(next_pred))
y_ext = np.append(y, next_pred)

plt.figure(figsize=(9, 5))
plt.plot(X, y, label='Original', color='black', alpha=0.4)
plt.plot(X[p:], in_pred, color='blue', lw=2, label='In-Sample')
plt.plot(X_ext[len(y):], next_pred, color='red', ls='--', lw=2, label='Future')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Auto-Regression Model [MSE: {ar.score(y):.4f}]')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()
