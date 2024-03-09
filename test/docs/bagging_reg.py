import __local__
from luma.ensemble.bagging import BaggingRegressor
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import RandomizedSearchCV
from luma.metric.regression import RootMeanSquaredError
from luma.visual.evaluation import ResidualPlot

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np


X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.3,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

param_dist = {'n_estimators': [10, 20, 50],
              'max_samples': np.linspace(0.5, 1.0, 5),
              'max_features': np.linspace(0.5, 1.0, 5),
              'bootstrap': [True, False],
              'bootstrap_feature': [True, False],
              'random_state': [42]}

rand = RandomizedSearchCV(estimator=BaggingRegressor(),
                          param_dist=param_dist,
                          cv=5,
                          max_iter=10,
                          metric=RootMeanSquaredError,
                          maximize=False, 
                          refit=True,
                          random_state=42,
                          verbose=True)

rand.fit(X_train_std, y_train)
bag_best: BaggingRegressor = rand.best_model

X_cat = np.concatenate((X_train_std, X_test_std))
y_cat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

res = ResidualPlot(bag_best, X_cat, y_cat)
res.plot(ax=ax1)

train_scores, test_scores = [], []
for tree, _ in bag_best:
    train_scores.append(tree.score(X_train_std, y_train, RootMeanSquaredError))
    test_scores.append(tree.score(X_test_std, y_test, RootMeanSquaredError))

ax2.plot(range(bag_best.n_estimators), train_scores, 
         marker='o', label='Train Scores')
ax2.plot(range(bag_best.n_estimators), test_scores, 
         marker='o', label='Test Scores')

ax2.set_xlabel('Base Estimators')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE of Base Estimators')
ax2.legend()

plt.tight_layout()
plt.show()
