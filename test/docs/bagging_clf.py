import __local__
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.linear import PCA
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import RandomizedSearchCV
from luma.ensemble.bagging import BaggingClassifier
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np


X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.2,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

param_dist = {'n_estimators': [5, 10, 20, 50, 100],
              'max_samples': np.linspace(0.5, 1.0, 5),
              'bootstrap': [True, False],
              'bootstrap_feature': [True, False],
              'random_state': [42]}

rand = RandomizedSearchCV(estimator=BaggingClassifier(),
                          param_dist=param_dist,
                          max_iter=10,
                          cv=5,
                          refit=True,
                          random_state=42,
                          verbose=True)

rand.fit(X_train_pca, y_train)
bag_best = rand.best_model

X_cat = np.concatenate((X_train_pca, X_test_pca))
y_cat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(bag_best, X_cat, y_cat)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_cat, bag_best.predict(X_cat))
conf.plot(ax=ax2, show=True)