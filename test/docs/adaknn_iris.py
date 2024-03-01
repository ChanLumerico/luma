import __local__
from luma.classifier.naive_bayes import GaussianNaiveBayes
from luma.classifier.neighbors import AdaptiveKNNClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.selection import SFS
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.2,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

sfs = SFS(estimator=GaussianNaiveBayes(),
          n_features=2,
          test_size=0.2,
          cv=5,
          random_state=42,
          verbose=True)

sfs.fit(X_train_std, y_train)
X_train_sfs = sfs.transform(X_train_std)
X_test_sfs = sfs.transform(X_test_std)

param_grid = {'n_density': range(2, 20)}

grid = GridSearchCV(estimator=AdaptiveKNNClassifier(),
                    param_grid=param_grid,
                    cv=5,
                    refit=True,
                    random_state=42)

grid.fit(X_train_sfs, y_train)
knn_best = grid.best_model

X_concat = np.concatenate((X_train_sfs, X_test_sfs))
y_concat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(knn_best, X_concat, y_concat)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_concat, knn_best.predict(X_concat))
conf.plot(ax=ax2, show=True)
