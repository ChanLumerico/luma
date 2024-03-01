import __local__
from luma.classifier.svm import KernelSVC
from luma.classifier.naive_bayes import GaussianNaiveBayes
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.selection import RFE
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import RandomizedSearchCV
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np


X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.3,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

rfe = RFE(estimator=GaussianNaiveBayes(),
          n_features=2,
          step_size=1,
          cv=5,
          random_state=42,
          verbose=True)

rfe.fit(X_train_std, y_train)
X_train_rfe = rfe.transform(X_train_std)
X_test_rfe = rfe.transform(X_test_std)

param_dist = {'C': np.logspace(-3, 2, 5),
              'deg': range(2, 10),
              'gamma': np.logspace(-3, 1, 5),
              'learning_rate': np.logspace(-3, -1, 5),
              'kernel': ['poly', 'rbf', 'sigmoid']}

rand = RandomizedSearchCV(estimator=KernelSVC(),
                          param_dist=param_dist,
                          max_iter=20,
                          cv=5,
                          refit=True,
                          random_state=42,
                          verbose=True)

rand.fit(X_train_rfe, y_train)
ksvc_best = rand.best_model

X_concat = np.concatenate((X_train_rfe, X_test_rfe))
y_concat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(ksvc_best, X_concat, y_concat)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_concat, ksvc_best.predict(X_concat))
conf.plot(ax=ax2, show=True)
