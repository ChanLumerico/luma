import __local__
from luma.classifier.svm import SVC
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.linear import LDA
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import RandomizedSearchCV
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y, 
                                                  test_size=0.2,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

lda = LDA(n_components=2)
X_train_tr = lda.fit_transform(X_train_std, y_train)
X_test_tr = lda.transform(X_test_std)

param_dist = {'C': np.logspace(-3, 2, 5),
              'learning_rate': np.logspace(-3, -0.5, 5)}

rand = RandomizedSearchCV(estimator=SVC(),
                          param_dist=param_dist,
                          max_iter=10,
                          cv=10,
                          refit=True,
                          random_state=42,
                          verbose=True)

rand.fit(X_train_tr, y_train)
svc_best = rand.best_model

X_concat = np.concatenate((X_train_tr, X_test_tr))
y_concat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(svc_best, X_concat, y_concat)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_concat, svc_best.predict(X_concat))
conf.plot(ax=ax2, show=True)

