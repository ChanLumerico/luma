import __local__
from luma.classifier.tree import DecisionTreeClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.linear import PCA
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import RandomizedSearchCV
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np


X, y = make_classification(n_samples=400,
                           n_classes=4,
                           n_features=5,
                           n_informative=4,
                           n_redundant=1,
                           n_clusters_per_class=1,
                           class_sep=2.0,
                           random_state=0)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.2,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

param_dist = {'max_depth': [5, 10, 20, 50],
              'criterion': ['gini', 'entropy'],
              'min_impurity_decrease': np.linspace(0, 0.2, 10)}

rand = RandomizedSearchCV(estimator=DecisionTreeClassifier(),
                          param_dist=param_dist,
                          max_iter=50,
                          cv=5,
                          refit=True,
                          random_state=42,
                          verbose=True)

rand.fit(X_train_pca, y_train)
tree_best = rand.best_model

X_concat = np.concatenate((X_train_pca, X_test_pca))
y_concat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(tree_best, X_concat, y_concat, cmap='plasma')
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_concat, tree_best.predict(X_concat))
conf.plot(ax=ax2, show=True)
