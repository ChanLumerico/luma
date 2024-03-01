from luma.classifier.discriminant import RDAClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV
from luma.reduction.linear import PCA
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
X_test_pca = pca.fit_transform(X_test_std)

param_grid = {'alpha': np.linspace(0.01, 1, 5),
              'gamma': np.linspace(0.01, 1, 5)}

grid = GridSearchCV(estimator=RDAClassifier(),
                    param_grid=param_grid,
                    cv=5,
                    refit=True, 
                    random_state=42)

grid.fit(X_train_pca, y_train)
rda_best = grid.best_model

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

X_trans = np.concatenate((X_train_pca, X_test_pca))
y_trans = np.concatenate((y_train, y_test))

dec = DecisionRegion(rda_best, X_trans, y_trans)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_trans, rda_best.predict(X_trans))
conf.plot(ax=ax2, show=True)
