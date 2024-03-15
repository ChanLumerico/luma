import __local__
from luma.ensemble.forest import RandomForestClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
from luma.reduction.linear import KernelPCA
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

X, y = load_digits(n_class=5, return_X_y=True)
indices = np.random.choice(X.shape[0], size=500)

X_sample = X[indices]
y_sample = y[indices]

sc = StandardScaler()
X_sample_std = sc.fit_transform(X_sample)

kpca = KernelPCA(n_components=2, gamma=0.01, kernel='rbf')
X_kpca = kpca.fit_transform(X_sample_std)

X_train, X_test, y_train, y_test = TrainTestSplit(X_kpca, y_sample, 
                                                  test_size=0.3).get

forest = RandomForestClassifier(n_trees=10,
                                max_depth=100,
                                criterion='gini',
                                min_impurity_decrease=0.01,
                                bootstrap=True)

forest.fit(X_train, y_train)

X_cat = np.concatenate((X_train, X_test))
y_cat = np.concatenate((y_train, y_test))

score = forest.score(X_cat, y_cat)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(forest, X_cat, y_cat, cmap='Spectral')
dec.plot(ax=ax1)
ax1.set_title(f'RandomForestClassifier [Acc: {score:.4f}]')

conf = ConfusionMatrix(y_cat, forest.predict(X_cat), cmap='BuPu')
conf.plot(ax=ax2, show=True)
