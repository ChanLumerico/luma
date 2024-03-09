import __local__
from luma.ensemble.boost import AdaBoostClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
from luma.reduction.linear import KernelPCA
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


X, y = load_digits(n_class=4, return_X_y=True)
indices = np.random.choice(X.shape[0], size=500)

X_sample = X[indices]
y_sample = y[indices]

sc = StandardScaler()
X_sample_std = sc.fit_transform(X_sample)

kpca = KernelPCA(n_components=2, gamma=0.01, kernel='rbf')
X_kpca = kpca.fit_transform(X_sample_std)

X_train, X_test, y_train, y_test = TrainTestSplit(X_kpca, y_sample, 
                                                  test_size=0.3).get

ada = AdaBoostClassifier(n_estimators=10, learning_rate=1.0)
ada.fit(X_train, y_train)

X_cat = np.concatenate((X_train, X_test))
y_cat = np.concatenate((y_train, y_test))

score = ada.score(X_cat, y_cat)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(ada, X_cat, y_cat, 
                     title=f'AdaBoostClassifier [Acc: {score:.4f}]', 
                     cmap='viridis')
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_cat, ada.predict(X_cat))
conf.plot(ax=ax2, show=True)