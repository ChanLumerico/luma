import __local__
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
from luma.classifier.neighbors import KNNClassifier
from luma.classifier.svm import KernelSVC
from luma.classifier.logistic import SoftmaxRegressor
from luma.ensemble.vote import VotingClassifier
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

X, y = load_wine(return_X_y=True)
X = X[:, [6, -1]]

sc = StandardScaler()
X_std = sc.fit_transform(X)

X_train, X_test, y_train, y_test = TrainTestSplit(X_std, y, test_size=0.2).get
X_cat = np.concatenate((X_train, X_test))
y_cat = np.concatenate((y_train, y_test))

estimators = [
    KNNClassifier(n_neighbors=5),
    KernelSVC(C=1.0, gamma=1.0, kernel='rbf'),
    SoftmaxRegressor(learning_rate=0.01, regularization='l2')
]

vote = VotingClassifier(estimators=estimators, voting='label')
vote.fit(X_train, y_train)

score = vote.score(X_cat, y_cat)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(vote, X_cat, y_cat, 
                     cmap='Spectral', 
                     title=f'{type(vote).__name__} [Acc: {score:.4f}]')
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_cat, vote.predict(X_cat), cmap='YlGn')
conf.plot(ax=ax2, show=True)