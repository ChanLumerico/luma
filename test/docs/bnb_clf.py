from luma.classifier.naive_bayes import BernoulliNaiveBayes
from luma.model_selection.split import TrainTestSplit
from luma.visual.evaluation import ConfusionMatrix, ROCCurve

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np


X, y = make_classification(n_samples=500, 
                           n_informative=10, 
                           n_redundant=10, 
                           n_clusters_per_class=1, 
                           random_state=42,
                           n_classes=3)

X_binary = (X > 0).astype(int)
X_train, X_test, y_train, y_test = TrainTestSplit(X_binary, y,
                                                  test_size=0.2,
                                                  random_state=42).get

bnb = BernoulliNaiveBayes()
bnb.fit(X_train, y_train)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

X_concat = np.concatenate((X_train, X_test))
y_concat = np.concatenate((y_train, y_test))

conf = ConfusionMatrix(y_concat, bnb.predict(X_concat))
conf.plot(ax=ax1)

roc = ROCCurve(y_concat, bnb.predict_proba(X_concat))
roc.plot(ax=ax2, show=True)
