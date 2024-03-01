from luma.classifier.naive_bayes import GaussianNaiveBayes
from luma.preprocessing.scaler import StandardScaler
from luma.reduction.linear import LDA
from luma.model_selection.split import TrainTestSplit
from luma.visual.evaluation import DecisionRegion, ConfusionMatrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = TrainTestSplit(X, y,
                                                  test_size=0.2,
                                                  random_state=42).get

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

gnb = GaussianNaiveBayes()
gnb.fit(X_train_lda, y_train)

X_concat = np.concatenate((X_train_lda, X_test_lda))
y_concat = np.concatenate((y_train, y_test))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

dec = DecisionRegion(gnb, X_concat, y_concat)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_concat, gnb.predict(X_concat))
conf.plot(ax=ax2, show=True)
