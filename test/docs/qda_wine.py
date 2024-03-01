from luma.classifier.discriminant import QDAClassifier
from luma.preprocessing.scaler import StandardScaler
from luma.model_selection.split import TrainTestSplit
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

qda = QDAClassifier()
qda.fit(X_train_pca, y_train)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

X_trans = np.concatenate((X_train_pca, X_test_pca))
y_trans = np.concatenate((y_train, y_test))

dec = DecisionRegion(qda, X_trans, y_trans)
dec.plot(ax=ax1)

conf = ConfusionMatrix(y_trans, qda.predict(X_trans))
conf.plot(ax=ax2, show=True)
