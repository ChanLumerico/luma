import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Scalar, Vector
from luma.metric.classification import Accuracy


__all__ = (
    'LDAClassifier',
    'QDAClassifier'
)


class LDAClassifier(Estimator, Supervised):
    
    """
    Linear Discriminant Analysis (LDA) is a statistical technique used for 
    dimensionality reduction and classification. It projects data onto a 
    lower-dimensional space with the goal of maximizing class separability. 
    LDA assumes that different classes generate data based on Gaussian 
    distributions with the same covariance matrix but different mean vectors. 
    It finds linear combinations of features that best separate the classes, 
    facilitating classification or visualization of complex data.
    
    Notes
    -----
    * To use LDA for dimensionality reduction, refer to 
        `luma.reduction.linear.LDA`
    """
    
    def __init__(self) -> None:
        self.means = None,
        self.priors = None,
        self.covs = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'LDAClassifier':
        m, n = X.shape
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        
        self.means = np.zeros((n_classes, n))
        self.priors = np.zeros(n_classes)
        self.covs = np.zeros((n, n))

        for i, cl in enumerate(class_labels):
            X_class = X[y == cl]
            self.means[i] = X_class.mean(axis=0)
            self.priors[i] = X_class.shape[0] / m
            self.covs += np.cov(X_class, rowvar=False) * (X_class.shape[0] - 1)
        
        self.covs /= (X.shape[0] - n_classes)
        self.cov_inv = np.linalg.inv(self.covs)

        self.coef_ = np.dot(self.cov_inv, self.means.T).T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means, self.coef_.T))
        self.intercept_ += np.log(self.priors)
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Vector:
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return np.argmax(scores, axis=1)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class QDAClassifier(Estimator, Supervised):
    
    """
    Quadratic Discriminant Analysis (QDA) is a classification technique 
    that assumes data from each class follows a Gaussian distribution 
    with class-specific covariance matrices. It calculates the probability 
    of a given sample belonging to each class based on a quadratic decision 
    boundary determined by these distributions. Unlike Linear Discriminant 
    Analysis (LDA), QDA allows for the modeling of nonlinear relationships 
    due to the class-specific covariances. It is well-suited for datasets 
    where classes exhibit different levels of variance.
    """
    
    def __init__(self) -> None:
        self.means = None
        self.covs = None
        self.priors = None
        self.classes = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'QDAClassifier':
        m, n = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.means = np.zeros((n_classes, n))
        self.covs = np.zeros((n_classes, n, n))
        self.priors = np.zeros(n_classes)
        
        for i, cl in enumerate(self.classes):
            X_cls = X[y == cl]
            self.means[i] = np.mean(X_cls, axis=0)
            self.covs[i] = np.cov(X_cls, rowvar=False)
            self.priors[i] = X_cls.shape[0] / m
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Vector:
        y_pred = [self._predict_sample(x) for x in X]
        return Vector(y_pred)
    
    def _predict_sample(self, x: Vector) -> Scalar:
        discs = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            diff = x - self.means[i]
            cov_inv = np.linalg.inv(self.covs[i])
            cov_det = np.linalg.det(self.covs[i])
            
            disc = np.log(cov_det) + diff.T.dot(cov_inv).dot(diff)
            disc *= -0.5
            disc += np.log(self.priors[i])
            discs[i] = disc
        
        return self.classes[np.argmax(discs)]
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

