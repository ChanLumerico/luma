from typing import Any
import numpy as np

from luma.interface.util import Matrix, Scalar, Vector
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.metric.classification import Accuracy


__all__ = (
    'GaussianNaiveBayes', 
    'BernoulliNaiveBayes',
    'QDA'
)


class GaussianNaiveBayes(Estimator, Supervised):
    
    """
    Gaussian Naive Bayes is a probabilistic classification algorithm. 
    It's based on Bayes' theorem and makes an assumption 
    that the features follow a Gaussian distribution.
    """

    def __init__(self) -> None:
        self._fitted = False
    
    def fit(self, X: Matrix, y: Matrix) -> 'GaussianNaiveBayes':
        self.classes = np.unique(y)
        self.parameters = []
        self.priors = []
        self.X_train = X
        self.y_train = y
        
        shared_cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for c in self.classes:
            X_c = X[y == c]
            cov_matrix = np.diag(X_c.var(axis=0))
            shared_cov_matrix += cov_matrix
            self.parameters.append([X_c.mean(axis=0), cov_matrix])
        
        for params in self.parameters:
            params[1] = shared_cov_matrix / len(self.classes)
        
        self._fitted = True
        return self
        
    def _calculate_likelihood(self, x: Matrix, mean: Matrix, 
                              cov: Matrix) -> float:
        dim = len(mean)
        dev = x - mean
        cov_inv = np.linalg.inv(cov)
        exponent = -0.5 * np.dot(dev, np.dot(cov_inv, dev))
        norm_term = 1 / np.sqrt((2 * np.pi) ** dim * np.linalg.det(cov))
        return norm_term * np.exp(exponent)

    def _calculate_prior(self, c: Any) -> float:
        return np.mean(self.y_train == c)

    def _classify_sample(self, x: Matrix) -> Any:
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = self._calculate_prior(c)
            likelihood = self._calculate_likelihood(x, *self.parameters[i])
            posterior = prior * likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return Matrix([self._classify_sample(x) for x in X])

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        probabilities = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                prior = self._calculate_prior(c)
                likelihood = self._calculate_likelihood(x, *self.parameters[i])
                posterior = prior * likelihood
                posteriors.append(posterior)
            posteriors = Vector(posteriors)
            probabilities.append(posteriors / np.sum(posteriors))          
        return Matrix(probabilities)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class BernoulliNaiveBayes(Estimator, Supervised):
    
    """
    Bernoulli Naive Bayes is a classification algorithm based on 
    Bayes' theorem, which assumes that the features are binary 
    (i.e., they are either present or absent) and that they are 
    conditionally independent given the class label.
    """
    
    def __init__(self) -> None:
        self._fitted = False
    
    def fit(self, X: Matrix, y: Matrix) -> 'BernoulliNaiveBayes':
        self.classes = np.unique(y)
        self.class_probs = np.zeros(len(self.classes))
        self.feature_probs = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            X_cls = X[y == c]
            self.class_probs[i] = len(X_cls) / len(y)
            self.feature_probs[i] = (X_cls.sum(axis=0) + 1) / (len(X_cls) + 2)
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        predictions = []
        for x in X:
            class_scores = np.zeros(len(self.classes))
            for i, _ in enumerate(self.classes):
                class_scores[i] = np.log(self.class_probs[i])
                for j, feature in enumerate(x):
                    if feature == 1: class_scores[i] += np.log(self.feature_probs[i, j])
                    else: class_scores[i] += np.log(1 - self.feature_probs[i, j])

            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)
        
        return Matrix(predictions)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        probas = []
        for x in X:
            class_probs = np.zeros(len(self.classes))
            for i, _ in enumerate(self.classes):
                class_probs[i] = np.log(self.class_probs[i])
                for j, feature in enumerate(x):
                    if feature == 1: class_probs[i] += np.log(self.feature_probs[i, j])
                    else: class_probs[i] += np.log(1 - self.feature_probs[i, j])

            exp_class_probs = np.exp(class_probs - np.max(class_probs))
            probas.append(exp_class_probs / exp_class_probs.sum())
        return Matrix(probas)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class QDA(Estimator, Supervised):
    
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
    
    def fit(self, X: Matrix, y: Vector) -> 'QDA':
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

