import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Scalar, Vector, KernelUtil
from luma.interface.exception import NotFittedError
from luma.metric.classification import Accuracy


__all__ = (
    'LDAClassifier',
    'QDAClassifier',
    'RDAClassifier',
    'KDAClassifier'
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
        if not self._fitted: raise not NotFittedError(self)
        scores = np.dot(X, self.coef_.T) + self.intercept_
        
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        scores = np.dot(X, self.coef_.T) + self.intercept_
        
        return softmax(scores, axis=1)
    
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
        if not self._fitted: raise not NotFittedError(self)
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
    
    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        log_probs = np.array([self._log_proba_sample(x) for x in X])
        
        return softmax(log_probs, axis=1)
    
    def _log_proba_sample(self, x: Vector) -> Vector:
        log_probs = np.zeros(len(self.classes))
        for i, _ in enumerate(self.classes):
            mvn = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            log_probs[i] = mvn.logpdf(x) + np.log(self.priors[i])
        
        return log_probs
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class RDAClassifier(Estimator, Supervised):
    
    """
    Regularized Discriminant Analysis (RDA) combines the features of 
    Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis 
    (QDA) by introducing regularization to the covariance matrices. It 
    adjusts the covariance estimates with a blend between the pooled 
    covariance (LDA approach) and class-specific covariances (QDA approach), 
    enhancing classification performance. RDA is particularly effective in 
    scenarios with high-dimensional data or when the number of samples is 
    limited. The method employs regularization parameters to balance bias 
    and variance, aiming to optimize model accuracy.
    
    Parameters
    ----------
    `alpha` : Balancing parameter between the class-specific covariance matrices
    (0 for LDA-like, 1 for QDA-like approach)
    `gamma` : Shrinkage applied to the covariance matrices
    (0 for large shrinkage, i.e. max regularization)
    
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 gamma: float = 0.5) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.classes = None
        self.means = None
        self.priors = None
        self.covs = None
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'RDAClassifier':
        m, n = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.means = np.zeros((n_classes, n))
        self.pooled_cov = np.zeros((n, n))
        self.covs = np.zeros((n_classes, n, n))
        self.priors = np.zeros(n_classes)
        
        for i, cl in enumerate(self.classes):
            X_cls = X[y == cl]
            self.means[i] = np.mean(X_cls, axis=0)
            self.priors[i] = X_cls.shape[0] / m
            class_cov = np.cov(X_cls, rowvar=False)
            self.pooled_cov += class_cov * (X_cls.shape[0] - 1)
        
        self.pooled_cov /= (m - n_classes)
        
        for i in range(n_classes):
            X_cls = X[y == self.classes[i]]
            class_cov = np.cov(X_cls, rowvar=False)
            self.covs[i] = self.alpha * class_cov
            self.covs[i] += (1 - self.alpha) *  self.pooled_cov
            
            self.covs[i] = self.gamma * self.covs[i]
            self.covs[i] += (1 - self.gamma) * np.eye(n) * np.trace(self.covs[i])
            self.covs[i] /= n
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise not NotFittedError(self)
        return Vector([self._predict_sample(x) for x in X])

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
    
    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        log_probs = np.array([self._log_proba_sample(x) for x in X])
        
        return softmax(log_probs, axis=1)
    
    def _log_proba_sample(self, x: Vector) -> Vector:
        log_probs = np.zeros(len(self.classes))
        for i, _ in enumerate(self.classes):
            mvn = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            log_probs[i] = mvn.logpdf(x) + np.log(self.priors[i])
        
        return log_probs
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class KDAClassifier(Estimator, Supervised):
    
    """
    A Kernel Discriminant Analysis (KDA) classifier is an extension of 
    traditional discriminant analysis methods that incorporates kernel 
    techniques to perform classification tasks in a transformed, 
    high-dimensional feature space. By applying a kernel function, it 
    enables the classifier to find nonlinear boundaries between classes, 
    enhancing its ability to handle complex patterns that are not linearly 
    separable. The KDA classifier operates by projecting input data into 
    a space where classes are maximally distant from each other, using 
    the calculated projections to classify new instances based on their 
    proximity to class centroids.
    
    Parameters
    ----------
    `deg` : Polynomial degree of `poly` kernel
    `gamma` : Shape parameter of `rbf`, `sigmoid`, `laplacian`
    `coef` : Additional coefficient of `poly`, `sigmoid`
    `kernel` : Type of kernel functions
    
    Notes
    -----
    * To use KDA for dimensionality reduction, refer to 
        `luma.reduction.linear.KDA`
    
    """
    
    def __init__(self, 
                 deg: int = 2,
                 gamma: float = 1.0,
                 alpha: float = 1.0,
                 coef: int = 0.0,
                 kernel: KernelUtil.func_type = 'rbf') -> None:
        self.deg = deg
        self.alpha = alpha
        self.gamma = gamma
        self.coef = coef
        self.kernel = kernel
        self.X_ = None
        self._fitted = False
        
        self.kernel_params = {
            'deg': self.deg,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'coef': self.coef
        }
    
    def fit(self, X: Matrix, y: Vector) -> 'KDAClassifier':
        m, _ = X.shape
        self._X = X
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        self.ku_ = KernelUtil(self.kernel, **self.kernel_params)
        self.class_means = np.zeros((self.n_classes, m))
        
        self.K = self.ku_.kernel_func(X)
        for i, cl in enumerate(self.classes):
            self.class_means[i] = np.mean(self.K[y == cl], axis=0)
        
        self._fitted = False
        return self
    
    def _project(self, X: Matrix) -> Vector:
        K = self.ku_.kernel_func(X, self._X)
        proj = np.dot(K, self.class_means.T)
        return proj
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        proj = self._project(X)
        
        return self.classes[np.argmax(proj, axis=1)]
    
    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        proj = self._project(X)
        
        return softmax(proj, axis=1)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

