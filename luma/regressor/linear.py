import numpy as np

from luma.interface.util import Matrix
from luma.interface.exception import NotFittedError
from luma.interface.super import Estimator, Evaluator, Supervised
from luma.metric.regression import MeanSquaredError


__all__ = (
    'LinearRegressor', 
    'RidgeRegressor', 
    'LassoRegressor', 
    'ElasticNetRegressor'
)


class RidgeRegressor(Estimator, Supervised):
    
    """
    Ridge regression is a linear regression technique used to 
    prevent overfitting in predictive models. 
    It adds a penalty term called "L2 regularization" to help 
    reduce the complexity of the model and prevent it from 
    fitting noise in the data.
    
    Parameters
    ----------
    `alpha` : L2-regularization strength
    
    """
    
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'RidgeRegressor':
        X = np.column_stack((np.ones(X.shape[0]), X))
        identity_matrix = np.identity(X.shape[1])
        self.coefficients = np.linalg.inv(X.T.dot(X) + self.alpha * identity_matrix)
        self.coefficients = self.coefficients.dot(X.T).dot(y)
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coefficients)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def set_params(self, alpha: float = None) -> None:
        if alpha is not None: self.alpha = float(alpha)


class LassoRegressor(Estimator, Supervised):
    
    """
    Lasso regression is a linear regression technique used for 
    feature selection and regularization. It adds a penalty term, 
    called "L1 regularization," to the standard linear regression 
    objective function. Lasso encourages some of the model's 
    coefficients to become exactly zero, effectively eliminating 
    certain features from the model.
    
    Parameters
    ----------
    `alpha` : L1-regularization strength
    `max_iter` : Number of iteration
    `learning_rate` : Step size of the gradient descent update
    
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 max_iter: int = 100, 
                 learning_rate: float = 0.01, 
                 verbose: bool = False) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._fitted = False

    def _soft_threshold(self, x: Matrix, threshold: float) -> Matrix:
        return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

    def fit(self, X: Matrix, y: Matrix) -> 'LassoRegressor':
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coefficients = np.zeros(X.shape[1])
        
        for i in range(self.max_iter):
            coefficients_prev = self.coefficients.copy()
            y_pred = X.dot(self.coefficients)
            gradient = -(X.T.dot(y - y_pred)) * self.learning_rate
            self.coefficients = self.coefficients - (1.0 / X.shape[0]) * gradient
            self.coefficients = self._soft_threshold(self.coefficients, self.alpha / X.shape[0])
            
            if self.verbose and i % 10 == 0:
                print(f'[LassoReg] iteration: {i}/{self.max_iter}', end='')
                print(f' - delta-coeff norm: {np.linalg.norm(self.coefficients - coefficients_prev)}')
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X.dot(self.coefficients)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   alpha: float = None, 
                   max_iter: int = None,
                   learning_rate: float = None) -> None:
        if alpha is not None: self.alpha = float(alpha)
        if max_iter is not None: self.max_iter = int(max_iter)
        if learning_rate is not None: self.learning_rate = float(learning_rate)


class ElasticNetRegressor(Estimator, Supervised):
    
    """
    Elastic-Net regression is a linear regression technique 
    that combines both L1 (Lasso) and L2 (Ridge) regularization methods. 
    It adds a combination of L1 and L2 penalty terms to the 
    standard linear regression objective function.
    
    Parameters
    ----------
    `alpha` : Regularization strength
    `rho` : Balancing parameter between `l1` and `l2`
    `max_iter` : Number of iteration
    `learning_rate` : Step size of the gradient descent update
    
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 rho: float = 0.5, 
                 max_iter: int = 100, 
                 learning_rate: float = 0.01,
                 verbose: bool = False) -> None:
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.coef_ = None
        self._fitted = False
        
    def _soft_threshold(self, x: Matrix, alpha: float) -> Matrix:
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

    def fit(self, X: Matrix, y: Matrix) -> 'ElasticNetRegressor':
        N, p = X.shape
        self.coef_ = np.zeros(p)

        for i in range(self.max_iter):
            y_pred = X.dot(self.coef_)
            gradient = -(1 / N) * X.T.dot(y - y_pred)
            lefthand = self.coef_ - self.learning_rate * gradient
            righthand = self.alpha * self.rho
            
            self.coef_ = self._soft_threshold(lefthand, righthand)
            self.coef_ /= 1 + self.alpha * (1 - self.rho)
            
            if self.verbose and i % 10 == 0: 
                print(f'[ElasticReg] iteration: {i}/{self.max_iter}')
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        if self.coef_ is None: raise ValueError()
        return X.dot(self.coef_)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   alpha: float = None,
                   rho: float = None,
                   max_iter: int = None,
                   learning_rate: float = None) -> None:
        if alpha is not None: self.alpha = float(alpha)
        if rho is not None: self.rho = float(rho)
        if max_iter is not None: self.max_iter = int(max_iter)
        if learning_rate is not None: self.learning_rate = float(learning_rate)


class LinearRegressor(Estimator, Supervised):
    
    """
    An Ordinary Least Squares (OLS) Linear Regressor is a statistical method 
    used in linear regression analysis. It estimates the coefficients of the 
    linear equation, minimizing the sum of the squared differences between 
    observed and predicted values. This results in a line of best fit through 
    the data points in multidimensional space. OLS is widely used for its 
    simplicity and efficiency in modeling linear relationships.
    """
    
    def __init__(self):
        self.coefficients = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'LinearRegressor':
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        X_T = np.transpose(X)
        self.coefficients = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
        
        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        return X.dot(self.coefficients)
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def set_params(self) -> None: ...

