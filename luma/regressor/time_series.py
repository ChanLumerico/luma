from typing import Tuple
import numpy as np

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.interface.util import Matrix, Vector
from luma.metric.regression import MeanSquaredError


__all__ = (
    'AutoRegressor'
)


class AutoRegressor(Estimator, Estimator.TimeSeries, Supervised):
    
    """
    Autoregression (AR) models predict future values of a time series 
    based on its own past values. These models assume that current 
    observations are linearly dependent on previous observations plus a 
    random error term. The number of past values used for prediction, 
    known as the lag order, determines the model's complexity. AR models 
    are widely used for time series forecasting in finance, weather, and 
    many other fields. They provide a simple yet powerful method for 
    understanding and predicting temporal data dynamics.
    
    Parameters
    ----------
    `p` : Number of lagged observations of the predictor variable
    
    Methods
    -------
    For in-sample predictions and future predictions(forecasts):
    
        - `step` equal to 0: In-sample prediction (default)
        - `step` bigger than 0: Future prediction

        ```py
        def predict(self, y: Vector, step: int = 0) -> Vector
        ```
    """
    
    def __init__(self, p: int = 1):
        self.p = p
        self.coefficients = None
        self._fitted = False
    
    def fit(self, y: Vector) -> 'AutoRegressor':
        X, y_lag = self._prepare_data(y)
        X = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y_lag
        
        self._fitted = True
        return self
    
    def _prepare_data(self, y: Vector) -> Tuple[Vector, Vector]:
        X = Matrix([y[i - self.p:i] for i in range(self.p, len(y))])
        y_lag = y[self.p:]
        return X, y_lag

    def predict(self, y: Vector, step: int = 0) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        if step == 0: preds = self._in_sample(y)
        elif step > 0: preds = self._forecast(y, step=step)
        else: raise UnsupportedParameterError(step)
        
        return preds
    
    def _in_sample(self, y: Vector) -> Vector:
        if len(y) < self.p:
            raise ValueError("Series length must be greater than p.")
        preds = []
        for i in range(self.p, len(y)):
            X_last = y[i-self.p:i]
            preds.append(self._predict_next(X_last))
        
        return Vector(preds)

    def _forecast(self, y: Vector, step: int) -> Vector:
        preds = [self._predict_next(y[-self.p:])]
        for _ in range(1, step):
            y = np.append(y, preds[-1])
            preds.append(self._predict_next(y[-self.p:]))
        
        return Vector(preds)

    def _predict_next(self, X_last):
        X_last = np.insert(X_last, 0, 1)
        prediction = X_last @ self.coefficients
        
        return prediction
    
    def score(self, y: Matrix, metric: Evaluator = MeanSquaredError) -> float:
        y_pred = self.predict(y)
        return metric.score(y_true=y[self.p:], y_pred=y_pred)

