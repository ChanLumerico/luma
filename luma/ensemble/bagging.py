import numpy as np

from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.util import Matrix, Vector, Clone
from luma.interface.exception import NotFittedError
from luma.classifier.tree import DecisionTreeClassifier
from luma.metric.classification import Accuracy


__all__ = (
    'BaggingClassifier'
)


class BaggingClassifier(Estimator, Supervised):
    
    """
    A Bagging Classifier is an ensemble learning technique that trains 
    multiple base models, typically decision trees, on random subsets 
    of the original dataset. Each model is trained on a bootstrap sample 
    (random sample with replacement) of the data, potentially using a 
    random subset of features. The final prediction is made by aggregating 
    the predictions of all models, typically through voting for 
    classification or averaging for regression. This approach reduces 
    variance, improves model stability, and can prevent overfitting.
    
    Parameters
    ----------
    `base_estimator` : Base estimator for training multiple models
    (Default `DecisionTreeClassifier`)
    `n_estimators` : Number of base estimators to fit
    `max_samples` : Maximum number of data to sample (`0~1` proportion)
    `max_features` : Maximum number of features to sample (`0~1` proporton)
    `bootstrap` : Whether to bootstrap data samples
    `bootstrap_feature`: Whether to bootstrap features
    `random_state` : Seed for random sampling
    
    Examples
    --------
    >>> bag = BaggingClassifier(base_estimator=AnyEstimator(),
                                n_estimators=100,
                                max_samples=1.0,
                                max_features=1.0,
                                bootstrap=True,
                                bootstrap_feature=False)
    >>> bag.fit(X, y)
    >>> y_pred = bag.predict(X)
    >>> est = bag[i] # Get i-th base estimator from `bag`
    
    """
    
    def __init__(self, 
                 base_estimator: Estimator = DecisionTreeClassifier(),
                 n_estimators: int = 50,
                 max_samples: float | int = 1.0,
                 max_features: float | int = 1.0,
                 bootstrap: bool = True,
                 bootstrap_feature: bool = False,
                 random_state: int = None,
                 verbose: bool = False) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.random_state = random_state
        self.verbose = verbose
        self._fitted = False
    
    def fit(self, X: Matrix, y: Vector) -> 'BaggingClassifier':
        np.random.seed(self.random_state)
        self.estimators_ = []
        
        m, n = X.shape
        if self.max_samples <= 1.0: max_samples = int(m * self.max_samples)
        else: max_samples = self.max_samples
        
        if self.max_features <= 1.0: max_features = int(n * self.max_features)
        else: max_features = self.max_features
        
        for i in range(self.n_estimators):
            s_indices = np.random.choice(m, max_samples, self.bootstrap)
            f_indices = np.random.choice(n, max_features, self.bootstrap_feature)
            
            X_sample = X[s_indices][:, f_indices]
            y_sample = y[s_indices]
            
            estimator = Clone(self.base_estimator).get
            estimator.fit(X_sample, y_sample)
            self.estimators_.append((estimator, f_indices))
            
            if self.verbose:
                print(f'[Bagging] Finished fitting',
                      f'{type(self.base_estimator).__name__}',
                      f'{i + 1}/{self.n_estimators}')
        
        self._fitted = True
        return self
    
    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        
        predictions = []
        for i, (estimator, f_indices) in enumerate(self.estimators_):
            predictions.append(estimator.predict(X[:, f_indices]))
            if self.verbose:
                print(f'[Bagging] Finished prediction of',
                      f'{type(self.base_estimator).__name__}',
                      f'{i + 1}/{self.n_estimators}')
        
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=len(np.unique(x))).argmax(),
            axis=0,
            arr=predictions
        )
        return majority_vote
    
    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self, 
                   base_estimator: Estimator = None,
                   n_estimators: int = None,
                   max_samples: float | int = None,
                   max_features: float | int = None,
                   bootstrap: bool = False,
                   bootstrap_feature: bool = False,
                   random_state: int = None) -> None:
        if base_estimator is not None: self.base_estimator = base_estimator
        if n_estimators is not None: self.n_estimators = int(n_estimators)
        if max_samples is not None: self.max_samples = float(max_samples)
        if max_features is not None: self.max_features = float(max_features)
        if bootstrap is not None: self.bootstrap = bootstrap
        if bootstrap_feature is not None: self.bootstrap_feature = bootstrap_feature
        if random_state is not None: self.random_state = random_state
    
    def __getitem__(self, index: int) -> Estimator:
        return self.estimators_[index]
