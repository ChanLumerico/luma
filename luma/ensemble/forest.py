from typing import Literal
from scipy.stats import mode
import numpy as np

from luma.interface.util import Matrix, Vector
from luma.classifier.tree import DecisionTreeClassifier
from luma.regressor.tree import DecisionTreeRegressor
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError


__all__ = (
    'RandomForestClassifier', 
    'RandomForestRegressor'
)


class RandomForestClassifier(Estimator, Estimator.Meta, Supervised):
    
    """
    A Random Forest Classifier is an ensemble learning method in machine learning, 
    widely used for both classification and regression tasks. It operates by 
    constructing a multitude of decision trees during the training phase and 
    outputs the mode of the classes for classification.
    
    Parameters
    ----------
    `n_trees` : Number of trees in the forest
    `max_depth` :  Maximum depth of each trees
    `min_samples_split` : Minimum number of samples required to split a node
    `bootstrap` : Whether to bootstrap the samples of dataset
    `bootstrap_feature` : Whether to bootstrap the features of each data
    `n_features` : Number of features to be sampled when bootstrapping features
    `sample_weights` : Weight of each sample (`1.0` if set to `None`)
    
    """
    
    def __init__(self, 
                 n_trees: int = 10, 
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 bootstrap: bool = True,
                 bootstrap_feature: bool = False,
                 n_features: int | Literal['auto'] = 'auto',
                 sample_weights: Vector = None,
                 verbose: bool = False) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.trees = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'RandomForestClassifier':
        m, n = X.shape
        if self.n_features == 'auto': 
            self.n_features = int(n ** 0.5)
        else:
            if isinstance(self.n_features, str):
                raise UnsupportedParameterError(self.n_features)
        
        self.trees = [DecisionTreeClassifier() for _ in range(self.n_trees)]
        for i, tree in enumerate(self.trees, start=1):
            X_sample, y_sample = X, y
            if self.bootstrap:
                bootstrap_data = np.random.choice(m, m, replace=True)
                X_sample = X_sample[bootstrap_data]
                y_sample = y_sample[bootstrap_data]
            if self.bootstrap_feature:
                bootstrap_feature = np.random.choice(n, self.n_features, replace=True)
                X_sample = X_sample[:, bootstrap_feature]
            
            tree.set_params(max_depth=self.max_depth, 
                            min_samples_split=self.min_samples_split,
                            sample_weights=self.sample_weights)
            
            tree.fit(X_sample, y_sample)
            
            if self.verbose:
                print(f'[RandomForest] Finished fitting tree {i}/{self.n_trees}')
        
        self._fitted = True
        return self
    
    def print_forest(self) -> None:
        if not self._fitted: raise NotFittedError(self)
        for i, tree in enumerate(self.trees, start=1):
            print(f'[Tree {i}/{self.n_trees}]')
            tree.print_tree()

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        preds = Matrix([tree.predict(X) for tree in self.trees])
        majority, _ = mode(preds, axis=0)
        
        return majority.flatten()
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)


class RandomForestRegressor(Estimator, Estimator.Meta, Supervised):
    
    """
    A Random Forest Regressor is an ensemble learning algorithm for regression 
    tasks. It builds multiple decision trees during training, each on a random 
    subset of features and bootstrapped data. The final prediction is the average 
    of individual tree predictions, providing a robust and accurate result.
    
    Parameters
    ----------
    `n_trees` : Number of trees in the forest
    `max_depth` :  Maximum depth of each trees
    `min_samples_split` : Minimum number of samples required to split a node
    `bootstrap` : Whether to bootstrap the samples of dataset
    `bootstrap_feature` : Whether to bootstrap the features of each data
    `n_features` : Number of features to be sampled when bootstrapping features
    `sample_weights` : Weight of each sample (`1.0` if set to `None`)
    
    """
    
    def __init__(self, 
                 n_trees: int = 10, 
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 bootstrap: bool = True,
                 bootstrap_feature: bool = False,
                 n_features: int | Literal['auto'] = 'auto',
                 sample_weights: Vector = None,
                 verbose: bool = False) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.trees = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'RandomForestRegressor':
        m, n = X.shape
        if self.n_features == 'auto': 
            self.n_features = int(n ** 0.5)
        else:
            if isinstance(self.n_features, str):
                raise UnsupportedParameterError(self.n_features)
        
        self.trees = [DecisionTreeRegressor() for _ in range(self.n_trees)]
        for i, tree in enumerate(self.trees, start=1):
            X_sample, y_sample = X, y
            if self.bootstrap:
                bootstrap_data = np.random.choice(m, m, replace=True)
                X_sample = X_sample[bootstrap_data]
                y_sample = y_sample[bootstrap_data]
            if self.bootstrap_feature:
                bootstrap_feature = np.random.choice(n, self.n_features, replace=True)
                X_sample = X_sample[:, bootstrap_feature]
            
            tree.set_params(max_depth=self.max_depth, 
                            min_samples_split=self.min_samples_split,
                            sample_weights=self.sample_weights)

            tree.fit(X_sample, y_sample)
            
            if self.verbose:
                print(f'[RandomForest] Finished fitting tree {i}/{self.n_trees}')
        
        self._fitted = True
        return self
    
    def print_forest(self) -> None:
        if not self._fitted: raise NotFittedError(self)
        for i, tree in enumerate(self.trees, start=1):
            print(f'[Tree {i}/{self.n_trees}]')
            tree.print_tree()

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        preds = Matrix([tree.predict(X) for tree in self.trees])
        average_predictions = np.mean(preds, axis=0)
        
        return average_predictions
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

