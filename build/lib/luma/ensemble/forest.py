from typing import *
from scipy.stats import mode
import numpy as np

from luma.classifier.tree import DecisionTreeClassifier
from luma.regressor.tree import DecisionTreeRegressor
from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError


__all__ = ['RandomForestClassifier', 'RandomForestRegressor']


class RandomForestClassifier(Estimator, Supervised):
    
    """
    A Random Forest Classifier is an ensemble learning method in machine learning, 
    widely used for both classification and regression tasks. It operates by 
    constructing a multitude of decision trees during the training phase and 
    outputs the mode of the classes for classification.
    
    Parameters
    ----------
    ``n_trees`` : Number of trees in the forest \n
    ``max_depth`` :  Maximum depth of each trees \n
    ``min_samples_split`` : Minimum number of samples required to split a node \n
    ``bootstrap`` : Whether to bootstrap the samples of dataset \n
    ``bootstrap_feature`` : Whether to bootstrap the features of each data \n
    ``n_features`` : Number of features to be sampled when bootstrapping features
    
    """
    
    def __init__(self, 
                 n_trees: int = 10, 
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 bootstrap: bool = True,
                 bootstrap_feature: bool = False,
                 n_features: int | Literal['auto'] = 'auto',
                 verbose: bool = False) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.verbose = verbose
        self.trees = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
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
                            min_samples_split=self.min_samples_split)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        preds = np.array([tree.predict(X) for tree in self.trees])
        majority, _ = mode(preds, axis=0)
        
        return majority.flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   n_trees: int = None,
                   max_depth: int = None,
                   min_samples_split: int = None,
                   n_features: int | str = None,
                   bootstrap: bool = None,
                   bootstrap_feature: bool = None) -> None:
        if n_trees is not None: self.n_trees = int(n_trees)
        if max_depth is not None: self.max_depth = int(max_depth)
        if bootstrap is not None: self.bootstrap = bool(bootstrap)
        
        if bootstrap_feature is not None: 
            self.bootstrap_feature = bool(bootstrap_feature)
        if min_samples_split is not None: 
            self.min_samples_split = int(min_samples_split)
        if n_features is not None:
            if isinstance(n_features, int):
                self.n_features = int(n_features)
            elif isinstance(n_features, str):
                self.n_features = str(n_features)


class RandomForestRegressor(Estimator, Supervised):
    
    """
    A Random Forest Regressor is an ensemble learning algorithm for regression 
    tasks. It builds multiple decision trees during training, each on a random 
    subset of features and bootstrapped data. The final prediction is the average 
    of individual tree predictions, providing a robust and accurate result.
    
    Parameters
    ----------
    ``n_trees`` : Number of trees in the forest \n
    ``max_depth`` :  Maximum depth of each trees \n
    ``min_samples_split`` : Minimum number of samples required to split a node \n
    ``bootstrap`` : Whether to bootstrap the samples of dataset \n
    ``bootstrap_feature`` : Whether to bootstrap the features of each data \n
    ``n_features`` : Number of features to be sampled when bootstrapping features
    
    """
    
    def __init__(self, 
                 n_trees: int = 10, 
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 bootstrap: bool = True,
                 bootstrap_feature: bool = False,
                 n_features: int | Literal['auto'] = 'auto',
                 verbose: bool = False) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.verbose = verbose
        self.trees = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
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
                            min_samples_split=self.min_samples_split)
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        preds = np.array([tree.predict(X) for tree in self.trees])
        average_predictions = np.mean(preds, axis=0)
        
        return average_predictions
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)
    
    def set_params(self,
                   n_trees: int = None,
                   max_depth: int = None,
                   min_samples_split: int = None,
                   n_features: int | str = None,
                   bootstrap: bool = None,
                   bootstrap_feature: bool = None) -> None:
        if n_trees is not None: self.n_trees = int(n_trees)
        if max_depth is not None: self.max_depth = int(max_depth)
        if bootstrap is not None: self.bootstrap = bool(bootstrap)
        
        if bootstrap_feature is not None: 
            self.bootstrap_feature = bool(bootstrap_feature)
        if min_samples_split is not None: 
            self.min_samples_split = int(min_samples_split)
        if n_features is not None:
            if isinstance(n_features, int):
                self.n_features = int(n_features)
            elif isinstance(n_features, str):
                self.n_features = str(n_features)

