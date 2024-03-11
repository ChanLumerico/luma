from typing import Tuple
import numpy as np

from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import DecisionTreeNode
from luma.metric.regression import MeanSquaredError


__all__ = (
    'DecisionTreeRegressor',
)


class DecisionTreeRegressor(Estimator, Supervised):
    
    """
    A Decision Tree Regressor is a machine learning algorithm used for 
    predicting continuous values, unlike its counterpart, the Decision 
    Tree Classifier, which is used for predicting categorical outcomes. 
    The regressor works by splitting the data into distinct regions based 
    on feature values. At each node of the tree, it chooses the split 
    that minimizes the variance (or another specified criterion) of the 
    target variable within the regions created by the split.
    
    Parameters
    ----------
    `max_depth` : Maximum depth of the tree
    `criterion` : Function used to measure the quality of a split
    `min_samples_split` : Minimum samples required to split a node
    `min_samples_leaf` : Minimum samples required to be at a leaf node
    `max_features` : Number of features to consider
    `min_variance_decrease` : Minimum decrement of variance for a split
    `max_leaf_nodes` : Maximum amount of leaf nodes
    `random_state` : The randomness seed of the estimator
    
    """
    
    def __init__(self, 
                 max_depth: int = 10, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1, 
                 max_features: int = None, 
                 min_variance_decrease: float = 0.0,
                 max_leaf_nodes: int = None,
                 random_state: int = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_variance_decrease = min_variance_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.root = None
        self._fitted = False
    
    def fit(self, 
            X: Matrix, 
            y: Vector, 
            sample_weights: Vector = None) -> 'DecisionTreeRegressor':
        if sample_weights is None: sample_weights = np.ones(len(y))
        sample_weights = Vector(sample_weights)
        np.random.seed(self.random_state)
        
        self.root = self._build_tree(X, y, 0, sample_weights)
        self._fitted = True
        return self
    
    def _stopping_criteria(self, y: Vector, depth: int) -> bool:
        if len(y) <= self.min_samples_split: return True
        if depth == self.max_depth: return True
        return False
    
    def _build_tree(self, 
                    X: Matrix, 
                    y: Vector, 
                    depth: int, 
                    sample_weights: Vector) -> DecisionTreeNode:
        if self._stopping_criteria(y, depth):
            return DecisionTreeNode(value=np.average(y, weights=sample_weights))
        
        split_idx, split_threshold = self._best_split(X, y, sample_weights)
        if split_idx is None:
            return DecisionTreeNode(value=np.average(y, weights=sample_weights))
        
        left = X[:, split_idx] <= split_threshold
        right = X[:, split_idx] > split_threshold
        
        X_left, X_right = X[left], X[right]
        y_left, y_right = y[left], y[right]
        
        weights_left, weights_right = sample_weights[left], sample_weights[right]
        
        if np.sum(weights_left) < self.min_samples_leaf or \
            np.sum(weights_right) < self.min_samples_leaf:
            return DecisionTreeNode(value=np.average(y, weights=sample_weights))
        
        left_subtree = self._build_tree(X_left, y_left, depth + 1, weights_left)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, weights_right)
        
        return DecisionTreeNode(feature_index=split_idx, 
                                threshold=split_threshold, 
                                left=left_subtree, 
                                right=right_subtree)

    def _calculate_var_reduction(self, 
                                 y: Vector, 
                                 y_left: Vector, 
                                 y_right: Vector, 
                                 sample_weights: Vector, 
                                 weights_left: Vector, 
                                 weights_right: Vector) -> float:
        total_weight = np.sum(sample_weights)
        y_avg = np.average(y, weights=sample_weights)
        weighted_total_var = np.average((y - y_avg) ** 2, weights=sample_weights)
        
        y_left_avg = np.average(y_left, weights=weights_left)
        y_right_avg = np.average(y_right, weights=weights_right)
        
        left_var = np.average((y_left - y_left_avg) ** 2, weights=weights_left)
        right_var = np.average((y_right - y_right_avg) ** 2, weights=weights_right)
        
        weighted_var_red = weighted_total_var
        weighted_var_red -= (np.sum(weights_left) / total_weight) * left_var
        weighted_var_red -= (np.sum(weights_right) / total_weight) * right_var
        
        return weighted_var_red

    def _best_split(self, 
                    X: Matrix, 
                    y: Vector, 
                    sample_weights: Vector) -> Tuple[int, float]:
        best_reduction = -np.inf
        split_idx, split_threshold = None, None
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left = X[:, feature_index] <= threshold
                right = X[:, feature_index] > threshold
                
                if np.sum(sample_weights[left]) < self.min_samples_split or \
                    np.sum(sample_weights[right]) < self.min_samples_split:
                    continue
                
                reduction = self._calculate_var_reduction(
                    y, y[left], y[right], sample_weights, 
                    sample_weights[left], sample_weights[right]
                )
                
                if reduction > best_reduction and \
                    reduction >= self.min_variance_decrease:
                    best_reduction = reduction
                    split_idx, split_threshold = feature_index, threshold
        
        return split_idx, split_threshold
    
    def _predict_single(self, node: DecisionTreeNode, x: Vector) -> float:
        if node.value is not None: 
            return node.value
        if x[node.feature_index] <= node.threshold: 
            return self._predict_single(node.left, x)
        
        return self._predict_single(node.right, x)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        return np.array([self._predict_single(self.root, x) for x in X])

    def score(self, X: Matrix, y: Vector, 
              metric: Evaluator = MeanSquaredError) -> float:
        predictions = self.predict(X)
        return metric.score(y, predictions)

