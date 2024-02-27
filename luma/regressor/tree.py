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
    `min_impurity_decrease` : Minimum decrement of impurity for a split
    `max_leaf_nodes` : Maximum amount of leaf nodes
    `random_state` : The randomness seed of the estimator
    
    """
    
    def __init__(self, 
                 max_depth: int = 10, 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1, 
                 max_features: int = None, 
                 min_impurity_decrease: float = 0.01,
                 max_leaf_nodes: int = None, 
                 random_state: int = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.root = None
        self._fitted = False
        
        np.random.seed(random_state)
    
    def fit(self, X: Matrix, y: Vector) -> 'DecisionTreeRegressor':
        self.root = self._build_tree(X, y, 0)
        self._fitted = True
        return self

    def _build_tree(self, X: Matrix, y: Vector, depth: int = 0) -> DecisionTreeNode:
        if len(y) <= self.min_samples_split or depth == self.max_depth:
            return DecisionTreeNode(value=np.mean(y))
        
        split_idx, split_threshold = self._best_split(X, y)
        if split_idx is None:
            return DecisionTreeNode(value=np.mean(y))
        
        left = X[:, split_idx] <= split_threshold
        right = X[:, split_idx] > split_threshold
        left_X, right_X = X[left], X[right]
        left_y, right_y = y[left], y[right]
        
        if len(left_y) < self.min_samples_leaf or \
            len(right_y) < self.min_samples_leaf:
            return DecisionTreeNode(value=np.mean(y))
        
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)
        
        return DecisionTreeNode(feature_index=split_idx, 
                                threshold=split_threshold,
                                left=left_subtree, 
                                right=right_subtree)

    def _calculate_var_reduction(self, 
                                      y: Vector, 
                                      left_y: Vector, 
                                      right_y: Vector) -> float:
        total_var = np.var(y)
        weight_left = len(left_y) / len(y)
        weight_right = len(right_y) / len(y)
        
        var_reduction = total_var
        var_reduction -= weight_left * np.var(left_y)
        var_reduction -= weight_right * np.var(right_y)
        
        return var_reduction

    def _best_split(self, X: Matrix, y: Vector) -> Tuple[int, float]:
        best_reduction = -np.inf
        split_idx, split_threshold = None, None
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left = X[:, feature_index] <= threshold
                right = X[:, feature_index] > threshold
                
                if np.sum(left) < self.min_samples_split or \
                    np.sum(right) < self.min_samples_split:
                    continue
                
                reduction = self._calculate_var_reduction(y, y[left], y[right])
                if reduction > best_reduction and \
                    reduction >= self.min_impurity_decrease:
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

