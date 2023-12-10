from typing import Tuple
import numpy as np

from luma.interface.exception import NotFittedError
from luma.interface.super import Estimator, Evaluator, Supervised
from luma.interface.util import TreeNode
from luma.metric.regression import MeanSquaredError


__all__ = ['DecisionTreeRegressor']


class DecisionTreeRegressor(Estimator, Supervised):
    
    """
    A Decision Tree Regressor is a supervised machine learning algorithm 
    used for regression tasks. It works by recursively partitioning 
    the input data into subsets based on the values of different features. 
    At each node of the tree, a decision is made based on a specific feature, 
    and the data is split into branches accordingly.

    Parameters
    ----------
    ``max_depth`` : Maximum depth of the tree \n
    ``min_samples_split`` : Minimum number of samples required to split a node

    """
    
    def __init__(self,
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 verbose: bool = False) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose = verbose
        self.root = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        _, self.n_features = X.shape
        self.root = self._grow_tree(X, y, depth=0)

        self._fitted = True
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, 
                   depth: int = 0) -> TreeNode:
        _, n = X.shape
        if self._stopping_criteria(X, depth):
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        feature_indices = np.arange(n)
        best_feature, best_thresh = self._best_criteria(X, y, feature_indices)
        left_indices, right_indices = self._split(X[:, best_feature], best_thresh)

        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        if self.verbose:
            print(f'[DecisionTree] Depth {depth} reached -', end=' ')
            print(f'best_feature: {best_feature}, best_threshold: {best_thresh}')

        return TreeNode(best_feature, best_thresh, left, right)

    def _stopping_criteria(self, X: np.ndarray, depth: int) -> bool:
        if depth >= self.max_depth:
            return True
        if X.shape[0] < self.min_samples_split:
            return True
        return False

    def _best_criteria(self, X: np.ndarray, y: np.ndarray, 
                       indices: np.ndarray) -> Tuple[int, float]:
        best_var = np.var(y)
        split_idx, split_thresh = 0, 0.0
        for idx in indices:
            X_col = X[:, idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                left_indices, right_indices = self._split(X_col, threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                var_left = np.var(y[left_indices])
                var_right = np.var(y[right_indices])

                weighted_var = (len(left_indices) / len(y)) * var_left
                weighted_var += (len(right_indices) / len(y)) * var_right

                if weighted_var < best_var:
                    best_var = weighted_var
                    split_idx = idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _split(self, X_col: np.ndarray, thresh: float) -> Tuple[np.ndarray]:
        left_indices = np.argwhere(X_col <= thresh).flatten()
        right_indices = np.argwhere(X_col > thresh).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, x: np.ndarray, node: TreeNode) -> float:
        if node.isLeaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def print_tree(self) -> None:
        if not self._fitted: raise NotFittedError(self)
        self._print_tree_recursive(self.root, depth=0)
    
    def _print_tree_recursive(self, node: TreeNode, depth: int) -> None:
        if node.isLeaf: print(f"{(' ' * 4 + '|') * depth}--- Leaf: {node.value}")
        else:
            print(f"{(' ' * 4 + '|') * depth}---", 
                  f"Feature {node.feature} <= {node.threshold}")
            self._print_tree_recursive(node.left, depth + 1)
            self._print_tree_recursive(node.right, depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted: raise NotFittedError(self)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: Evaluator = MeanSquaredError) -> float:
        X_pred = self.predict(X)
        return metric.compute(y_true=y, y_pred=X_pred)

    def set_params(self,
                   max_depth: int = None,
                   min_samples_split: int = None) -> None:
        if max_depth is not None: self.max_depth = int(max_depth)
        if min_samples_split is not None: 
            self.min_samples_split = int(min_samples_split)

