from typing import Tuple
from collections import Counter
import numpy as np

from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import TreeNode
from luma.metric.classification import Accuracy


__all__ = (
    'DecisionTreeClassifier'
)


class DecisionTreeClassifier(Estimator, Supervised):
    
    """
    A Decision Tree Classifier is a supervised machine learning algorithm 
    used for classification tasks. It works by recursively partitioning 
    the input data into subsets based on the values of different features. 
    At each node of the tree, a decision is made based on a specific feature, 
    and the data is split into branches accordingly.
    
    Parameters
    ----------
    `max_depth` : Maximum depth of tree
    `min_samples_split` : Minimum number of samples required to split a node
    `sample_weights` : Weight of each sample (`1.0` if set to `None`)
    
    """

    def __init__(self, 
                 max_depth: int = 100,
                 min_samples_split: int = 2,
                 sample_weights: Vector = None,
                 verbose: bool = False) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.root = None
        self._fitted = False

    def fit(self, X: Matrix, y: Matrix) -> 'DecisionTreeClassifier':
        self.n_samples, self.n_features = X.shape
        
        if self.sample_weights is None: 
            self.sample_weights = np.ones(self.n_samples)
        self.root = self._grow_tree(X, y, self.sample_weights)
        
        self._fitted = True
        return self

    def _grow_tree(self, X: Matrix, y: Matrix, w: Vector, 
                   depth: int = 0) -> TreeNode:
        _, n = X.shape
        if self._stopping_criteria(X, y, depth):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_indices = np.random.choice(n, self.n_features, replace=False)
        
        best_feature, best_thresh = self._best_criteria(X, y, w, feature_indices)
        left_indices, right_indices = self._split(X[:, best_feature], best_thresh)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)
        
        left = self._grow_tree(X[left_indices], y[left_indices], 
                               w[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], 
                                w[right_indices], depth + 1)
        
        if self.verbose:
            print(f'[DecisionTree] Depth {depth} reached -', end=' ')
            print(f'best_feature: {best_feature}, best_threshold: {best_thresh}')
        
        return TreeNode(best_feature, best_thresh, left, right)
    
    def _stopping_criteria(self, X: Matrix, y: Matrix, 
                           depth: int) -> bool:
        if depth >= self.max_depth: return True
        if len(np.unique(y)) == 1: return True
        if X.shape[0] < self.min_samples_split: return True
        return False

    def _best_criteria(self, X: Matrix, y: Matrix, w: Vector,
                       indices: Matrix) -> Tuple[int, float]:
        best_gain = -1
        split_idx, split_thresh = 0, 0.0
        for idx in indices:
            X_col = X[:, idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(X_col, y, w, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, X_col: Matrix, y: Matrix, w: Vector, 
                          thresh: float) -> float:
        parent_entropy = self._entropy(y, w)
        left_indices, right_indices = self._split(X_col, thresh)
        if not len(left_indices) or not len(right_indices): return 0

        n = np.sum(w)
        n_l, n_r = np.sum(w[left_indices]), np.sum(w[right_indices])
        e_l = self._entropy(y[left_indices], w[left_indices])
        e_r = self._entropy(y[right_indices], w[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_col: Matrix, thresh: float) -> Tuple[Matrix]:
        left_indices = np.argwhere(X_col <= thresh).flatten()
        right_indices = np.argwhere(X_col > thresh).flatten()
        return left_indices, right_indices

    def _traverse_tree(self, x: Matrix, node: TreeNode) -> float | None:
        if node.isLeaf: return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y: Matrix) -> int:
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _entropy(self, y: Matrix, w: Vector) -> float:
        unique_labels, _ = np.unique(y, return_counts=True)
        ps = Matrix([np.sum(w[y == label]) for label in unique_labels]) / np.sum(w)
        
        return -np.sum([p * np.log2(p) for p in ps if p])
    
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

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted: raise NotFittedError(self)
        return Matrix([self._traverse_tree(x, self.root) for x in X])

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

