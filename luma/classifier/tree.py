from typing import Literal, Tuple
import numpy as np

from luma.interface.util import Matrix, Vector
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import DecisionTreeNode
from luma.metric.classification import Accuracy


__all__ = (
    'DecisionTreeClassifier'
)


class DecisionTreeClassifier(Estimator, Supervised):
    
    """
    A Decision Tree Classifier is a machine learning algorithm that 
    classifies data by making decisions based on asking a series of 
    questions. It splits the dataset into branches at each node, 
    based on feature values, aiming to increase homogeneity in each 
    branch. The process continues recursively until it reaches a 
    leaf node, where the final decision or classification is made. 
    This model is intuitive and can easily handle categorical and 
    numerical data.
    
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
                 criterion: Literal['gini', 'entropy'] ='gini', 
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1, 
                 max_features: int = None, 
                 min_impurity_decrease: float = 0.01,
                 max_leaf_nodes: int = None, 
                 random_state: int = None) -> None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.root = None
        self._fitted = False
        
        np.random.seed(random_state)
    
    def fit(self, X: Matrix, y: Vector) -> 'DecisionTreeClassifier':
        self.root = self._build_tree(X, y)
        self._fitted = True
        return self
    
    def _build_tree(self, X: Matrix, y: Vector, depth: int = 0) -> DecisionTreeNode:
        if self._stopping_criteria(y, depth):
            unique, counts = np.unique(y, return_counts=True)
            value = dict(zip(unique, counts))
            return DecisionTreeNode(value=value)
        
        feature_index, threshold = self._best_split(X, y)
        
        if feature_index is None or \
            (self.max_leaf_nodes is not None and self.max_leaf_nodes <= 1):
            unique, counts = np.unique(y, return_counts=True)
            value = dict(zip(unique, counts))
            return DecisionTreeNode(value=value)
        
        left = np.where(X[:, feature_index] <= threshold)
        right = np.where(X[:, feature_index] > threshold)
        
        if len(y[left]) < self.min_samples_leaf or \
            len(y[right]) < self.min_samples_leaf:
            unique, counts = np.unique(y, return_counts=True)
            value = dict(zip(unique, counts))
            return DecisionTreeNode(value=value)
        
        left_subtree = self._build_tree(X[left], y[left], depth + 1)
        right_subtree = self._build_tree(X[right], y[right], depth + 1)
        
        return DecisionTreeNode(feature_index=feature_index, 
                                threshold=threshold, 
                                left=left_subtree, 
                                right=right_subtree)

    def _calculate_gini(self, y: Vector) -> Vector:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _calculate_entropy(self, y: Vector) -> Vector:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        
        if np.any(probabilities == 0): return 0.0
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_information_gain(self, 
                                    y: Vector, 
                                    y_left: Vector, 
                                    y_right: Vector) -> float:
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        
        if self.criterion == 'gini':
            gain = self._calculate_gini(y)
            gain -= weight_left * self._calculate_gini(y_left)
            gain -= weight_right * self._calculate_gini(y_right)
            
        elif self.criterion == 'entropy':
            gain = self._calculate_entropy(y)
            gain -= weight_left * self._calculate_entropy(y_left)
            gain -= weight_right * self._calculate_entropy(y_right)
            
        else: raise UnsupportedParameterError(self.criterion)
        return gain

    def _best_split(self, X: Matrix, y: Vector) -> Tuple[int, float]:
        best_gain = -1
        split_idx, split_threshold = None, None
        n = X.shape[1] if self.max_features is None else self.max_features
        
        features = np.random.choice(X.shape[1], n, replace=False)

        for feature_index in features:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left = np.where(X[:, feature_index] <= threshold)
                right = np.where(X[:, feature_index] > threshold)
                
                if len(left[0]) < self.min_samples_split \
                    or len(right[0]) < self.min_samples_split:
                    continue
                
                gain = self._calculate_information_gain(
                    y, y[left], y[right]
                )
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    split_idx, split_threshold = feature_index, threshold
        
        return split_idx, split_threshold
    
    def _stopping_criteria(self, y: Vector, depth: int) -> bool:
        if len(np.unique(y)) == 1: return True
        if self.max_depth is not None and depth >= self.max_depth: return True
        if len(y) < self.min_samples_split: return True
        
        return False

    def _predict_single(self, node: DecisionTreeNode, x: Vector) -> int:
        if node.value is not None: 
            return max(node.value, key=node.value.get)
        if x[node.feature_index] <= node.threshold: 
            return self._predict_single(node.left, x)

        return self._predict_single(node.right, x)

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted: raise NotFittedError(self)
        return np.array([self._predict_single(self.root, x) for x in X])

    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

