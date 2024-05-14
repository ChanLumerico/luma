from typing import Literal, Self, Tuple
import numpy as np

from luma.interface.typing import Matrix, Vector
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.util import DecisionTreeNode
from luma.metric.classification import Accuracy


__all__ = "DecisionTreeClassifier"


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
    `max_depth` : int, default=10
        Maximum depth of the tree
    `criterion` : {"gini", "entropy"}, default="gini"
        Function used to measure the quality of a split
    `min_samples_split` : int, default=2
        Minimum samples required to split a node
    `min_samples_leaf` : int, default=1
        Minimum samples required to be at a leaf node
    `max_features` : int, optional, default=None
        Number of features to consider
    `min_impurity_decrease` : float, default=0.0
        Minimum decrement of impurity for a split
    `max_leaf_nodes` : int, optional, default=None
        Maximum amount of leaf nodes
    `random_state` : int, optional, default=None
        The randomness seed of the estimator

    """

    def __init__(
        self,
        max_depth: int = 10,
        criterion: Literal["gini", "entropy"] = "gini",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        min_impurity_decrease: float = 0.0,
        max_leaf_nodes: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        self._fitted = False

        self.set_param_ranges(
            {
                "max_depth": ("0<,+inf", int),
                "min_samples_split": ("0,+inf", int),
                "min_samples_leaf": ("0,+inf", int),
                "min_impurity_decrease": ("0,+inf", None),
                "max_leaf_nodes": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        np.random.seed(random_state)

    def fit(self, X: Matrix, y: Vector, sample_weights: Vector = None) -> Self:
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        sample_weights = Vector(sample_weights)

        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, 0, sample_weights)
        self._fitted = True
        return self

    def _stopping_criteria(self, y: Vector, depth: int) -> bool:
        if len(np.unique(y)) == 1:
            return True
        if depth == self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        return False

    def _build_tree(
        self, X: Matrix, y: Vector, depth: int, sample_weights: Vector
    ) -> DecisionTreeNode:
        if self._stopping_criteria(y, depth):
            leaf_value = self._calculate_leaf_value(y, sample_weights)
            return DecisionTreeNode(value=leaf_value)

        feature_index, threshold = self._best_split(X, y, sample_weights)
        if feature_index is None:
            leaf_value = self._calculate_leaf_value(y, sample_weights)
            return DecisionTreeNode(value=leaf_value)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        X_left, X_right = X[left_indices], X[right_indices]
        y_left, y_right = y[left_indices], y[right_indices]

        weights_left = sample_weights[left_indices]
        weights_right = sample_weights[right_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1, weights_left)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, weights_right)

        return DecisionTreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _calculate_leaf_value(self, y: Vector, sample_weights: Vector) -> dict:
        classes, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(classes):
            weighted_counts[i] = sample_weights[y == cl].sum()

        return dict(zip(classes, weighted_counts))

    def _best_split(
        self, X: Matrix, y: Vector, sample_weights: Vector
    ) -> Tuple[int, float]:
        best_gain = 0
        best_feature, best_threshold = None, None
        current_impurity = self._calculate_impurity(y, sample_weights)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if (
                    np.sum(left_indices) < self.min_samples_split
                    or np.sum(right_indices) < self.min_samples_split
                ):
                    continue

                gain = self._calculate_gain(
                    y, sample_weights, left_indices, right_indices, current_impurity
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_impurity(self, y: Vector, sample_weights: Vector) -> float:
        if self.criterion == "gini":
            return self._calculate_gini(y, sample_weights)
        elif self.criterion == "entropy":
            return self._calculate_entropy(y, sample_weights)
        else:
            raise UnsupportedParameterError(self.criterion)

    def _calculate_gini(self, y: Vector, sample_weights: Vector) -> float:
        total_weight = np.sum(sample_weights)
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(np.unique(y)):
            weighted_counts[i] = np.sum(sample_weights[y == cl])

        prob = weighted_counts / total_weight
        gini = 1 - np.sum(prob**2)

        return gini

    def _calculate_entropy(self, y: Vector, sample_weights: Vector) -> float:
        total_weight = np.sum(sample_weights)
        _, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(counts, dtype=np.float64)

        for i, cl in enumerate(np.unique(y)):
            weighted_counts[i] = np.sum(sample_weights[y == cl])

        prob = weighted_counts / total_weight
        entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))

        return entropy

    def _calculate_gain(
        self,
        y: Vector,
        sample_weights: Vector,
        left_indices: Vector,
        right_indices: Vector,
        current_impurity: float,
    ) -> float:
        left_weight = np.sum(sample_weights[left_indices])
        right_weight = np.sum(sample_weights[right_indices])
        total_weight = left_weight + right_weight

        weights_left = sample_weights[left_indices]
        weights_right = sample_weights[right_indices]

        left_impurity = self._calculate_impurity(y[left_indices], weights_left)
        right_impurity = self._calculate_impurity(y[right_indices], weights_right)

        weighted_impurity = (left_weight / total_weight) * left_impurity
        weighted_impurity += (right_weight / total_weight) * right_impurity
        gain = current_impurity - weighted_impurity

        return gain

    def predict(self, X: Matrix) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        preds = [self._predict_single(self.root, x) for x in X]

        return Vector(preds)

    def _predict_single(self, node: DecisionTreeNode, x: Vector) -> int:
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return max(node.value, key=node.value.get)

    def predict_proba(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        proba_preds = [self._predict_proba_single(self.root, x) for x in X]

        return Matrix(proba_preds)

    def _predict_proba_single(self, node: DecisionTreeNode, x: Vector) -> Vector:
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        total_samples = sum(node.value.values())
        cl_probs = {cl: count / total_samples for cl, count in node.value.items()}

        probabilities = Matrix([cl_probs.get(cl, 0.0) for cl in self.classes_])
        return probabilities

    def score(self, X: Matrix, y: Vector, metric: Evaluator = Accuracy) -> float:
        preds = self.predict(X)
        return metric.score(y_true=y, y_pred=preds)
