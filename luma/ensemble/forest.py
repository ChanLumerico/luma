from typing import Literal, Self
from scipy.stats import mode
import numpy as np

from luma.interface.typing import Matrix, Vector
from luma.classifier.tree import DecisionTreeClassifier
from luma.regressor.tree import DecisionTreeRegressor
from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError


__all__ = ("RandomForestClassifier", "RandomForestRegressor")


class RandomForestClassifier(Estimator, Estimator.Meta, Supervised):
    """
    A Random Forest Classifier is an ensemble learning method in machine learning,
    widely used for both classification and regression tasks. It operates by
    constructing a multitude of decision trees during the training phase and
    outputs the mode of the classes for classification.

    Parameters
    ----------
    `n_trees` : int, default=10
        Number of trees in the forest
    `max_depth` :  int, default=100
        Maximum depth of each trees
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
    `bootstrap` : bool, default=True
        Whether to bootstrap the samples of dataset
    `bootstrap_feature` : bool, default=False
        Whether to bootstrap the features of each data
    `n_features` : int or {"auto"}, default="auto"
        Number of features to be sampled when bootstrapping features

    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 100,
        criterion: Literal["gini", "entropy"] = "gini",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        min_impurity_decrease: float = 0.0,
        max_leaf_nodes: int | None = None,
        bootstrap: bool = True,
        bootstrap_feature: bool = False,
        n_features: int | Literal["auto"] = "auto",
        verbose: bool = False,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.verbose = verbose
        self.trees = None
        self._fitted = False

        self.set_param_ranges(
            {
                "n_trees": ("0<,+inf", int),
                "max_depth": ("0<,+inf", int),
                "min_samples_split": ("0<,+inf", int),
                "max_samples_leaf": ("0<,+inf", int),
                "min_impurity_decrease": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix, sample_weights: Vector = None) -> Self:
        m, n = X.shape
        if self.n_features == "auto":
            self.n_features = int(n**0.5)
        else:
            if isinstance(self.n_features, str):
                raise UnsupportedParameterError(self.n_features)

        _tree_params = {
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_leaf_nodes": self.max_leaf_nodes,
        }

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

            tree.set_params(**_tree_params)
            tree.fit(X_sample, y_sample, sample_weights=sample_weights)

            if self.verbose:
                print(f"[RandomForest] Finished fitting tree {i}/{self.n_trees}")

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        preds = Matrix([tree.predict(X) for tree in self.trees])
        majority, _ = mode(preds, axis=0)

        return majority.flatten()

    def score(self, X: Matrix, y: Matrix, metric: Evaluator = Accuracy) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __gettiem__(self, index: int) -> DecisionTreeClassifier:
        return self.trees[index]


class RandomForestRegressor(Estimator, Estimator.Meta, Supervised):
    """
    A Random Forest Regressor is an ensemble learning algorithm for regression
    tasks. It builds multiple decision trees during training, each on a random
    subset of features and bootstrapped data. The final prediction is the average
    of individual tree predictions, providing a robust and accurate result.

    Parameters
    ----------
    `n_trees` : int, default=10
        Number of trees in the forest
    `max_depth` : int, default=10
        Maximum depth of each trees
    `min_samples_split` : int, default=2
        Minimum samples required to split a node
    `min_samples_leaf` : int, default=1
        Minimum samples required to be at a leaf node
    `max_features` : int, optional, default=None
        Number of features to consider
    `min_variance_decrease` : float, default=0.0
        Minimum decrement of variance for a split
    `max_leaf_nodes` : int, optional, default=None
        Maximum amount of leaf nodes
    `bootstrap` : bool, default=True
        Whether to bootstrap the samples of dataset
    `bootstrap_feature` : bool, default=False
        Whether to bootstrap the features of each data
    `n_features` : int or {"auto"}, default="auto"
        Number of features to be sampled when bootstrapping features

    """

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        min_variance_decrease: float = 0.0,
        max_leaf_nodes: int | None = None,
        bootstrap: bool = True,
        bootstrap_feature: bool = False,
        n_features: int | Literal["auto"] = "auto",
        sample_weights: Vector = None,
        verbose: bool = False,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_variance_decrease = min_variance_decrease
        self.max_leaf_nodes = max_leaf_nodes
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_feature = bootstrap_feature
        self.sample_weights = sample_weights
        self.verbose = verbose
        self.trees = None
        self._fitted = False

        self.set_param_ranges(
            {
                "n_trees": ("0<,+inf", int),
                "max_depth": ("0<,+inf", int),
                "min_samples_split": ("0<,+inf", int),
                "max_samples_leaf": ("0<,+inf", int),
                "min_variance_decrease": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def fit(self, X: Matrix, y: Matrix) -> Self:
        m, n = X.shape
        if self.n_features == "auto":
            self.n_features = int(n**0.5)
        else:
            if isinstance(self.n_features, str):
                raise UnsupportedParameterError(self.n_features)

        _tree_params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_variance_decrease": self.min_variance_decrease,
            "max_leaf_nodes": self.max_leaf_nodes,
        }

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

            tree.set_params(**_tree_params)
            tree.fit(X_sample, y_sample, sample_weights=self.sample_weights)

            if self.verbose:
                print(f"[RandomForest] Finished fitting tree {i}/{self.n_trees}")

        self._fitted = True
        return self

    def predict(self, X: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        preds = Matrix([tree.predict(X) for tree in self.trees])
        average_predictions = np.mean(preds, axis=0)

        return average_predictions

    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator = MeanSquaredError
    ) -> float:
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)

    def __gettiem__(self, index: int) -> DecisionTreeRegressor:
        return self.trees[index]
