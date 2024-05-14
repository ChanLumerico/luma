from typing import List, Literal, Optional, Tuple
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from luma.interface.typing import Matrix, Vector
from luma.core.super import Evaluator, Visualizer, Estimator
from luma.model_selection.split import TrainTestSplit
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Accuracy, Precision, Recall, Specificity
from luma.model_selection.cv import CrossValidator
from luma.model_selection.fold import FoldType, KFold


__all__ = (
    "DecisionRegion",
    "ClusterPlot",
    "ROCCurve",
    "PrecisionRecallCurve",
    "ConfusionMatrix",
    "ResidualPlot",
    "LearningCurve",
    "ValidationCurve",
    "InertiaPlot",
)


class DecisionRegion(Visualizer):
    def __init__(
        self,
        estimator: Estimator,
        X: Matrix,
        y: Optional[Matrix] = None,
        title: str | Literal["auto"] = "auto",
        xlabel: str = r"$x_1$",
        ylabel: str = r"$x_2$",
        cmap: ListedColormap = "rainbow",
        alpha: float = 0.4,
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha

        if self.title == "auto":
            self.title = type(self.estimator).__name__

        if self.y is None and hasattr(self.estimator, "labels"):
            self.y = self.estimator.labels

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        size: float = 250,
        scale: float = 10.0,
        grid: bool = True,
        show: bool = False,
    ) -> plt.Axes:
        x1_min, x1_max = self.X[:, 0].min(), self.X[:, 0].max()
        x2_min, x2_max = self.X[:, 1].min(), self.X[:, 1].max()
        delta_1, delta_2 = (x1_max - x1_min) / size, (x2_max - x2_min) / size

        x1_min, x1_max = x1_min - delta_1 * scale, x1_max + delta_1 * scale
        x2_min, x2_max = x2_min - delta_2 * scale, x2_max + delta_2 * scale
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, delta_1), np.arange(x2_min, x2_max, delta_2)
        )

        Z = self.estimator.predict(Matrix([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.contourf(
            xx1, xx2, Z, alpha=self.alpha, cmap=self.cmap, levels=len(np.unique(self.y))
        )

        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())

        ax.scatter(
            self.X[:, 0],
            self.X[:, 1],
            c=self.y,
            cmap=self.cmap,
            alpha=0.8,
            edgecolors="black",
        )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        if grid:
            ax.grid(alpha=0.2)

        ax.figure.tight_layout()
        if show:
            plt.show()
        return ax


class ClusterPlot(Visualizer):
    def __init__(
        self,
        estimator: Estimator,
        X: Matrix,
        title: str | Literal["auto"] = "auto",
        xlabel: str = r"$x_1$",
        ylabel: str = r"$x_2$",
        cmap: ListedColormap = "rainbow",
        alpha: float = 0.8,
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha
        self.labels = self.estimator.labels

        if self.title == "auto":
            self.title = type(self.estimator).__name__

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.scatter(
            self.X[self.labels == -1, 0],
            self.X[self.labels == -1, 1],
            marker="x",
            c="black",
            label="Noise",
        )

        ax.scatter(
            self.X[self.labels != -1, 0],
            self.X[self.labels != -1, 1],
            marker="o",
            c=self.labels[self.labels != -1],
            cmap=self.cmap,
            alpha=self.alpha,
            edgecolors="black",
        )

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if len(self.X[self.labels == -1]):
            ax.legend()
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax


class ROCCurve(Visualizer):
    def __init__(self, y_true: Vector, y_scores: Matrix) -> None:
        self.y_true = y_true
        self.y_scores = y_scores
        self.n_classes = y_scores.shape[1]

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        y_binary = LabelBinarizer().fit_transform(self.y_true)

        fprs, tprs = [], []
        for cl in range(self.n_classes):
            fpr, tpr = self._fpr_tpr(y_binary[:, cl], self.y_scores[:, cl])
            fprs.append(fpr)
            tprs.append(tpr)

            auc = self._auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve {cl} (area = {auc:.2f})")

        mean_fpr, mean_tpr = np.mean(fprs, axis=0), np.mean(tprs, axis=0)
        mean_auc = self._auc(mean_fpr, mean_tpr)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="dimgray",
            linewidth=2,
            linestyle="--",
            label=f"Mean ROC (area = {mean_auc:.2f})",
        )

        ax.plot(
            [0, 1], [0, 1], color="darkgray", linestyle=":", label="Random Guessing"
        )

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("False Positive Rate")

        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    def _fpr_tpr(self, y_true: Vector, y_score: Vector) -> Tuple[Vector, Vector]:
        thresholds = np.sort(y_score)[::-1]
        fpr, tpr = [], []
        for threshold in thresholds:
            y_pred = y_score >= threshold
            tpr.append(Recall.score(y_true, y_pred))
            fpr.append(1 - Specificity.score(y_true, y_pred))

        return Vector(fpr), Vector(tpr)

    def _auc(self, fpr: Vector, tpr: Vector) -> float:
        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
        return auc


class PrecisionRecallCurve(Visualizer):
    def __init__(self, y_true: Vector, y_scores: Matrix) -> None:
        self.y_true = y_true
        self.y_scores = y_scores
        self.n_classes = y_scores.shape[1]

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        y_binary = LabelBinarizer().fit_transform(self.y_true)
        pres, recs = [], []
        for cl in range(self.n_classes):
            pre, rec = self._pre_rec(y_binary[:, cl], self.y_scores[:, cl])
            pres.append(pre)
            recs.append(rec)

            ap = self._average_precision(pre, rec)
            ax.plot(rec, pre, linewidth=2, label=f"PR Curve {cl} (AP = {ap:.2f})")

        mean_pre, mean_rec = np.mean(pres, axis=0), np.mean(recs, axis=0)
        mean_ap = self._average_precision(mean_pre, mean_rec)

        ax.plot(
            mean_rec,
            mean_pre,
            color="dimgray",
            linewidth=2,
            linestyle="--",
            label=f"Mean PR (AP = {mean_ap:.2f})",
        )

        ax.plot(
            [1, 0], [0, 1], color="darkgray", linestyle=":", label="Random Guessing"
        )

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    def _pre_rec(self, y_true: Vector, y_score: Vector) -> Tuple[Vector, Vector]:
        thresholds = np.sort(y_score)[::-1]
        pre, rec = [], []
        for threshold in thresholds:
            y_pred = y_score >= threshold
            pre.append(Precision.score(y_true, y_pred))
            rec.append(Recall.score(y_true, y_pred))

        return Vector(pre), Vector(rec)

    def _average_precision(self, pre: Vector, rec: Vector) -> float:
        ap = 0
        for i in range(1, len(rec)):
            ap += (rec[i] - rec[i - 1]) * pre[i]
        return ap


class ConfusionMatrix(Visualizer):
    def __init__(
        self,
        y_true: Vector,
        y_pred: Vector,
        labels: List[str] | None = None,
        title: str | Literal["auto"] = "auto",
        cmap: ListedColormap = "Blues",
    ) -> None:
        self.cmap = cmap
        self.conf_matrix = self._confusion_matrix(y_true, y_pred)
        self.labels = labels
        self.title = title

        if labels is None:
            self.labels = np.arange(len(np.unique(y_true)))

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        cax = ax.imshow(self.conf_matrix, interpolation="nearest", cmap=self.cmap)
        if self.title == "auto":
            ax.set_title("Confusion Matrix")
        else:
            ax.set_title(self.title)

        plt.colorbar(cax, ax=ax)

        tick_marks = np.arange(len(self.labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.labels)

        thresh = self.conf_matrix.max() / 2.0
        for i, j in np.ndindex(self.conf_matrix.shape):
            ax.text(
                j,
                i,
                format(self.conf_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if self.conf_matrix[i, j] > thresh else "black",
            )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_aspect("equal", adjustable="box")
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    def _confusion_matrix(self, y_true: Vector, y_pred: Vector) -> Matrix:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1

        return matrix


class ResidualPlot(Visualizer):
    def __init__(
        self,
        estimator: Estimator,
        X: Matrix,
        y: Vector,
        alpha: float = 0.8,
        cmap: ListedColormap = "RdYlBu",
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.alpha = alpha
        self.cmap = cmap

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        resid = self._calculate_residuals()
        cax = ax.scatter(
            self.estimator.predict(self.X),
            resid,
            c=resid,
            s=20,
            cmap=self.cmap,
            alpha=self.alpha,
        )

        ax.axhline(y=0, c="black", lw=2, label="Perfect Fit")

        ax.axhline(y=resid.mean(), c="gray", ls="--", lw=2, label="Average Residual")

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residual Plot of {type(self.estimator).__name__}")

        ax.figure.colorbar(cax)
        ax.legend()
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

    def _calculate_residuals(self) -> Vector:
        if not self.estimator._fitted:
            self.estimator.fit(self.X, self.y)

        predictions = self.estimator.predict(self.X)
        residuals = self.y - predictions
        return residuals


class LearningCurve(Visualizer):
    """
    A learning curve in machine learning is a graph that compares the performance
    of a model on training and validation data over a series of training iterations.
    It helps to diagnose problems like overfitting or underfitting by showing how
    the model's error changes as it learns. A steep learning curve indicates rapid
    learning, while a plateau suggests no further learning. The ideal curve shows
    decreasing error on both training and validation datasets, converging to a
    low level.

    Parameters
    ----------
    `estimator` : Estimator
        An estimator to evaluate
    `X` : Matrix
        Feature data for training
    `y`: Vector
        Target data for training
    `train_sizes` : Vector, optional, default=None
        List of fractional values for each dataset size
    `test_size` : float, default=0.3
        Proportional value for validation set size
    `cv` : int, default=5
        Number of times for cross-validation
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `stratify` : bool, default=False
        Whether to perform stratified split
    `fold_type` : FoldType, default=KFold
        Fold type
    `metric` : Evaluator, default=Accuracy
        Evaluation metric for scoring
    `random_state` : int, optional, default=None
        Seed for random sampling upon splitting samples

    """

    def __init__(
        self,
        estimator: Estimator,
        X: Matrix,
        y: Vector,
        train_sizes: Vector | None = None,
        test_size: float = 0.3,
        cv: int = 5,
        shuffle: bool = True,
        stratify: bool = False,
        fold_type: FoldType = KFold,
        metric: Evaluator = Accuracy,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.train_sizes = train_sizes
        self.test_size = test_size
        self.cv = cv
        self.shuffle = shuffle
        self.stratify = stratify
        self.fold_type = fold_type
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        if train_sizes is None:
            self.train_sizes = np.linspace(0.1, 1.0, 5)

    def _evaluate(self) -> None:
        m, _ = self.X.shape
        self.train_scores, self.test_scores = [], []

        for train_size in self.train_sizes:
            n_samples = int(train_size * m)

            X_train, X_val, y_train, y_val = TrainTestSplit(
                self.X,
                self.y,
                test_size=n_samples,
                shuffle=self.shuffle,
                stratify=self.stratify,
                random_state=self.random_state,
            ).get

            if n_samples == len(self.X):
                X_train = X_val = self.X
                y_train = y_val = self.y

            self.estimator.fit(X_train, y_train)
            cv_param = (
                self.estimator,
                self.metric,
                self.cv,
                self.fold_type,
                self.shuffle,
                self.random_state,
                self.verbose,
            )

            if self.verbose:
                print(f"Starting CV for training set with {self.cv} folds")

            cv_train = CrossValidator(*cv_param)
            cv_train._fit(X_train, y_train)
            self.train_scores.append(cv_train.test_scores_)

            if self.verbose:
                print(f"Starting CV for validation set with {self.cv} folds")

            cv_test = CrossValidator(*cv_param)
            cv_test._fit(X_val, y_val)
            self.test_scores.append(cv_test.test_scores_)

            if self.verbose:
                print(
                    f"[LearningCurve] Finished evaluating for", f"{n_samples} samples"
                )

        self.train_scores = Matrix(self.train_scores)
        self.test_scores = Matrix(self.test_scores)

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        self._evaluate()
        self.train_sizes = (self.train_sizes * self.X.shape[0]).astype(int)
        metric_name = self.metric.__name__

        train_mean = self.train_scores.mean(axis=1)
        train_std = self.train_scores.std(axis=1)
        test_mean = self.test_scores.mean(axis=1)
        test_std = self.test_scores.std(axis=1)

        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.plot(
            self.train_sizes,
            train_mean,
            "o-",
            color="royalblue",
            label=f"Training {metric_name}",
        )
        ax.plot(
            self.train_sizes,
            test_mean,
            "o--",
            color="seagreen",
            label=f"Validation {metric_name}",
        )

        ax.fill_between(
            self.train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            color="royalblue",
            alpha=0.2,
        )

        ax.fill_between(
            self.train_sizes,
            test_mean + test_std,
            test_mean - test_std,
            color="seagreen",
            alpha=0.2,
        )

        ax.set_title(f"Learning Curve")
        ax.set_xlabel("Number of Training Examples")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0.0, 1.1)
        ax.legend()
        ax.grid()
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax


class ValidationCurve(Visualizer):
    """
    A validation curve in machine learning is a graph that compares the performance
    of a model on training and validation data over a range of values for a specific
    hyperparameter. It helps to diagnose the best hyperparameter setting by showing
    how the model's error changes as the value of the hyperparameter changes.

    Parameters
    ----------
    `estimator` : Estimator
        An estimator to evaluate
    `X` : Matrix
        Feature data for training
    `y`: Vector
        Target data for training
    `param_name` : str
        Name of the hyperparameter to be varied
    `param_range` : Vector
        Range of values for the hyperparameter
    `cv` : int, default=5
        Number of times for cross-validation
    `shuffle` : bool, default=True
        Whether to shuffle the dataset
    `fold_type` : FoldType, default=KfFold
        Fold type
    `metric` : Evaluator, default=Accuracy
        Evaluation metric for scoring
    `random_state` : int, optional, default=None
        Seed for random sampling upon splitting samples

    """

    def __init__(
        self,
        estimator: Estimator,
        X: Matrix,
        y: Vector,
        param_name: str,
        param_range: Vector,
        cv: int = 5,
        shuffle: bool = True,
        fold_type: FoldType = KFold,
        metric: Evaluator = Accuracy,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.param_name = param_name
        self.param_range = param_range
        self.cv = cv
        self.shuffle = shuffle
        self.fold_type = fold_type
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

    def _evaluate(self) -> None:
        self.train_scores, self.test_scores = [], []
        for param_value in self.param_range:
            setattr(self.estimator, self.param_name, param_value)

            cv_model = CrossValidator(
                estimator=self.estimator,
                metric=self.metric,
                cv=self.cv,
                shuffle=self.shuffle,
                fold_type=self.fold_type,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            cv_model._fit(self.X, self.y)
            self.train_scores.append(cv_model.train_scores_)
            self.test_scores.append(cv_model.test_scores_)

            if self.verbose:
                print(
                    f"[ValidationCurve] Finished evaluating for",
                    f"{self.param_name}={param_value}",
                )

        self.train_scores = Matrix(self.train_scores)
        self.test_scores = Matrix(self.test_scores)

    def plot(
        self, ax: Optional[plt.Axes] = None, xscale: str = None, show: bool = False
    ) -> plt.Axes:
        self._evaluate()
        metric_name = self.metric.__name__

        train_mean = self.train_scores.mean(axis=1)
        train_std = self.train_scores.std(axis=1)
        test_mean = self.test_scores.mean(axis=1)
        test_std = self.test_scores.std(axis=1)

        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.plot(
            self.param_range,
            train_mean,
            "o-",
            color="royalblue",
            label=f"Training {metric_name}",
        )
        ax.plot(
            self.param_range,
            test_mean,
            "o--",
            color="seagreen",
            label=f"Validation {metric_name}",
        )

        ax.fill_between(
            self.param_range,
            train_mean + train_std,
            train_mean - train_std,
            color="royalblue",
            alpha=0.2,
        )

        ax.fill_between(
            self.param_range,
            test_mean + test_std,
            test_mean - test_std,
            color="seagreen",
            alpha=0.2,
        )

        ax.set_title(f"Validation Curve for '{self.param_name}'")
        ax.set_xlabel(self.param_name)
        ax.set_ylabel(metric_name)

        ax.set_xscale(xscale if xscale else "linear")
        ax.set_ylim(0.0, 1.1)
        ax.legend()
        ax.grid()
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax


class InertiaPlot(Visualizer):
    """
    An inertia plot in clustering visualizes the inertia against the number of
    clusters, helping to determine the optimal number of clusters by identifying
    the "elbow" point where the rate of decrease in inertia sharply changes.
    It reflects how well clusters are compact and separated, with lower inertia
    indicating better clustering. The plot aids in applying the elbow method for
    selecting a reasonable number of clusters in algorithms like K-means. The
    goal is to choose the number of clusters where inertia begins to decrease
    more slowly.

    Parameters
    ----------
    `inertia_list` : list for float
        List of inertia values for various cluster sizes
    `n_clusters_list` : list of int
        List of the number of clusters

    Examples
    --------
    ```py
    inertia_list = []
    for i in range(2, 10):
        km = KMeansClustering(n_clusters=i)
        km.fit(data)

        inr = Inertia.score(data, km.centroids)
        inertia_list.append(inr)

    inr_plot = InertiaPlot(inertia_list=inertia_list,
                           n_clusters_list=list(range(2, 10)))
    inr_plot.plot()
    >>> plt.Axes
    ```
    """

    def __init__(self, inertia_list: List[float], n_clusters_list: List[int]) -> None:
        self.inertia_list = inertia_list
        self.n_clusters_list = n_clusters_list

    def _derivate_inertia(self) -> Vector:
        deriv = [0.0]
        for i in range(1, len(self.n_clusters_list) - 1):
            df = (self.inertia_list[i - 1] - self.inertia_list[i + 1]) / 2
            deriv.append(df)

        deriv.append(0.0)
        return np.abs(deriv)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        marker: str = "o",
        linestyle: str = "-",
        show: bool = False,
    ) -> None:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        ax.plot(
            self.n_clusters_list,
            self.inertia_list,
            marker=marker,
            linestyle=linestyle,
            label="Inertia",
        )

        ax.plot(
            self.n_clusters_list,
            self._derivate_inertia(),
            marker=marker,
            linestyle=linestyle,
            label="Absolute Derivative",
        )

        best_idx = np.argmax(self._derivate_inertia())
        best_n_clusters = self.n_clusters_list[best_idx]
        ax.axvspan(
            best_n_clusters - 0.25,
            best_n_clusters + 0.25,
            color="orange",
            alpha=0.2,
            label="Best Number of Clusters",
        )

        ax.set_title("Inertia Plot")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia / Deriv. of Inertia")
        ax.legend()
        ax.grid()
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax
