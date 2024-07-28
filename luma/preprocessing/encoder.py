from typing import Literal, Self
import numpy as np

from luma.core.super import Transformer
from luma.interface.typing import Matrix, Vector
from luma.interface.exception import NotFittedError, UnsupportedParameterError


__all__ = (
    "OneHotEncoder",
    "LabelEncoder",
    "OrdinalEncoder",
    "LabelBinarizer",
    "LabelSmoothing",
)


class OneHotEncoder(Transformer, Transformer.Feature):
    def __init__(self, features: list = None):
        self.categories = None
        self.features = features
        self._fitted = False

    def fit(self, X: Matrix) -> Self:
        if self.features is None:
            self.features = range(X.shape[1])

        self.categories = [np.unique(X[:, col]) for col in self.features]
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix[int]:
        if not self._fitted:
            raise NotFittedError(self)

        X_out = []
        for i, col in enumerate(X.T):
            if i in self.features:
                categories = self.categories[self.features.index(i)]
                label_to_index = {label: idx for idx, label in enumerate(categories)}
                matrix = np.zeros((len(X), len(categories)))

                for j, item in enumerate(col):
                    if item in label_to_index:
                        matrix[j, label_to_index[item]] = 1
                    elif item is np.nan:
                        continue
                    else:
                        raise ValueError(f"Unknown label {item} found in column {i}")
                X_out.append(matrix)

            else:
                X_out.append(X[:, i].reshape(-1, 1))

        return np.hstack(X_out).astype(int)

    def fit_transform(self, X: Matrix) -> Matrix[int]:
        self.fit(X)
        return self.transform(X)


class LabelEncoder(Transformer, Transformer.Target):
    def __init__(self):
        self.classes = None
        self._fitted = False

    def fit(self, y: Matrix) -> Self:
        self.classes = np.unique(y)
        self._fitted = True
        return self

    def transform(self, y: Matrix) -> Matrix[int]:
        if not self._fitted:
            raise NotFittedError(self)
        class_to_index = {k: v for v, k in enumerate(self.classes)}

        X_transformed = Matrix([class_to_index.get(y_, -1) for y_ in y])
        if -1 in X_transformed:
            raise ValueError("Unknown label found in input data.")

        return X_transformed

    def inverse_transform(self, y: Matrix) -> Matrix:
        if not self._fitted:
            raise NotFittedError(self)
        index_to_class = {v: k for k, v in enumerate(self.classes)}

        X_inversed = Matrix([index_to_class.get(y_, None) for y_ in y])
        if None in X_inversed:
            raise ValueError("Unknown index found in input data.")

        return X_inversed

    def fit_transform(self, y: Matrix) -> Matrix[int]:
        self.fit(y)
        return self.transform(y)


class OrdinalEncoder(Transformer, Transformer.Feature):
    def __init__(self, strategy: Literal["occur", "alpha"] = "occur"):
        self.categories = None
        self.strategy = strategy
        self._fitted = False

    def fit(self, X: Matrix) -> Self:
        if self.strategy == "occur":
            self.categories = [np.unique(col, return_index=True)[0] for col in X.T]
        elif self.strategy == "alpha":
            self.categories = [np.sort(np.unique(col)) for col in X.T]
        else:
            raise UnsupportedParameterError(self.strategy)

        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix[int]:
        if not self._fitted:
            raise NotFittedError(self)
        X_out = np.zeros(X.shape, dtype=int)

        for i, categories in enumerate(self.categories):
            category_to_index = {
                category: index for index, category in enumerate(categories)
            }
            for j, item in enumerate(X[:, i]):
                if item in category_to_index:
                    X_out[j, i] = category_to_index[item]
                else:
                    raise ValueError(f"Unknown label {item} found in column {i}")

        return X_out

    def fit_transform(self, X: Matrix) -> Matrix[int]:
        self.fit(X)
        return self.transform(X)


class LabelBinarizer(Transformer, Transformer.Target):
    def __init__(self, negative_target: bool = False) -> None:
        self.negative_target = negative_target
        self.classes_ = None
        self._fitted = False

    def fit(self, y: Vector) -> Self:
        self.classes_ = np.unique(y)
        self._fitted = True
        return self

    def transform(self, y: Vector) -> Vector:
        if not self._fitted:
            raise NotFittedError(self)
        if self.negative_target:
            binarized = np.full((len(y), len(self.classes_)), -1)
        else:
            binarized = np.full((len(y), len(self.classes_)), 0)

        for i, label in enumerate(y):
            class_index = np.where(self.classes_ == label)[0][0]
            binarized[i, class_index] = 1

        return binarized

    def fit_transform(self, y: Vector) -> Vector:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: Vector) -> Vector:
        return self.classes_[np.argmax(y, axis=1)]


class LabelSmoothing(Transformer, Transformer.Target):
    def __init__(self, smoothing: float = 0.1) -> None:
        self.smoothing = smoothing
        self.classes_ = None
        self.fitted_ = False

    def fit(self, y: Matrix) -> Self:
        if y.ndim != 2:
            raise ValueError("Target values must be one-hot encoded!")
        self.classes_ = y.shape[1]
        self.fitted_ = True
        return self

    def transform(self, y: Matrix) -> Matrix:
        if not self.fitted_:
            raise NotFittedError(self)

        y_smooth = y * (1 - self.smoothing)
        y_smooth += self.smoothing / self.classes_
        return y_smooth

    def fit_transform(self, y: Matrix) -> Matrix:
        return self.fit(y).transform(y)
