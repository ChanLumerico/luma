from typing import Literal
import numpy as np

from luma.interface.super import Transformer
from luma.interface.util import Matrix
from luma.interface.exception import NotFittedError, UnsupportedParameterError


class OneHotEncoder(Transformer, Transformer.Feature):
    def __init__(self):
        self.categories_ = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'OneHotEncoder':
        self.categories_ = [np.unique(col) for col in X.T]
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix[int]:
        if not self._fitted: raise NotFittedError(self)
        X_out = []
        for i, categories in enumerate(self.categories_):
            label_to_index = {label: idx for idx, label in enumerate(categories)}
            matrix = np.zeros((len(X), len(categories)))
            
            for j, item in enumerate(X[:, i]):
                if item in label_to_index:
                    matrix[j, label_to_index[item]] = 1
                else:
                    raise ValueError(f"Unknown label {item} found in column {i}")
            
            X_out.append(matrix)

        return np.hstack(X_out).astype(int)

    def fit_transform(self, X: Matrix) -> Matrix[int]:
        self.fit(X)
        return self.transform(X)
    
    def set_params(self) -> None: ...


class LabelEncoder(Transformer, Transformer.Target):
    def __init__(self):
        self.classes_ = None
        self._fitted = False

    def fit(self, y: Matrix) -> 'LabelEncoder':
        self.classes_ = np.unique(y)
        self._fitted = True
        return self

    def transform(self, y: Matrix) -> Matrix[int]:
        if not self._fitted: raise NotFittedError(self)
        class_to_index = {k: v for v, k in enumerate(self.classes_)}
        
        X_transformed = np.array([class_to_index.get(y_, -1) for y_ in y])
        if -1 in X_transformed:
            raise ValueError("Unknown label found in input data.")

        return X_transformed

    def fit_transform(self, y: Matrix) -> Matrix[int]:
        self.fit(y)
        return self.transform(y)

    def set_params(self) -> None: ...


class OrdinalEncoder(Transformer, Transformer.Feature):
    def __init__(self, 
                 strategy: Literal['appear', 'alpha'] = 'appear'):
        self.categories_ = None
        self.strategy = strategy
        self._fitted = False

    def fit(self, X: Matrix) -> 'OrdinalEncoder':
        if self.strategy == 'appear':
            self.categories_ = [np.unique(col, return_index=True)[0] for col in X.T]
        elif self.strategy == 'alpha':
            self.categories_ = [np.sort(np.unique(col)) for col in X.T]
        else:
            raise UnsupportedParameterError(self.strategy)

        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix[int]:
        if not self._fitted: raise NotFittedError(self)
        X_out = np.zeros(X.shape, dtype=int)
        
        for i, categories in enumerate(self.categories_):
            category_to_index = {category: index for index, category in enumerate(categories)}
            for j, item in enumerate(X[:, i]):
                if item in category_to_index:
                    X_out[j, i] = category_to_index[item]
                else:
                    raise ValueError(f"Unknown label {item} found in column {i}")

        return X_out

    def fit_transform(self, X: Matrix) -> Matrix[int]:
        self.fit(X)
        return self.transform(X)

    def set_params(self) -> None: ...
