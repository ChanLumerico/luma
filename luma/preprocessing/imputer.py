from typing import Literal
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np
import pandas as pd

from luma.interface.util import Matrix
from luma.core.super import Transformer
from luma.interface.exception import NotFittedError, UnsupportedParameterError


__all__ = (
    'SimpleImputer', 
    'KNNImputer', 
    'HotDeckImputer'
)


class SimpleImputer(Transformer, Transformer.Feature):
    def __init__(self, 
                 strategy: Literal['mean', 'median', 'mode'] = 'mean'):
        self.strategy = strategy
        self.statistics = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'SimpleImputer':
        if self.strategy == 'mean':
            self.statistics = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics = np.nanmedian(X, axis=0)
        elif self.strategy == 'mode':
            self.statistics = []
            
            for x_col in X.T:
                if np.issubdtype(x_col.dtype, np.number):
                    column_mode, _ = mode(x_col, nan_policy='omit')
                    self.statistics.append(column_mode if column_mode.size > 0 else np.nan)
                else:
                    unique, counts = np.unique(x_col[~pd.isnull(x_col)], return_counts=True)
                    if unique.size > 0: self.statistics.append(unique[np.argmax(counts)])
                    else: self.statistics.append(np.nan)
        
        else: raise UnsupportedParameterError(self.strategy)
        
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: NotFittedError(self)
        X_imp = X.copy()
        for i, stat in enumerate(self.statistics):
            if np.issubdtype(X_imp[:, i].dtype, np.number):
                X_imp[:, i] = X_imp[:, i].astype(float)
                mask = np.isnan(X_imp[:, i])
                X_imp[mask, i] = stat
            else:
                mask = pd.isnull(X_imp[:, i]) | (X_imp[:, i] == 'nan')
                X_imp[mask, i] = stat
        
        X_ret = np.zeros_like(X_imp, dtype=object)
        for i in range(X_imp.shape[1]):
            try: X_ret[:, i] = X_imp[:, i].astype(float)
            except: X_ret[:, i] = X_imp[:, i]
        
        return X_ret

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)


class KNNImputer(Transformer, Transformer.Feature):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.distances = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'KNNImputer':
        if np.isnan(X).any():
            self.distances = self._compute_distances(X)
        
        self._fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix:
        if not self._fitted: NotFittedError(self)
        X_imp = X.copy()
        
        for i, row in enumerate(X):
            nan_indices = np.where(np.isnan(row))[0]
            for x_col in nan_indices:
                neighbor_indices = np.argsort(self.distances[i])
                neighbor_values = []
                
                for neighbor_index in neighbor_indices:
                    if not np.isnan(X[neighbor_index, x_col]) and neighbor_index != i:
                        neighbor_values.append(X[neighbor_index, x_col])
                        if len(neighbor_values) == self.n_neighbors: break

                if neighbor_values: X_imp[i, x_col] = np.mean(neighbor_values)
                else: X_imp[i, x_col] = np.nanmean(X[:, x_col])

        return X_imp

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)

    def _compute_distances(self, X: Matrix) -> Matrix:
        nan_replacement = np.nanmax(X) + 1
        data_without_nan = np.where(np.isnan(X), nan_replacement, X)
        distances = cdist(data_without_nan, data_without_nan, metric='euclidean')
        
        return distances


class HotDeckImputer(Transformer, Transformer.Feature):
    def __init__(self):
        self._X = None
        self._similar_rows = None
        self._fitted = False

    def fit(self, X: Matrix) -> 'HotDeckImputer':
        self._X = X
        self._similar_rows = [self._find_similar_row(row) for row in self._X]
        
        self._fitted = True
        return self

    def _find_similar_row(self, row: Matrix) -> Matrix:
        min_diff = np.inf
        similar_row = None

        for row in self._X:
            if not np.any(np.isnan(row)):
                diff = np.sum(np.not_equal(row, row) & ~np.isnan(row))
                if diff < min_diff:
                    min_diff = diff
                    similar_row = row

        return similar_row

    def transform(self, X: Matrix) -> Matrix:
        imputed_data = X.copy()
        for i, row in enumerate(imputed_data):
            if np.any(np.isnan(row)) and self._similar_rows[i] is not None:
                missing_indices = np.where(np.isnan(row))[0]
                imputed_data[i, missing_indices] = self._similar_rows[i][missing_indices]

        return imputed_data

    def fit_transform(self, X: Matrix) -> Matrix:
        self.fit(X)
        return self.transform(X)

