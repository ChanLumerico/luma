from typing import *
import matplotlib.pyplot as plt
import numpy as np

from LUMA.Interface.Super import _Visualizer
from LUMA.Interface.Type import Estimator


__all__ = ['DecisionRegion', 'ClusteredRegion']


class DecisionRegion(_Visualizer):
    def __init__(self, 
                 estimator: Estimator, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 title: str='', 
                 xlabel: str=r'$x_2$', 
                 ylabel: str=r'$x_2$',
                 cmap: str='Spectral',
                 alpha: float=0.7) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha

    def plot(self, resolution: float=0.025) -> None:
        if self.X.shape[1] > 2: return
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                               np.arange(x2_min, x2_max, resolution))
        
        Z = self.estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, 
                     alpha=0.4, cmap=self.cmap, 
                     levels=len(np.unique(self.y)))
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.scatter(self.X[:, 0], self.X[:, 1], 
                    c=self.y, cmap=self.cmap, 
                    alpha=0.8, edgecolors='black')

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.tight_layout()
        plt.show()


class ClusteredRegion(_Visualizer):
    def __init__(self, 
                 estimator: Estimator, 
                 X: np.ndarray, 
                 title: str='', 
                 xlabel: str=r'$x_2$', 
                 ylabel: str=r'$x_2$',
                 cmap: str='Spectral',
                 alpha: float=0.7) -> None:
        self.estimator = estimator
        self.X = X
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha
    
    def plot(self, resolution: float=0.025) -> None:
        if self.X.shape[1] > 2: return
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                               np.arange(x2_min, x2_max, resolution))
        
        X_pred = self.estimator.predict(self.X)
        Z = self.estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, 
                     alpha=self.alpha, cmap=self.cmap, 
                     levels=len(np.unique(X_pred)))
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.scatter(self.X[:, 0], self.X[:, 1], 
                    c=X_pred, cmap=self.cmap, 
                    alpha=0.8, edgecolors='black')

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.tight_layout()
        plt.show()

