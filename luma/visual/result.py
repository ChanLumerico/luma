from typing import Literal, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import numpy as np

from luma.interface.util import Matrix
from luma.interface.super import Visualizer, Estimator


__all__ = (
    'DecisionRegion', 
    'ClusterPlot'
)


class DecisionRegion(Visualizer):
    def __init__(self, 
                 estimator: Estimator,
                 X: Matrix, 
                 y: Optional[Matrix] = None, 
                 title: str | Literal['auto'] = 'auto', 
                 xlabel: str = r'$x_1$', 
                 ylabel: str = r'$x_2$',
                 cmap: ListedColormap = 'rainbow',
                 alpha: float = 0.4) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha
        
        if self.title == 'auto':
            self.title = type(self.estimator).__name__

        if self.y is None and hasattr(self.estimator, 'labels'):
            self.y = self.estimator.labels

    def plot(self, size: float = 250, scale: float = 10.0) -> None:
        x1_min, x1_max = self.X[:, 0].min(), self.X[:, 0].max()
        x2_min, x2_max = self.X[:, 1].min(), self.X[:, 1].max()
        delta_1, delta_2 = (x1_max - x1_min) / size, (x2_max - x2_min) / size
        
        x1_min, x1_max = x1_min - delta_1 * scale, x1_max + delta_1 * scale
        x2_min, x2_max = x2_min - delta_2 * scale, x2_max + delta_2 * scale
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta_1), 
                               np.arange(x2_min, x2_max, delta_2))
        
        Z = self.estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, 
                     alpha=self.alpha, cmap=self.cmap, 
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


class ClusterPlot(Visualizer):
    def __init__(self,
                 estimator: Estimator,
                 X: Matrix,
                 title: str | Literal['auto'] = 'auto',
                 xlabel: str = r'$x_1$',
                 ylabel: str = r'$x_2$',
                 cmap: ListedColormap = 'rainbow',
                 alpha: float = 0.8) -> None:
        self.estimator = estimator
        self.X = X
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.alpha = alpha
        self.labels = self.estimator.labels
        
        if self.title == 'auto':
            self.title = type(self.estimator).__name__
    
    def plot(self) -> None:
        plt.scatter(self.X[self.labels == -1, 0], 
                    self.X[self.labels == -1, 1],
                    marker='x',
                    c='black',
                    label='Noise')
        
        plt.scatter(self.X[self.labels != -1, 0], 
                    self.X[self.labels != -1, 1], 
                    marker='o',
                    c=self.labels[self.labels != -1],
                    cmap=self.cmap,
                    alpha=self.alpha,
                    edgecolors='black')
        
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        
        plt.legend() if len(self.X[self.labels == -1]) else Ellipsis
        plt.tight_layout()
        plt.show()

