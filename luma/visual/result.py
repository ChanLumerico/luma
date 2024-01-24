from typing import List, Literal, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import numpy as np

from luma.interface.util import Matrix, Vector
from luma.interface.super import Visualizer, Estimator
from luma.preprocessing.encoder import LabelBinarizer
from luma.metric.classification import Recall, Specificity


__all__ = (
    'DecisionRegion', 
    'ClusterPlot',
    'ROCCurve',
    'ConfusionMatrix'
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

    def plot(self, 
             ax: Optional[plt.Axes] = None, 
             size: float = 250, 
             scale: float = 10.0, 
             show: bool = False) -> plt.Axes:
        x1_min, x1_max = self.X[:, 0].min(), self.X[:, 0].max()
        x2_min, x2_max = self.X[:, 1].min(), self.X[:, 1].max()
        delta_1, delta_2 = (x1_max - x1_min) / size, (x2_max - x2_min) / size
        
        x1_min, x1_max = x1_min - delta_1 * scale, x1_max + delta_1 * scale
        x2_min, x2_max = x2_min - delta_2 * scale, x2_max + delta_2 * scale
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta_1), 
                               np.arange(x2_min, x2_max, delta_2))
        
        Z = self.estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        if ax is None: _, ax = plt.subplots()
        ax.contourf(xx1, xx2, Z, 
                    alpha=self.alpha, 
                    cmap=self.cmap, 
                    levels=len(np.unique(self.y)))
        
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())

        ax.scatter(self.X[:, 0], self.X[:, 1], 
                   c=self.y, 
                   cmap=self.cmap, 
                   alpha=0.8, 
                   edgecolors='black')
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        ax.figure.tight_layout()
        
        if show: plt.show()
        return ax


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
    
    def plot(self, 
             ax: Optional[plt.Axes] = None, 
             show: bool = False) -> plt.Axes:
        if ax is None: _, ax = plt.subplots()
        ax.scatter(self.X[self.labels == -1, 0], 
                   self.X[self.labels == -1, 1],
                   marker='x',
                   c='black', 
                   label='Noise')
        
        ax.scatter(self.X[self.labels != -1, 0], 
                   self.X[self.labels != -1, 1], 
                   marker='o',
                   c=self.labels[self.labels != -1],
                   cmap=self.cmap,
                   alpha=self.alpha,
                   edgecolors='black')
        
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        
        if len(self.X[self.labels == -1]): ax.legend()
        ax.figure.tight_layout()

        if show: plt.show()
        return ax


class ROCCurve(Visualizer):
    def __init__(self,
                 y_true: Vector,
                 y_scores: Matrix) -> None:
        self.y_true = y_true
        self.y_scores = y_scores
        self.n_classes = y_scores.shape[1]
    
    def plot(self, 
             ax: Optional[plt.Axes] = None, 
             show: bool = False) -> plt.Axes:
        if ax is None: _, ax = plt.subplots()
        y_binary = LabelBinarizer().fit_transform(self.y_true)
        
        fprs, tprs = [], []
        for cl in range(self.n_classes):
            fpr, tpr = self._fpr_tpr(y_binary[:, cl], self.y_scores[:, cl])
            fprs.append(fpr)
            tprs.append(tpr)
            
            auc = self._auc(fpr, tpr)
            ax.plot(fpr, tpr, 
                    linewidth=2,
                    label=f'ROC Curve {cl} (area = {auc:.2f})')
        
        mean_fpr, mean_tpr = np.mean(fprs, axis=0), np.mean(tprs, axis=0)
        mean_auc = self._auc(mean_fpr, mean_tpr)
        
        ax.plot(mean_fpr, mean_tpr,
                color='dimgray', 
                linewidth=2,
                linestyle='--', 
                label=f'Mean ROC (area = {mean_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 
                color='darkgray', 
                linestyle=':', 
                label='Random Guessing')
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.figure.tight_layout()

        if show: plt.show()
        return ax
    
    def _fpr_tpr(self, y_true: Vector, y_score: Vector) -> Tuple[Vector, Vector]:
        thresholds = np.sort(y_score)[::-1]
        fpr, tpr = [], []
        for threshold in thresholds:
            y_pred = y_score >= threshold
            tpr.append(Recall.score(y_true, y_pred))
            fpr.append(1 - Specificity.score(y_true, y_pred))
        
        return np.array(fpr), np.array(tpr)
    
    def _auc(self, fpr: Vector, tpr: Vector) -> float:
        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
        return auc


class ConfusionMatrix(Visualizer):
    def __init__(self, 
                 y_true: Vector, 
                 y_pred: Vector, 
                 labels: List[str] | None = None,
                 cmap: ListedColormap = 'Blues') -> None:
        self.cmap = cmap
        self.conf_matrix = self._confusion_matrix(y_true, y_pred)
        
        if labels is None:
            self.labels = np.arange(len(np.unique(y_true)))

    def plot(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Axes:
        if ax is None: ax = plt.gca()
        
        cax = ax.imshow(self.conf_matrix, interpolation='nearest', cmap=self.cmap)
        ax.set_title('Confusion Matrix')
        plt.colorbar(cax, ax=ax)

        tick_marks = np.arange(len(self.labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.labels, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.labels)

        thresh = self.conf_matrix.max() / 2.
        for i, j in np.ndindex(self.conf_matrix.shape):
            ax.text(j, i, format(self.conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if self.conf_matrix[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_aspect('equal', adjustable='box')

        if show: plt.show()
        return ax

    def _confusion_matrix(self, y_true: Vector, y_pred: Vector) -> Matrix:
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1
        
        return matrix

