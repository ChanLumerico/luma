from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from luma.core.super import Visualizer


__all__ = (
    'CorrelationHeatMap', 
    'CorrelationBar', 
    'JointPlot', 
    'MissingProportion'
)


class CorrelationHeatMap(Visualizer):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.corr = data.corr()
    
    def plot(self, 
             ax: Optional[plt.Axes] = None, 
             colorMap: str = 'rocket', 
             annotate: bool = True, 
             colorBar: bool = True,
             show: bool = False) -> plt.Axes:
        if ax is None:
            n_features = self.data.shape[1]
            size = n_features / 2 if n_features < 20 else 10
            _, ax = plt.subplots(figsize=(size + 1, size))

        sns.heatmap(self.corr, 
                    ax=ax, 
                    cmap=colorMap, 
                    annot=annotate, 
                    cbar=colorBar, 
                    fmt='0.2f')
        
        ax.set_title('Correlation Heat Map')
        ax.figure.tight_layout()
        
        if show: plt.show()
        return ax


class CorrelationBar(Visualizer):
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        self.data = data
        self.target = target
    
    def plot(self, 
             ax: Optional[plt.Axes] = None, 
             show: bool = False) -> plt.Axes:
        if ax is None: _, ax = plt.subplots()
        corr_bar = [abs(self.data[col].corr(self.data[self.target])) for col in self.data]
        sns.barplot(x=self.data.columns, y=corr_bar, hue=self.data.columns, ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'Correlations with {self.target}')
        ax.set_xlabel('Features')
        ax.set_ylabel('Correlation')
        ax.figure.tight_layout()

        if show: plt.show()
        return ax


class JointPlot(Visualizer):
    def __init__(self, data: pd.DataFrame, x: str, y: str) -> None:
        self.data = data
        self.x = x
        self.y = y

    def plot(self, color: str = 'tab:blue') -> None:    
        sns.jointplot(data=self.data, x=self.x, y=self.y, kind='reg', color=color)
        plt.title(f'{self.x} vs. {self.y}')
        plt.tight_layout()
        plt.show()


class MissingProportion(Visualizer):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def nan_proportions(self) -> pd.DataFrame:
        nan_props = self.data.isna().mean()
        return nan_props

    def plot(self, 
             ax: Optional[plt.Axes] = None,
             show: bool = False) -> plt.Axes:
        if ax is None: _, ax = plt.subplots()
        nan_props = self.nan_proportions()
        sns.barplot(x=nan_props.index, y=nan_props.values, hue=self.data.columns, ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Proportion')
        ax.set_title('Missing Value Proportions')
        ax.figure.tight_layout()
        
        if show: plt.show()
        return ax

