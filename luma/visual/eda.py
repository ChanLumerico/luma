import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from luma.interface.super import Visualizer


__all__ = ['CorrelationHeatMap', 'CorrelationBar', 'JointPlot']


class CorrelationHeatMap(Visualizer):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.corr = data.corr()
    
    def plot(self, 
             colorMap: str = 'rocket', 
             annotate: bool = True, 
             colorBar: bool = True) -> None:
        n_features = self.data.shape[1]
        size = n_features / 2 if n_features < 20 else 10
        plt.figure(figsize=(size + 1, size))
        
        sns.heatmap(self.corr, cmap=colorMap, annot=annotate, cbar=colorBar, fmt='0.2f')
        plt.title('Correlation Heat Map')
        plt.tight_layout()
        plt.show()


class CorrelationBar(Visualizer):
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        self.data = data
        self.target = target
    
    def plot(self) -> None:
        corr_bar = []
        for col in self.data:
            corr_bar.append(abs(self.data[col].corr(self.data[self.target])))
            
        self.correlation_bar = corr_bar
        sns.barplot(x=self.data.columns, y=corr_bar, hue=self.data.columns)
        plt.title(f'Correlations with {self.target}')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.tight_layout()
        plt.show()


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

