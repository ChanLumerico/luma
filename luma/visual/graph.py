from matplotlib import colors, pyplot as plt
import numpy as np

from luma.interface.super import Visualizer
from luma.interface.util import Matrix


__all__ = (
    'GraphPlot'
)


class GraphPlot(Visualizer):
    
    """
    A visualizer which plots the structure of the given graph.
    For a weighted graph, edge weights are represented as a 
    gradient color of a colormap, while the edges of an
    unweighted graph are shown as an uniform color.
    
    Parameters
    ----------
    `nodes` : Coordinate matrix of nodes
    `edges` : Adjacency matrix of a graph
    
    Notes
    -----
    * Currently, only adjacency matrices are supported.
    * Unweighted graphs are not supported temporarily.
    
    """
    
    def __init__(self,
                 nodes: Matrix,
                 edges: Matrix,
                 xlabel: str = r'$x_1$',
                 ylabel: str = r'$x_2$',
                 title: str | None = None,
                 grid: bool = False) -> None:
        self.nodes = nodes
        self.edges = edges
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.grid = grid
    
    def plot(self, size: float = 250, scale: float = 10.0) -> None:
        x_min, x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        y_min, y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()
        delta_x, delta_y = (x_max - x_min) / size, (y_max - y_min) / size
        
        m, _ = self.nodes.shape

        i_upper, j_upper = np.triu_indices(m, k=1)
        weights = self.edges[i_upper, j_upper]
        valid_edges = (weights != 0) & (weights != np.inf)

        edges = np.column_stack((i_upper[valid_edges], j_upper[valid_edges]))
        weights = weights[valid_edges]

        norm = colors.Normalize(vmin=weights.min(), vmax=weights.max())
        cmap = plt.cm.plasma

        _, ax = plt.subplots()
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1],  c='black', s=20)

        for edge, weight in zip(edges, weights):
            points = self.nodes[edge]
            ax.plot(points[:, 0], points[:, 1], 
                    color=cmap(norm(weight)), linewidth=2, alpha=0.8)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge Weights')

        plt.axis('equal')
        plt.xlim(x_min - delta_x * scale, x_max + delta_x * scale)
        plt.ylim(y_min - delta_y * scale, y_max + delta_y * scale)
        
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title('Graph Plot' if self.title is None else self.title)
        
        plt.grid() if self.grid else _
        plt.tight_layout()
        plt.show()

