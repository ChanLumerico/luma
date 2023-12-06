from typing import *


class TreeNode:
    
    """
    Internal class for node used in tree-based models.
    
    Parameters
    ----------
    ``feature`` : Feature of node \n
    ``threshold`` : Threshold for split point \n
    ``left`` : Left-child node \n
    ``right`` : Right-child node \n
    ``value`` : Most popular label of leaf node
    
    """
    
    def __init__(self,
                 feature: int = None,
                 threshold: float = None,
                 left: 'TreeNode' = None,
                 right: 'TreeNode' = None,
                 value: int | float = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def isLeaf(self) -> bool:
        return self.value is not None

