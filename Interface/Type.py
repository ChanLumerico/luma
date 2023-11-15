from typing import TypeVar
from LUMA.Interface.Super import _Estimator, _Transformer, _Evaluator, _Visualizer
from LUMA.Interface.Super import _Supervised, _Unsupervised, _Distance


Estimator = TypeVar('Estimator', bound=_Estimator)
Transformer = TypeVar('Transformer', bound=_Transformer)
Evaluator = TypeVar('Evaluator', bound=_Evaluator)
Visualizer = TypeVar('Visualizer', bound=_Visualizer)

Supervised = TypeVar('Supervised', bound=_Supervised)
Unsupervised = TypeVar('Unsupervised', bound=_Unsupervised)

Distance = TypeVar('Distance', bound=_Distance)
