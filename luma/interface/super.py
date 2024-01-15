from abc import *
from typing import Any

from luma.core.main import LUMA


__all__ = (
    'Estimator', 
    'Transformer', 
    'Evaluator', 
    'Visualizer', 
    'Supervised', 
    'Unsupervised', 
    'Distance'
)


class Estimator(LUMA, metaclass=ABCMeta):
    
    """
    An estimator is a mathematical model or algorithm 
    used to make predictions or estimates based on data. 
    It takes input data and learns from it to generate output predictions, 
    typically for tasks like classification or regression. 
    Estimators can be as simple as linear regression or as complex as 
    deep neural networks, and their goal is to capture patterns and 
    relationships within the data to make accurate predictions on new, unseen data.
    
    Example
    -------
    >>> model = AnyEstimator(*args)
    >>> model.fit(X_train, y_train)
    >>> prediction = model.predict(X_test)
    
    """
    
    @abstractmethod
    def fit(self, *args) -> 'Estimator': ...
    
    @abstractmethod
    def predict(self, *args) -> ...: ...
    
    @abstractmethod
    def score(self, *args) -> float: ...
    
    @abstractmethod
    def set_params(self, *args) -> None: ...


class Transformer(LUMA, metaclass=ABCMeta):
    
    """
    A transformer (preprocessor) is a component or set of operations used to prepare and 
    clean raw data before it is fed into a machine learning model. 
    Preprocessing tasks can include data normalization, handling missing values, 
    feature scaling, one-hot encoding, and more. 
    The goal of a preprocessor is to make the data suitable for the specific 
    machine learning algorithm being used, enhancing the model's performance 
    by ensuring the data is in the right format and is free from inconsistencies or noise.
    
    Example
    -------
    >>> trans = AnyTransformer()
    >>> trans.fit(data)
    >>> new_data = trans.transform(data)
    
    To fit and transform at the same time,
    >>> new_data = trans.fit_transform(data)
    
    """
    
    class Feature: 
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of feature data in a dataset.
        
        Default class for uninherited models.
        """
        
    class Target:
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of target data in a dataset.
        """
         
    class Both: 
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of both feature and target data in a dataset.
        """
    
    @abstractmethod
    def fit(self, *args) -> 'Transformer': ...
    
    @abstractmethod
    def transform(self, *args) -> Any: ...
    
    @abstractmethod
    def fit_transform(self, *args) -> Any: ...
    
    @abstractmethod
    def set_params(self, *args) -> None: ...


class Evaluator(LUMA, metaclass=ABCMeta):
    
    """
    Evaluators, a.k.a. metrics are quantitative measures used to assess the performance 
    and effectiveness of a machine learning model. These metrics provide insights into 
    how well a model is performing on a particular task, 
    such as classification or regression. 
    
    Example
    -------
    >>> metric = AnyEvaluator()
    >>> score = metric.compute(y_true=target, y_pred=predictions)
    
    """
    
    @abstractstaticmethod
    def compute(*args) -> float: ...


class Visualizer(LUMA, metaclass=ABCMeta):
    
    """
    A visualizer is a tool that helps visualize and understand various aspects 
    of machine learning models, datasets, and the results the analysis. 
    Visualizers play a crucial role in simplifying the interpretation of complex 
    machine learning processes and results.
    
    Example
    -------
    >>> plotter = AnyVisualizer()
    >>> plotter.plot()
    
    """
    
    @abstractmethod
    def plot(self, *args) -> None: ...


class Supervised(LUMA):
    
    """
    Supervised learning is a type of machine learning where the algorithm learns 
    to make predictions or decisions by training on a labeled dataset. 
    In this approach, the algorithm is provided with input data and corresponding 
    target labels, and it learns to map the inputs to the correct outputs. 
    """
    
    def __init__(self, *args) -> None: ...


class Unsupervised(LUMA):
    
    """
    Unsupervised learning is a machine learning paradigm where the algorithm 
    is given input data without explicit target labels. Instead of making predictions, 
    the algorithm's goal is to discover hidden patterns, structures, 
    or relationships within the data.
    
    Default class for uninherited models.
    """
    
    def __init__(self, *args) -> None: ...
    
    @property
    def labels(self) -> Any:...


class Distance(LUMA, metaclass=ABCMeta):
    
    """
    In mathematics and machine learning, distance is a measure of how much "separation" 
    or "difference" there is between two points, objects, or distributions. Different 
    types of distances serve various purposes, and they are used in different contexts.
    
    Example
    -------
    >>> metric = AnyDistance()
    >>> dist = metric.distance(x, y)
    
    """
    
    @abstractstaticmethod
    def distance(*args) -> float: ...

