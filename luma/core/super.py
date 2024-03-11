from typing import Any
from abc import (ABCMeta, 
                 abstractmethod, 
                 abstractstaticmethod)

from luma.core.base import *


__all__ = (
    'Estimator', 
    'Transformer', 
    'Optimizer',
    'Evaluator', 
    'Visualizer', 
    'Supervised', 
    'Unsupervised', 
    'Distance'
)


class Estimator(ModelBase, metaclass=ABCMeta):
    
    """
    An estimator is a mathematical model or algorithm 
    used to make predictions or estimates based on data. 
    It takes input data and learns from it to generate output predictions, 
    typically for tasks like classification or regression. 
    Estimators can be as simple as linear regression or as complex as 
    deep neural networks, and their goal is to capture patterns and 
    relationships within the data to make accurate predictions on new, unseen data.
    
    Methods
    -------
    For training:
    ```py 
        @abstractmethod
        def fit(self, *args) -> Estimator
    ```
    
    For prediction:
    ```py
        @abstractmethod
        def predict(self, *args) -> Vector
    ```
    
    For scoring
    ```py
        @abstractmethod
        def score(self, *args) -> float
    ```
    
    For setting parameters when tuning
    ```py
        def set_params(self, *args) -> None
    ```
    """
    
    class Meta:
        
        """
        An inner class of `Estimator` for meta-estimator.
        
        A meta-estimator is a type of estimator in machine learning that 
        combines or extends other estimators to improve performance or 
        functionality, such as ensemble methods or pipelines.
        """
    
    class TimeSeries:
        
        """
        An inner class of `Estimator` dedicated to time series analysis.
        
        Time series analysis encompasses methods for analyzing time series 
        data to extract meaningful statistics and characteristics. The 
        `TimeSeries` class is specialized for handling and modeling 
        time-dependent data, with a focus on identifying patterns, forecasting 
        future values, and analyzing temporal dynamics. It serves as a base 
        for models designed to work with ordered, time-stamped data, providing 
        functionalities tailored to the unique needs of time series forecasting.
        
        Notes
        -----
        - Classes under `TimeSeries` are specifically designed for time 
          series data and may not be directly compatible with classes 
          under `Meta`, which are aimed at enhancing or combining 
          estimators for improved performance or functionality. When 
          integrating `TimeSeries` models with `Meta` estimators, 
          special consideration is needed to ensure compatibility and 
          effective integration.
        
        """
    
    @abstractmethod
    def fit(self, *args) -> 'Estimator': ...
    
    @abstractmethod
    def predict(self, *args) -> Any: ...
    
    @abstractmethod
    def score(self, *args) -> float: ...
    
    def set_params(self, **kwargs) -> None:
        return super().set_params(**kwargs)


class Transformer(ModelBase, metaclass=ABCMeta):
    
    """
    A transformer (preprocessor) is a component or set of operations used to 
    prepare and clean raw data before it is fed into a machine learning model. 
    Preprocessing tasks can include data normalization, handling missing values, 
    feature scaling, one-hot encoding, and more. 
    The goal of a preprocessor is to make the data suitable for the specific 
    machine learning algorithm being used, enhancing the model's performance 
    by ensuring the data is in the right format and is free from inconsistencies 
    or noise.
    
    Methods
    -------
    For fitting:
    ```py
        @abstractmethod
        def fit(self, *args) -> Transformer
    ```
    
    For transformation:
    ```py
        @abstractmethod
        def transform(self, *args) -> Matrix
    ```
    
    For fitting and transformation at once:
    ```py
        @abstractmethod
        def fit_transform(self, *args) -> Matrix
    ```
    
    For setting parameters when tuning:
    ```py
        def set_params(self, *args) -> None
    ```
    """
    
    class Feature: 
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of feature data in a dataset.
        
        Default class for uninherited models.
        
        * `fit` method only gets parameter: `X`
        
            ```py
            def fit(self, X: Matrix) -> Transformer
            ```
        * `transform` method only gets `X` and returns transformed `X`
        
            ```py
            def transform(self, X: Matrix) -> Matrix
            ```
        """
        
    class Target:
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of target data in a dataset.
        
        * `fit` method only gets parameter: `y`

            ```py
            def fit(self, y: Vector) -> Transformer
            ```
        * `transform` method only gets `y` and returns transformed `y`
        
            ```py
            def transform(self, y: Vector) -> Matrix | Vector
            ```
        """
         
    class Both: 
        
        """
        An inner class of `Transformer` dedicated to the processing and 
        handling of both feature and target data in a dataset.
        
        * `fit` method gets two parameters: `X`, `y`
        
            ```py
            def fit(self, X: Matrix, y: Vector) -> Transformer
            ```
            To ignore `y`, replace with placeholder `_`:
            ```py
            def fit(self, X: Matrix, _ = None) -> Transformer
            ```
        * `transform` method gets `X` and `y` and returns transformed `X` and `y`
        
            ```py
            def transform(self, X: Matrix, y: Vector) -> Tuple[Matrix, Matrix | Vector]
            ```
        """
    
    @abstractmethod
    def fit(self, *args) -> 'Transformer': ...
    
    @abstractmethod
    def transform(self, *args) -> Any: ...
    
    @abstractmethod
    def fit_transform(self, *args) -> Any: ...
    
    def set_params(self, **kwargs) -> None:
        return super().set_params(**kwargs)


class Optimizer(ModelBase, metaclass=ABCMeta):
    
    """
    The Optimizer class serves as a superclass for optimization techniques in 
    the luma module, focusing on hyperparameter tuning and model optimization. 
    This class inherits from ModelBase, indicating its role in enhancing and 
    fine-tuning machine learning models. It provides an abstract base for 
    different optimization strategies, offering a standardized interface for 
    systematically exploring and evaluating different combinations of model 
    parameters.
    
    Properties
    ----------
    Get the best(optimized) estimator or transformer:
    ```py
        @property
        def best_model(self) -> Estimator | Transformer
    
    """
    
    @property
    def best_model(self) -> Estimator | Transformer: ...


class Evaluator(MetricBase, metaclass=ABCMeta):
    
    """
    Evaluators, a.k.a. metrics are quantitative measures used to assess the performance 
    and effectiveness of a machine learning model. These metrics provide insights into 
    how well a model is performing on a particular task, 
    such as classification or regression. 
    
    Methods
    -------
    For scoring:
    ```py
        @abstractstaticmethod
        def score(*args) -> float
    ```
    """
    
    @abstractstaticmethod
    def score(*args) -> float: ...


class Visualizer(VisualBase, metaclass=ABCMeta):
    
    """
    A visualizer is a tool that helps visualize and understand various aspects 
    of machine learning models, datasets, and the results the analysis. 
    Visualizers play a crucial role in simplifying the interpretation of complex 
    machine learning processes and results.
    
    Methods
    -------
    For plotting:
    ```py
        @abstractmethod
        def plot(self, *args) -> None
    ```
    """
    
    @abstractmethod
    def plot(self, *args) -> None: ...


class Supervised(ParadigmBase):
    
    """
    Supervised learning is a type of machine learning where the algorithm learns 
    to make predictions or decisions by training on a labeled dataset. 
    In this approach, the algorithm is provided with input data and corresponding 
    target labels, and it learns to map the inputs to the correct outputs. 
    """
    
    def __init__(self, *args) -> None: ...


class Unsupervised(ParadigmBase):
    
    """
    Unsupervised learning is a machine learning paradigm where the algorithm 
    is given input data without explicit target labels. Instead of making predictions, 
    the algorithm's goal is to discover hidden patterns, structures, 
    or relationships within the data.
    
    Default class for uninherited models.
    
    Properties
    ----------
    Get assigned labels:
    ```py
        @property
        def labels(self) -> Vector
    ```
    """
    
    def __init__(self, *args) -> None: ...
    
    @property
    def labels(self) -> Any:...


class Distance(MetricBase, metaclass=ABCMeta):
    
    """
    In mathematics and machine learning, distance is a measure of how much "separation" 
    or "difference" there is between two points, objects, or distributions. Different 
    types of distances serve various purposes, and they are used in different contexts.
    
    Methods
    -------
    For computing the distance:
    ```py
        @abstractstaticmethod
        def compute(*args) -> float
    ```
    """
    
    @abstractstaticmethod
    def compute(*args) -> float: ...

