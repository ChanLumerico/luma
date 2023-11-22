from typing import *
from typing_extensions import Self


class _Estimator(Protocol):
    
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
    >>> model = AnyEstimator(**params)
    >>> model.fit(X_train, y_train)
    >>> prediction = model.predict(X_test)
    
    """
    
    def fit(self, *data: Tuple[Any]) -> Self: ...
    
    def predict(self, *data: Tuple[Any]) -> Any: ...


class _Transformer(Protocol):
    
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
    
    def fit(self, *data: Tuple[Any]) -> Self: ...
    
    def transform(self, *data: Tuple[Any]) -> Any: ...
    
    def fit_transform(self, *data: Tuple[Any]) -> Any: ...


class _Evaluator(Protocol):
    
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
    
    def compute(self, *data: Tuple[Any]) -> float: ...


class _Visualizer(Protocol):
    
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
    
    def plot(self, *params: Tuple[Any]) -> None: ...


class _Supervised:
    
    """
    Supervised learning is a type of machine learning where the algorithm learns 
    to make predictions or decisions by training on a labeled dataset. 
    In this approach, the algorithm is provided with input data and corresponding 
    target labels, and it learns to map the inputs to the correct outputs. 
    """
    
    def __init__(self, *params: Tuple[Any]) -> None: ...


class _Unsupervised:
    
    """
    Unsupervised learning is a machine learning paradigm where the algorithm 
    is given input data without explicit target labels. Instead of making predictions, 
    the algorithm's goal is to discover hidden patterns, structures, 
    or relationships within the data.
    """
    
    def __init__(self, *params: Tuple[Any]) -> None: ...


class _Distance:
    
    """
    In mathematics and machine learning, distance is a measure of how much "separation" 
    or "difference" there is between two points, objects, or distributions. Different 
    types of distances serve various purposes, and they are used in different contexts. 
    """
    
    def __init__(self, *param: Any) -> None: ...

