from typing import Any

from luma.core.main import Luma


__all__ = (
    'ModelBase',
    'ParadigmBase',
    'MetricBase',
    'VisualBase'
)


class ModelBase(Luma):
    
    """
    The ModelBase class serves as a superclass for core machine learning components 
    involved in modeling and data transformation. It provides a foundational base 
    for classes like Estimator and Transformer, encapsulating shared functionalities 
    and properties essential for building, preparing, and processing machine 
    learning models. This class is intended to streamline the development of 
    machine learning pipelines by offering a unified interface for both model 
    creation and data preprocessing tasks.
    
    Methods
    -------
    Train an estimator or a transformer for further use:
    ```py
        def fit(self, **kwargs) -> Any
    ```
    
    Set the parameters of an estimator or a transformer:
    ```py
        def set_params(self, **kwargs) -> None
    ```
    This method iterates over the given keyword arguments and sets the
    attributes of the instance accordingly. 
    If an attribute corresponding to a given keyword does not exist, 
    a message is printed indicating that the attribute was not found.
    
    Inheritances
    ------------
    `Estimator`, `Transformer`, `Optimizer`
    
    """
    
    def __validate__(self) -> None:
        return super().__validate__()
    
    def __alloc__(self, *args, **kwargs) -> None:
        return super().__alloc__(*args, **kwargs)
    
    def __dealloc__(self) -> None:
        return super().__dealloc__()
    
    def fit(self, **kwargs) -> Any: kwargs
    
    def set_params(self, **kwargs) -> None:
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                print(f"'{type(self).__name__}' has no attribute '{key}'")


class ParadigmBase(Luma):
    
    """
    The ParadigmBase class is a superclass for different learning paradigms in 
    machine learning, such as Supervised and Unsupervised learning. This class 
    offers a common base for various types of learning approaches, providing a 
    standardized interface and shared properties relevant to different learning 
    methodologies. It helps in categorizing and organizing machine learning 
    models based on their learning strategy, facilitating the development and 
    usage of models tailored to specific types of data and tasks.
    
    Inheritances
    ------------
    `Supervised`, `Unsupervised`

    """
    
    def __validate__(self) -> None:
        return super().__validate__()
    
    def __alloc__(self, *args, **kwargs) -> None:
        return super().__alloc__(*args, **kwargs)
    
    def __dealloc__(self) -> None:
        return super().__dealloc__()


class MetricBase(Luma):
    
    """
    The MetricBase class serves as a superclass for evaluation metrics measures 
    used in machine learning. It focuses on quantitative measures to assess model 
    performance and compute similarities or differences. This class provides a 
    unified base for various types of evaluation methods and other metric calculations, 
    which are essential for effectively measuring and comparing the efficacy of 
    machine learning models and for analyzing data characteristics.
    
    Inheritances
    ------------
    `Evaluator`, `Distance`
    
    """
    
    def __validate__(self) -> None:
        return super().__validate__()
    
    def __alloc__(self, *args, **kwargs) -> None:
        return super().__alloc__(*args, **kwargs)
    
    def __dealloc__(self) -> None:
        return super().__dealloc__()


class VisualBase(Luma):
    
    """
    The VisualBase class is a superclass for visualization tools in machine 
    learning. This class offers a common framework for visualizing different 
    aspects of machine learning models, datasets, and analysis results. It plays 
    a crucial role in simplifying the interpretation and understanding of complex 
    machine learning processes, helping users to gain insights into model behavior, 
    data patterns, and performance metrics through intuitive visual representations.
    
    Inheritances
    ------------
    `Visualizer`
    
    """
    
    def __validate__(self) -> None:
        return super().__validate__()
    
    def __alloc__(self, *args, **kwargs) -> None:
        return super().__alloc__(*args, **kwargs)
    
    def __dealloc__(self) -> None:
        return super().__dealloc__()

