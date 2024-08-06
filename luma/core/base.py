from typing import Any, Dict

from luma.core.main import Luma
from luma.interface.util import ParamRange


__all__ = (
    "ModelBase",
    "ParadigmBase",
    "MetricBase",
    "VisualBase",
    "NeuralBase",
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

    Notes
    -----
    Upon the call of `set_params`, the method `check_param_ranges` is
    automatically called after resetting the parameters.

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

    def set_params(self, ignore_missing: bool = False, **kwargs) -> None:
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                if not ignore_missing:
                    print(f"'{type(self).__name__}' has no attribute '{key}'")

        self.check_param_ranges()

    def set_param_ranges(self, range_dict: Dict[str, tuple]) -> None:
        self._param_range_dict = {
            name_: ParamRange(range_, type_)
            for name_, (range_, type_) in range_dict.items()
        }

    def check_param_ranges(self) -> None:
        if not hasattr(self, "_param_range_dict"):
            return
        for name, val in self.__dict__.items():
            if name not in self._param_range_dict:
                continue
            self._param_range_dict[name].check(param_name=name, param_value=val)


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


class NeuralBase(Luma):
    """
    This class provides the foundational attributes and methods that are common
    across different types of neural network models. It is not intended to be
    instantiated directly but should be subclassed by specific types of neural
    models that implement the specific functionalities.

    Inheritances
    ------------
    `NeuralModel`

    """

    def __validate__(self) -> None:
        return super().__validate__()

    def __alloc__(self, *args, **kwargs) -> None:
        return super().__alloc__(*args, **kwargs)

    def __dealloc__(self) -> None:
        return super().__dealloc__()
