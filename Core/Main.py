from typing import *
import sys


class LUMA:
    
    """
    ``LUMA``
    ========
    LUMA is a powerful and flexible Python module designed to simplify and streamline 
    various machine learning tasks. It is specifically created to enhance the ease of 
    building, training, and deploying machine learning models while offering extensive 
    customization options for data scientists and developers.

    Submodules:
    -----------

    ``LUMA.Classifier``
    -------------------
    The `LUMA.Classifier` submodule is a comprehensive toolkit for building, training,
    and evaluating classification models. It provides a wide range of classification algorithms, 
    including decision trees, support vector machines, random forests, and neural networks.

    ``LUMA.Clustering``
    -------------------
    The `LUMA.Clustering` submodule focuses on unsupervised machine learning tasks, 
    specifically clustering. It encompasses algorithms such as K-Means, hierarchical clustering, etc. 
    It simplifies cluster creation, analysis, and visualization, allowing users to gain insights 
    from their data without the need for labels.

    ``LUMA.Core``
    -------------
    The `LUMA.Core` submodule serves as the foundational backbone for the entire LUMA framework. 
    It provides essential data structures and utility functions that are used throughout the 
    LUMA ecosystem.

    ``LUMA.Interface``
    ------------------
    The `LUMA.Interface` submodule contains files that define protocols and custom data types used 
    internally within the LUMA framework. These files are not intended for direct external use but 
    play a crucial role in the functionality and communication between various LUMA components.

    ``LUMA.Metric``
    ---------------
    The `LUMA.Metric` submodule provides a rich collection of performance metrics for evaluating 
    machine learning models. It includes metrics for classification tasks like ROC-AUC, log-loss, 
    and confusion matrices. For regression tasks, it offers metrics such as mean squared error (MSE) 
    and R-squared. These metrics are essential for assessing model quality and guiding model selection.

    ``LUMA.ModelSelection``
    -----------------------
    The `LUMA.ModelSelection` submodule streamlines the process of selecting the best machine learning 
    model and optimizing hyperparameters. It offers tools for hyperparameter tuning, cross-validation, 
    and model selection, enabling users to find the optimal model configuration for their specific task.

    ``LUMA.Preprocessing``
    ----------------------
    The `LUMA.Preprocessing` submodule includes a variety of data preprocessing functions to ensure data 
    is properly prepared for machine learning tasks. It covers tasks like feature scaling, one-hot encoding, 
    handling missing values, and data splitting. Proper data preprocessing is crucial for model performance 
    and accuracy.

    ``LUMA.Reduction``
    ---------------------
    The `LUMA.Reduction` submodule specializes in dimensionality reduction techniques. It provides methods 
    for feature selection and extraction, reducing the dimensionality of high-dimensional datasets. 
    This not only improves model performance but also reduces computational time and complexity.

    ``LUMA.Regressor``
    ------------------
    The `LUMA.Regressor` submodule is tailored for regression tasks. It offers a comprehensive range of 
    regression algorithms, such as linear regression, decision tree regression, and support vector regression. 
    Additionally, it includes a suite of regression-specific evaluation metrics to assess model 
    accuracy and performance.

    ``LUMA.Visual``
    -------------------
    The `LUMA.Visual` submodule simplifies model visualization. It includes tools for plotting data,
    visualizing decision boundaries, and creating performance charts. These visualization aids help 
    users gain insights from their machine learning models and communicate results effectively.
    
    Version:
    --------
    ``alpha 1.1``
    
    """

    
    def __init__(self) -> None: ...
    
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...
    
    def __init_subclass__(cls) -> None: ...


    if sys.version_info <= (3, 9):
        print("LUMA requires Python 3.10 or more", file=sys.stderr)
