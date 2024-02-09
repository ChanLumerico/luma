from typing import Dict, Literal, Tuple, List, TypeVar, Any

from luma.core.super import Estimator, Transformer, Evaluator
from luma.core.super import Supervised
from luma.interface.util import Matrix
from luma.interface.exception import NotFittedError, UnsupportedParameterError
from luma.metric.classification import Accuracy
from luma.metric.regression import MeanSquaredError

MT = TypeVar('MT', bound=Estimator | Transformer)


__all__ = (
    'Pipeline'
)


class Pipeline(Estimator, Estimator.Meta):
    
    """
    A pipeline is a sequence of steps to process data and build a model. 
    It includes data collection and preprocessing, model selection and training, 
    evaluation, and deployment. This systematic approach ensures efficiency 
    and accuracy in model development. Pipelines are often automated for 
    scalability and reproducibility in real-world applications.
    
    Parameters
    ----------
    `models` : List of models \n
    You can assign labels to each model by encapsulating the label and the model
    inside a tuple. \n
    Otherwise, the name of the model is automatically assigned by default.
    
    `param_dict` : Dictionary of parameters for each models \n
    You must specify the name(or label) of the model and its parameter name
    in the key of the dictionary. \n
    e.g. `{'model_name__param_name': value}`
    
    Examples
    --------
    >>> pipe = Pipeline(
            models=[
                ('trans_1', Transformer()),
                ('trans_2', Transformer()),
                ...,
                ('est', Estimator())
            ],
            param_dict={
                'trans_1__param': List[Any],
                'trans_2__param': List[Any],
                ...,
                'est__param': List[Any]
            }
        )
    >>> pipe.fit(X_train, y_train)
    >>> y_pred = pipe.predict(X_test)
    
    Properties
    ----------
    Getting list of transformers:
        ```py
        @property
        def transformers(self) -> List[Transformer]
        ```
    
    Getting final estimator:
        ```py
        @property
        def estimator(self) -> Estimator
        ```
    
    Getting transformed data:
        ```py
        @property
        def transformed_data(self) -> Tuple[Matrix, Matrix]
        ```
    
    Notes
    -----
    * To use `Pipeline` with visual methods of `luma.visual`, make sure to
    transform data using `pipe.transform()` if the pipeline sequence contains
    transformers
    
    * More than one estimator might cause procedural failure
    * Not all the models are compatible with `Pipeline`
    * Type `MT` is a generic type for `Estimator` and `Transformer`
    
    """
    
    def __init__(self,
                 models: List[Tuple[str, MT]] | List[MT],
                 param_dict: Dict[str, Any] = dict(),
                 verbose: bool = False) -> None:
        self.models: Dict[MT] = dict()
        self.param_dict = param_dict
        self.verbose = verbose
        self._X: Matrix
        self._y: Matrix
        self._fitted = False
        
        for model in models:
            if isinstance(model, tuple):
                _name, _model = model
                self.models[_name] = _model
            else: self.models[type(model).__name__] = model
        
        self.set_params(param_dict)
    
    def fit(self, X: Matrix, y: Matrix) -> 'Pipeline':
        self._X, self._y = self.fit_transform(X, y)
        model = self.estimator
        if hasattr(model, 'verbose'):
            model.verbose = self.verbose
        
        data = [self._X]
        if isinstance(model, Supervised): data.append(self._y)
        model.fit(*data)
        
        self._fitted = True
        return self
        
    def transform(self, X: Matrix, y: Matrix) -> Tuple[Matrix, Matrix]:
        if not self._fitted: raise NotFittedError(self)
        X_trans, y_trans = X, y
        for model in self.transformers:
            if isinstance(model, Transformer.Target):
                y_trans = model.transform(y_trans)
            elif isinstance(model, Transformer.Both):
                X_trans, y_trans = model.transform(X_trans, y_trans)
            else:
                X_trans = model.transform(X_trans)

        return X_trans, y_trans
    
    def predict(self, X: Matrix, transform: bool = True):
        if not self._fitted: raise NotFittedError(self)
        for model in self.transformers:
            if not transform: break
            if isinstance(model, Transformer.Target | Transformer.Both):
                continue
            
            X = model.transform(X)
        
        return self.estimator.predict(X)
    
    def fit_transform(self, X: Matrix, y: Matrix) -> Tuple[Matrix, Matrix]:
        X_trans, y_trans = X, y
        for model in self.transformers:
            data = [X_trans]
            if hasattr(model, 'verbose'):
                model.verbose = self.verbose
            
            if isinstance(model, Supervised):
                data.append(y_trans)
            if isinstance(model, Transformer.Target):
                y_trans = model.fit_transform(y_trans)
            elif isinstance(model, Transformer.Both):
                X_trans, y_trans = model.fit_transform(*data)
            else:
                X_trans = model.fit_transform(*data)

        return X_trans, y_trans
    
    def fit_predict(self, X: Matrix, y: Matrix) -> Matrix:
        self.fit(X, y)
        return self.predict(X)
    
    def set_params(self, param_dict: Dict[str, Any]) -> None:
        for param_name, value in param_dict.items():
            try: _name, _param = param_name.split('__')
            except: raise UnsupportedParameterError(param_name)
            self.models[_name].set_params(**{_param: value})
    
    def score(self, X: Matrix, y: Matrix, 
              metric: Evaluator | Literal['default'] = 'default') -> float:
        model = self.estimator
        pkg = model.__module__.split('.')[1]
        if metric == 'default':
            if pkg == 'classifier': metric = Accuracy
            elif pkg == 'regressor': metric = MeanSquaredError
        
        X_pred = self.predict(X)
        return metric.score(y_true=y, y_pred=X_pred)
    
    def dump(self) -> None:
        print(f"Configuration of a pipeline:")
        for name, model in self.models.items():
            print(f"\n[{type(model).__name__} as '{name}']")
            for attr in model.__dict__.items():
                print(*attr, sep=': ')
    
    @property
    def transformers(self) -> List[Transformer]:
        _trans = []
        for model in self.models.values():
            if isinstance(model, Transformer):
                _trans.append(model)

        return _trans

    @property
    def estimator(self) -> Estimator:
        for model in self.models.values():
            if isinstance(model, Estimator):
                return model
    
    @property
    def transformed_data(self) -> Tuple[Matrix, Matrix]:
        return self._X, self._y
    
    def __getitem__(self, index: int) -> MT:
        for i, model in enumerate(self.models.values()):
            if index == i: return model
        else: raise IndexError('Model index out of bounds!')
    
    def __setitem__(self, label: str, model: MT):
        self.models[label] = model

