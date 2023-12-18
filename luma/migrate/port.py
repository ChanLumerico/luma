from typing import Literal
import pickle
import os

from luma.interface.super import Estimator, Transformer
from luma.interface.exception import UnsupportedParameterError
from luma.interface.exception import ModelExtensionError

MODEL_EXTENSION = '.luma'


__all__ = ['ModelPorter']


class ModelPorter:
    def __init__(self) -> None:
        self._import_filename: str
        self._export_filename: str
        self._model_in: Estimator | Transformer
        self._model_out: Estimator | Transformer
    
    def save(self, 
             model: Estimator | Transformer, 
             filename: str | Literal['auto'] = 'auto',
             replace: bool = False) -> str:
        self._model_out = model
        self._export_filename = filename
        
        if not self._model_out._fitted:
            print(f"[ModelPorter] {self._model_out.__class__.__name__}",
                  "is not fitted! Saving unfitted model.")
            
        if not isinstance(self._export_filename, str): 
            raise UnsupportedParameterError(self._export_filename)
        
        if filename == 'auto': 
            self._export_filename = self._model_out.__class__.__name__
        if MODEL_EXTENSION not in filename: 
            self._export_filename += MODEL_EXTENSION
        
        if os.path.exists(self._export_filename) and not replace:
            raise FileExistsError(f"'{self._export_filename}' already exists!"
                                  + " Change 'replace' to 'True' to override.")

        with open(self._export_filename, 'wb') as file_out:
            pickle.dump(model, file_out)
        
        return self._export_filename
    
    def load(self, filename: str) -> Estimator | Transformer:
        self._import_filename = filename
        if self._import_filename[-5:] != MODEL_EXTENSION:
            raise ModelExtensionError(self._import_filename)
        
        with open(self._import_filename, 'rb') as file_in:
            self._model_in = pickle.load(file_in)

        return self._model_in

