from typing import Literal
import pickle
import os

from luma.core.super import Estimator, Transformer
from luma.interface.exception import ModelExtensionError, UnsupportedParameterError

MODEL_EXTENSION = ".luma"


__all__ = "ModelPorter"


class ModelPorter:
    """
    A utility class for exporting and importing machine learning models,
    specifically designed to handle `Estimator` and `Transformer` instances
    within the `luma` framework. It supports saving models to disk with
    automatic naming and enforcing model extension, as well as loading models
    ensuring the correct file extension is used.

    Parameters
    ----------
    `model` : Estimator or Transformer
        A model to port
    `path` : str
        Filepath of the model (both for import and export)
    `filename` : str or {"auto"}, default="auto"
        Filename of the model
    `replace` : bool, default=False
        Whether to replace the existing original file

    Methods
    -------
    To save(export) models:
    ```py
    def save(self, model, path, filename, replace) -> str
    ```
    It saves a given model to the specified path with an optional filename.
    If the model is not yet fitted, a warning will be printed. The method
    enforces the use of the correct model file extension and handles file
    existence checks based on the `replace` parameter.

    To load(import) models:
    ```py
    def load(self, filepath) -> Estimator | Transformer
    ```
    Loads a model from the given file path, ensuring that the file has the
    correct model extension(`.luma`) before proceeding.

    Raises
    ------
    - `ModelExtensionError` : If the file does not have `.luma` file extension.

    Examples
    --------
    ```py
    model = AnyModel()
    port = ModelPorter()

    model_path = port.save(model=model, path='...', filename='auto') # Export
    new_model = port.load(filepath=model_path) # Import
    ```
    """

    def __init__(self) -> None:
        self._import_filepath = None
        self._export_filepath = None
        self._model_in: Estimator | Transformer = None
        self._model_out: Estimator | Transformer = None

    def save(
        self,
        model: Estimator | Transformer,
        path: str,
        filename: str | Literal["auto"] = "auto",
        replace: bool = False,
    ) -> str:
        if not isinstance(model, (Estimator, Transformer)):
            raise UnsupportedParameterError(model)
        self._model_out = model

        if not self._model_out._fitted:
            print(
                f"[ModelPorter] {type(self._model_out).__name__}",
                "is not fitted! Saving unfitted model.",
            )

        path += "/" if path[-1] != "/" else ""
        self._export_filepath = path

        if filename == "auto":
            self._export_filepath += type(self._model_out).__name__
        else:
            self._export_filepath += filename

        if MODEL_EXTENSION not in filename:
            self._export_filepath += MODEL_EXTENSION

        if os.path.exists(self._export_filepath) and not replace:
            raise FileExistsError(
                f"'{self._export_filepath}' already exists!"
                + " Change 'replace' to 'True' to override."
            )

        with open(self._export_filepath, "wb") as file_out:
            pickle.dump(model, file_out)

        return self._export_filepath

    def load(self, filepath: str) -> Estimator | Transformer:
        self._import_filepath = filepath
        if self._import_filepath[-5:] != MODEL_EXTENSION:
            raise ModelExtensionError(self._import_filepath)

        with open(self._import_filepath, "rb") as file_in:
            self._model_in = pickle.load(file_in)

        return self._model_in
