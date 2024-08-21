import numpy as np
from scipy.ndimage import zoom
from typing import Any, List, Literal, Self

from luma.interface.typing import Tensor
from luma.interface.exception import UnsupportedParameterError
from luma.extension import not_used
from luma.core.super import Transformer


__all__ = (
    "ImageTransformer",
    "Resize",
    "CenterCrop",
    "Normalize",
    "RandomFlip",
    "RandomShift",
    "RandomCrop",
    "RandomRotate",
)


class ImageTransformer(Transformer, Transformer.Image):
    def __init__(
        self,
        *trans_arr: Transformer,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.trans_arr = trans_arr
        self.shuffle = shuffle

        self.rng_ = np.random.RandomState(random_state)

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Any:
        for trans in self.trans_arr:
            if self.shuffle:
                X = self.rng_.shuffle(X)
            X = trans.fit_transform(X)

        return X
    
    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)
    
    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)

    def __len__(self) -> int:
        return len(self.trans_arr)

    def __getitem__(self, index: int) -> Transformer:
        return self.trans_arr[index]

    def __reversed__(self) -> List[Transformer]:
        return self.trans_arr[::-1]


class Resize(Transformer, Transformer.Image):
    def __init__(
        self,
        size: tuple[int, int],
        interpolation: Literal["bilinear", "nearest"] = "bilinear",
    ) -> None:
        self.size = size
        self.interpolation = interpolation

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        if self.interpolation == "nearest":
            return self._resize_nearest(X)
        elif self.interpolation == "bilinear":
            return self._resize_bilinear(X)
        else:
            raise UnsupportedParameterError(self.interpolation)

    def _resize_nearest(self, X: Tensor) -> Tensor:
        N, C, H, W = X.shape
        new_H, new_W = self.size
        scale_H = new_H / H
        scale_W = new_W / W

        X_resized = np.zeros((N, C, new_H, new_W), dtype=X.dtype)
        for i in range(N):
            for j in range(C):
                X_resized[i, j] = np.repeat(
                    np.repeat(X[i, j], scale_H, axis=0), scale_W, axis=1
                )

        return X_resized

    def _resize_bilinear(self, X: Tensor) -> Tensor:
        N, C, H, W = X.shape
        new_H, new_W = self.size
        scale_H = new_H / H
        scale_W = new_W / W

        X_resized = np.zeros((N, C, new_H, new_W), dtype=X.dtype)

        for i in range(N):
            for j in range(C):
                zoom_factor = (scale_H, scale_W)
                X_resized[i, j] = zoom(X[i, j], zoom_factor, order=1)

        return X_resized
    
    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)
    
    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class CenterCrop(Transformer, Transformer.Image):
    NotImplemented


class Normalize(Transformer, Transformer.Image):
    NotImplemented


class RandomFlip(Transformer, Transformer.Image):
    NotImplemented


class RandomShift(Transformer, Transformer.Image):
    NotImplemented


class RandomCrop(Transformer, Transformer.Image):
    NotImplemented


class RandomRotate(Transformer, Transformer.Image):
    NotImplemented
