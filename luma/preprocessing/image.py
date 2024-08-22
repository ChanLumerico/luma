import numpy as np
from scipy.ndimage import zoom, shift, rotate
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
                self.rng_.shuffle(X)
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
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        _, _, H, W = X.shape
        crop_H, crop_W = self.size
        start_H = (H - crop_H) // 2
        start_W = (W - crop_W) // 2

        return X[
            :,
            :,
            start_H : start_H + crop_H,
            start_W : start_W + crop_W,
        ]

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class Normalize(Transformer, Transformer.Image):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        return (X - self.mean) / self.std

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class RandomFlip(Transformer, Transformer.Image):
    def __init__(
        self,
        hor: bool = True,
        ver: bool = True,
        prob: float | tuple[float, float] = 0.5,
        random_state: int | None = None,
    ) -> None:
        self.hor = hor
        self.ver = ver

        if isinstance(prob, float):
            self.prob = (prob, prob)
        else:
            self.prob = prob

        self.rng_ = np.random.RandomState(random_state)

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        N, _, _, _ = X.shape
        for i in range(N):
            if self.hor and self.rng_.rand() < self.prob[0]:
                X[i] = np.flip(X[i], axis=-1)

            if self.ver and self.rng_.rand() < self.prob[1]:
                X[i] = np.flip(X[i], axis=-2)

        return X

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class RandomShift(Transformer, Transformer.Image):
    def __init__(
        self,
        range_: tuple[int, int],
        random_state: int | None = None,
    ) -> None:
        self.range_ = range_
        self.rng_ = np.random.RandomState(random_state)

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        N, C, _, _ = X.shape
        max_shift_H, max_shift_W = self.range_

        for i in range(N):
            shift_H = self.rng_.uniform(-max_shift_H, max_shift_H)
            shift_W = self.rng_.uniform(-max_shift_W, max_shift_W)

            for j in range(C):
                X[i, j] = shift(
                    X[i, j],
                    shift=[shift_H, shift_W],
                    mode="nearest",
                )
        return X

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class RandomCrop(Transformer, Transformer.Image):
    def __init__(
        self,
        size: tuple[int, int],
        random_state: int | None = None,
    ) -> None:
        self.size = size
        self.rng_ = np.random.RandomState(random_state)

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        N, C, H, W = X.shape
        crop_H, crop_W = self.size
        cropped = np.zeros((N, C, crop_H, crop_W), dtype=X.dtype)

        for i in range(N):
            start_H = self.rng_.randint(0, H - crop_H)
            start_W = self.rng_.randint(0, W - crop_W)

            cropped[i] = X[
                i,
                :,
                start_H : start_H + crop_H,
                start_W : start_W + crop_W,
            ]
        return cropped

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)


class RandomRotate(Transformer, Transformer.Image):
    def __init__(
        self,
        range_: tuple[float, float],
        random_state: int | None = None,
    ) -> None:
        self.range_ = range_
        self.rng_ = np.random.RandomState(random_state)

    @Tensor.force_dim(4)
    def fit_transform(self, X: Tensor) -> Tensor:
        N, C, _, _ = X.shape
        for i in range(N):
            angle = self.rng_.uniform(*self.range_)
            for j in range(C):
                X[i, j] = rotate(
                    X[i, j],
                    angle,
                    reshape=False,
                    mode="nearest",
                )
        return X

    @not_used
    def fit(self, *args) -> Self:
        return super().fit(*args)

    @not_used
    def transform(self, *args) -> Any:
        return super().transform(*args)
