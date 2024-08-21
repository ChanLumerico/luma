import numpy as np
from typing import Any

from luma.interface.typing import Tensor
from luma.core.super import Transformer


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

    def __len__(self) -> int:
        return len(self.trans_arr)
    
    def __getitem__(self, index: int) -> Transformer:
        return self.trans_arr[index]
    
    def __reversed__(self) -> list[Transformer]:
        return self.trans_arr[::-1]


class Resize(Transformer, Transformer.Image):
    NotImplemented


class Crop(Transformer, Transformer.Image):
    NotImplemented


class Normalize(Transformer, Transformer.Image):
    NotImplemented


class RandomFlipHorizontal(Transformer, Transformer.Image):
    NotImplemented


class RandomFlipVertical(Transformer, Transformer.Image):
    NotImplemented


class RandomShift(Transformer, Transformer.Image):
    NotImplemented
