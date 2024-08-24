from typing import Any, Self, override, ClassVar

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.layer import *
from luma.neural.block import SeparableConv2D


__all__ = ("_Mobile_V1", "_Mobile_V2", "_Mobile_V3")


class _Mobile_V1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: Activation.FuncType = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        width_param: float = 1.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.width_param = width_param
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = Sequential()

        self.feature_sizes_ = [
            [3, 32],
            [32, 64, 128, 128, 256, 256],
            [512] * 5,
            [1024, 1024],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "width_param": ("0<,1", None),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        sep_args = {
            **base_args,
            "do_batch_norm": True,
            "momentum": self.momentum,
        }
        wp = self.width_param

        self.model += Conv2D(3, int(32 * wp), 3, 2, **base_args)
        self.model += BatchNorm2D(int(32 * wp), self.momentum)
        self.model += self.activation()

        self.model.extend(
            SeparableConv2D(int(32 * wp), int(64 * wp), 3, **sep_args),
            self.activation(),
            SeparableConv2D(int(64 * wp), int(128 * wp), 3, 2, **sep_args),
            self.activation(),
            SeparableConv2D(int(128 * wp), int(128 * wp), 3, **sep_args),
            self.activation(),
            SeparableConv2D(int(128 * wp), int(256 * wp), 3, 2, **sep_args),
            self.activation(),
            SeparableConv2D(int(256 * wp), int(256 * wp), 3, **sep_args),
            self.activation(),
            SeparableConv2D(int(256 * wp), int(512 * wp), 3, 2, **sep_args),
            self.activation(),
            deep_add=False,
        )

        for _ in range(5):
            self.model.extend(
                SeparableConv2D(int(512 * wp), int(512 * wp), 3, **sep_args),
                self.activation(),
                deep_add=False,
            )

        self.model.extend(
            SeparableConv2D(int(512 * wp), int(1024 * wp), 3, 2, **sep_args),
            self.activation(),
            SeparableConv2D(int(1024 * wp), int(1024 * wp), 3, 2, 1, **sep_args),
            self.activation(),
            deep_add=False,
        )

        self.model += GlobalAvgPool2D()
        self.model += Flatten()
        self.model += Dense(int(1024 * wp), self.out_features, **base_args)

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_Mobile_V1, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_Mobile_V1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_Mobile_V1, self).score_nn(X, y, metric, argmax)


class _Mobile_V2(Estimator, Supervised, NeuralModel): ...


class _Mobile_V3(Estimator, Supervised, NeuralModel): ...
