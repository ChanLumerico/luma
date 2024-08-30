from typing import Any, Self, override, Optional, ClassVar
from dataclasses import asdict, dataclass

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import ResNetBlock, BaseBlockArgs
from luma.neural.layer import (
    Conv2D,
    Pool2D,
    AdaptiveAvgPool2D,
    BatchNorm2D,
    Activation,
    Dense,
    Flatten,
    Sequential,
)

BasicBlock = ResNetBlock.Basic
Bottleneck = ResNetBlock.Bottleneck
PreActBottle = ResNetBlock.PreActBottleneck


def _make_layer(
    in_channels: int,
    out_channels: int,
    block: ResNetBlock,
    n_blocks: int,
    layer_num: int,
    conv_base_args: dict,
    res_base_args: dataclass,
    stride: int = 1,
) -> tuple[Sequential, int]:
    downsampling: Optional[Sequential] = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsampling = Sequential(
            Conv2D(
                in_channels,
                out_channels * block.expansion,
                1,
                stride,
                **conv_base_args,
            ),
            BatchNorm2D(out_channels * block.expansion),
        )

    first_block = block(
        in_channels,
        out_channels,
        stride,
        downsampling,
        **asdict(res_base_args),
    )
    layers: list = [(f"ResNetConv{layer_num}_1", first_block)]

    in_channels = out_channels * block.expansion
    for i in range(1, n_blocks):
        new_block = (
            f"ResNetConv{layer_num}_{i + 1}",
            block(in_channels, out_channels, **asdict(res_base_args)),
        )
        layers.append(new_block)

    return Sequential(*layers), in_channels


class _ResNet_18(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64] * 2,
            [128, 128] * 2,
            [256, 256] * 2,
            [512, 512] * 2,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, BasicBlock, 2, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, BasicBlock, 2, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, BasicBlock, 2, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, BasicBlock, 2, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(512 * BasicBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_18, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_18, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_18, self).score_nn(X, y, metric, argmax)


class _ResNet_34(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64] * 3,
            [128, 128] * 4,
            [256, 256] * 6,
            [512, 512] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, BasicBlock, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, BasicBlock, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, BasicBlock, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, BasicBlock, 3, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(512 * BasicBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_34, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_34, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_34, self).score_nn(X, y, metric, argmax)


class _ResNet_50(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 4,
            [256, 256, 1024] * 6,
            [512, 512, 2048] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, Bottleneck, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, Bottleneck, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_50, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_50, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_50, self).score_nn(X, y, metric, argmax)


class _ResNet_101(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 4,
            [256, 256, 1024] * 23,
            [512, 512, 2048] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, Bottleneck, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, Bottleneck, 23, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_101, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_101, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_101, self).score_nn(X, y, metric, argmax)


class _ResNet_152(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 8,
            [256, 256, 1024] * 36,
            [512, 512, 2048] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, Bottleneck, 8, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, Bottleneck, 36, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_152, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_152, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_152, self).score_nn(X, y, metric, argmax)


class _ResNet_200(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 24,
            [256, 256, 1024] * 36,
            [512, 512, 2048] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, PreActBottle, 24, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, PreActBottle, 36, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, PreActBottle, 3, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_200, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_200, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_200, self).score_nn(X, y, metric, argmax)


class _ResNet_269(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 30,
            [256, 256, 1024] * 48,
            [512, 512, 2048] * 8,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, PreActBottle, 30, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, PreActBottle, 48, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, PreActBottle, 8, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_269, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_269, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_269, self).score_nn(X, y, metric, argmax)


class _ResNet_1001(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
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
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 33,
            [256, 256, 1024] * 99,
            [512, 512, 2048] * 3,
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
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
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
        res_args = BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            momentum=self.momentum,
            **base_args,
        )

        self.model.extend(
            Conv2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = _make_layer(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = _make_layer(
            in_channels, 128, PreActBottle, 33, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = _make_layer(
            in_channels, 256, PreActBottle, 99, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = _make_layer(
            in_channels, 512, PreActBottle, 8, 5, base_args, res_args, stride=2
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            AdaptiveAvgPool2D((1, 1)),
            Flatten(),
            Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNet_1001, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNet_1001, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNet_1001, self).score_nn(X, y, metric, argmax)
