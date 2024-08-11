from typing import Any, Self, override, List, Optional
from dataclasses import asdict

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, TensorLike, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import ResNetBlock, BaseBlockArgs
from luma.neural.layer import (
    Convolution2D,
    Pooling2D,
    BatchNorm2D,
    Activation,
    Dense,
    Flatten,
    Sequential,
)

BasicBlock = ResNetBlock.Basic
Bottleneck = ResNetBlock.Bottleneck


def _make_layer(
    in_channels: int,
    out_channels: int,
    block: ResNetBlock,
    n_blocks: int,
    layer_num: int,
    conv_base_args: dict,
    res_base_args: dict,
    stride: int = 1,
) -> tuple[Sequential, int]:
    downsampling: Optional[Sequential] = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsampling = Sequential(
            Convolution2D(
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
        **res_base_args,
    )
    layers: list = [(f"ResNetConv{layer_num}_1", first_block)]

    in_channels = out_channels * block.expansion
    for i in range(1, n_blocks):
        new_block = (
            f"ResNetConv{layer_num}_{i + 1}",
            block(in_channels, out_channels, **res_base_args),
        )
        layers.append(new_block)

    return Sequential(*layers), in_channels


class _ResNet_18(Estimator, Supervised, NeuralModel):
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

        self.feature_sizes_ = []
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
            Convolution2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pooling2D(3, 2, "max", "same"),
        )

        self.layer_2, in_channels = _make_layer(
            64, 64, BasicBlock, 2, 2, base_args, asdict(res_args)
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
            # AdaptiveAvgPooling2D(),
            Flatten(),
            Dense(512 * BasicBlock.expansion, self.out_features, **base_args),
        )

    input_shape: tuple = (-1, 3, 224, 224)

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
