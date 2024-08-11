from typing import Any, Self, override, List, Optional

from luma.core.super import Estimator, Evaluator, Optimizer, Supervised
from luma.interface.typing import Matrix, Tensor, TensorLike, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import ResNetBlock
from luma.neural.layer import (
    Convolution2D,
    Pooling2D,
    GlobalAvgPooling2D,
    BatchNorm2D,
    Activation,
    Dropout,
    Dense,
    Flatten,
    Sequential,
)


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
        dropout_rate: float = 0.8,
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
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state

        self._in_channels = 64
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

    def _make_layer(
        self,
        out_channels: int,
        block: ResNetBlock,
        n_block: List[int],
        stride: int = 1,
        **kwargs: Any,
    ) -> Sequential:
        downsample: Optional[Sequential] = None
        if stride != 1 or self._in_channels != out_channels * block.expansion:
            downsample = Sequential(
                Convolution2D(
                    self._in_channels,
                    out_channels * block.expansion,
                    1,
                    stride,
                    **kwargs,
                ),
                BatchNorm2D(
                    out_channels * block.expansion,
                    self.momentum,
                ),
            )
        
        layers = []

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }

        self.model.extend(
            Convolution2D(3, 64, 7, 2, 3, **base_args),
            BatchNorm2D(64, self.momentum),
            self.activation(),
            Pooling2D(3, 2, "max", "same"),
        )

    input_shape: tuple = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, *args) -> Self:
        NotImplemented

    @Tensor.force_shape(input_shape)
    def predict(self, *args) -> Any:
        NotImplemented
