from typing import Literal

from luma.core.super import Optimizer
from luma.interface.util import InitUtil

from luma.neural.layer import *


__all__ = ("ConvBlock", "DenseBlock")


class ConvBlock(Sequential):
    """
    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Structure
    ---------
    ```py
    Convolution -> Activation -> Optional[Pooling]
    ```
    Parameters
    ----------
    `in_channels` : Number of input channels
    `out_channels` : Number of output channels
    `filter_size` : Size of the convolution filter
    `activation` : Type of activation function
    `padding` : Padding method
    `optimizer` : Type of optimizer for weight updating
    `initializer` : Type of weight initializer
    `stride` : Step size for filters during convolution
    `lambda_` : L2 regularization strength
    `do_pooling` : Whether to perform pooling (default `True`)
    `pool_filter_size` : Filter size for pooling
    `pool_stride` : Step size for pooling process
    `pool_mode` : Pooling strategy

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        *,
        activation: Activation.FuncType,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        padding: Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int = None,
    ) -> None:
        super(ConvBlock, self).__init__(
            Convolution(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                initializer,
                optimizer,
                lambda_,
                random_state,
            ),
            activation,
        )
        if do_pooling:
            super(ConvBlock, self).__add__(
                Pooling(
                    pool_filter_size,
                    pool_stride,
                    pool_mode,
                )
            )

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()


class DenseBlock(Sequential):
    """
    A typical dense block in a neural network configuration often
    includes a series of fully connected (dense) layers. Each layer
    within the block connects every input neuron to every output
    neuron through learned weights. Activation functions, such as ReLU,
    are applied after each dense layer to introduce non-linear processing,
    enhancing the network's ability to learn complex patterns.
    Optionally, dropout or other regularization techniques may be
    included to reduce overfitting by randomly deactivating a portion
    of the neurons during training.

    Structure
    ---------
    ```py
    Dense -> Activation -> Optional[Dropout]
    ```
    Parameters
    ----------
    `in_features` : Number of input features
    `out_features` : Number of output features
    `activation` : Type of activation function
    `optimizer` : Type of optimizer for weight update
    `initializer` : Type of weight initializer
    `lambda_` : L2 regularization strength
    `do_dropout` : Whethter to perform dropout (default `True`)
    `dropout_rate` : Dropout rate

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        activation: Activation.FuncType,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        random_state: int = None,
    ) -> None:
        super(DenseBlock, self).__init__(
            Dense(
                in_features,
                out_features,
                initializer,
                optimizer,
                lambda_,
                random_state,
            ),
            activation,
        )
        if do_dropout:
            super(DenseBlock, self).__add__(
                Dropout(
                    dropout_rate,
                    random_state,
                ),
            )

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "dropout_rate": ("0,1", None),
            }
        )
        self.check_param_ranges()
