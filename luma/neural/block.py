from dataclasses import dataclass
from typing import Literal, Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *

from luma.neural import _specials


__all__ = (
    "ConvBlock1D",
    "ConvBlock2D",
    "ConvBlock3D",
    "DenseBlock",
    "InceptionBlock",
    "InceptionBlockV2A",
    "InceptionBlockV2B",
    "InceptionBlockV2C",
    "InceptionBlockV2R",
    "InceptionBlockV4S",
    "InceptionBlockV4A",
    "InceptionBlockV4B",
    "InceptionBlockV4C",
    # "InceptionBlockV4RA",
    # "InceptionBlockV4RB"
)


@dataclass
class ConvBlockArgs:
    filter_size: Tuple[int, ...] | int
    activation: Activation.FuncType
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    padding: Tuple[int, ...] | int | Literal["same", "valid"] = "same"
    stride: int = 1
    lambda_: float = 0.0
    do_batch_norm: bool = True
    momentum: float = 0.9
    do_pooling: bool = True
    pool_filter_size: int = 2
    pool_stride: int = 2
    pool_mode: Literal["max", "avg"] = "max"
    random_state: int | None = None


class ConvBlock1D(Sequential):
    """
    Convolutional block for 1-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : FuncType
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int] | int,
        activation: Activation.FuncType,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(ConvBlock1D, self).__init__(
            Convolution1D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(ConvBlock1D, self).__add__(
                BatchNorm1D(
                    out_channels,
                    momentum,
                )
            )
        super(ConvBlock1D, self).__add__(
            activation(),
        )
        if do_pooling:
            super(ConvBlock1D, self).__add__(
                Pooling1D(
                    pool_filter_size,
                    pool_stride,
                    pool_mode,
                )
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class ConvBlock2D(Sequential):
    """
    Convolutional block for 2-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : FuncType
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int,
        activation: Activation.FuncType,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int, int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(ConvBlock2D, self).__init__(
            Convolution2D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(ConvBlock2D, self).__add__(
                BatchNorm2D(
                    out_channels,
                    momentum,
                )
            )
        super(ConvBlock2D, self).__add__(
            activation(),
        )
        if do_pooling:
            super(ConvBlock2D, self).__add__(
                Pooling2D(
                    pool_filter_size,
                    pool_stride,
                    pool_mode,
                )
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class ConvBlock3D(Sequential):
    """
    Convolutional block for 3-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : FuncType
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int, int] | int,
        activation: Activation.FuncType,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        padding: Tuple[int, int, int] | int | Literal["same", "valid"] = "same",
        stride: int = 1,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

        super(ConvBlock3D, self).__init__(
            Convolution3D(
                in_channels,
                out_channels,
                filter_size,
                stride,
                padding,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(ConvBlock3D, self).__add__(
                BatchNorm3D(
                    out_channels,
                    momentum,
                )
            )
        super(ConvBlock3D, self).__add__(
            activation(),
        )
        if do_pooling:
            super(ConvBlock3D, self).__add__(
                Pooling3D(
                    pool_filter_size,
                    pool_stride,
                    pool_mode,
                )
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)


@dataclass
class DenseBlockArgs:
    activation: Activation.FuncType
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    do_dropout: bool = True
    dropout_rate: float = 0.5
    random_state: int | None = None


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

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `out_features` : int
        Number of output features
    `activation` : FuncType
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_dropout` : bool, default=True
        Whethter to perform dropout
    `dropout_rate` : float, default=0.5
        Dropout rate

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation.FuncType,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: float = True,
        momentum: float = 0.9,
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "dropout_rate": ("0,1", None),
            }
        )
        self.check_param_ranges()

        super(DenseBlock, self).__init__(
            Dense(
                in_features,
                out_features,
                **basic_args,
            )
        )
        if do_batch_norm:
            super(DenseBlock, self).__add__(
                BatchNorm1D(
                    1,
                    momentum,
                )
            )
        super(DenseBlock, self).__add__(
            activation(),
        )
        if do_dropout:
            super(DenseBlock, self).__add__(
                Dropout(
                    dropout_rate,
                    random_state,
                ),
            )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        self.input_ = X
        out = X
        for _, layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                out = layer(out[:, np.newaxis, :], is_train=is_train).squeeze()
                continue
            out = layer(out, is_train=is_train)

        self.out_shape = out.shape
        return out

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        for _, layer in reversed(self.layers):
            if isinstance(layer, BatchNorm1D):
                d_out = layer.backward(d_out[:, np.newaxis, :]).squeeze()
                continue
            d_out = layer.backward(d_out)
        return d_out


@dataclass
class InceptionBlockArgs:
    activation: Activation.FuncType
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    random_state: int | None = None


class InceptionBlock(Sequential):
    """
    Inception block for neural networks.

    An inception block allows for multiple convolutional operations to be
    performed in parallel. This structure is inspired by the Inception modules
    of Google's Inception network, and it concatenates the outputs of different
    convolutions to capture rich and varied features from input data.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_1x1` : int
        Number of output channels for the 1x1 convolution.
    `red_3x3` : int
        Number of output channels for the dimension reduction before
        the 3x3 convolution.
    `out_3x3` : int
        Number of output channels for the 3x3 convolution.
    `red_5x5` : int
        Number of output channels for the dimension reduction before
        the 5x5 convolution.
    `out_5x5` : int
        Number of output channels for the 5x5 convolution.
    `out_pool` : int
        Number of output channels for the 1x1 convolution after max pooling.
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        out_pool: int,
        activation: Activation.FuncType = Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = False,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.out_1x1 = out_1x1
        self.out_3x3 = out_3x3
        self.out_5x5 = out_5x5
        self.out_pool = out_pool

        basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_1x1": ("0<,+inf", int),
                "red_3x3": ("0<,+inf", int),
                "out_3x3": ("0<,+inf", int),
                "red_5x5": ("0<,+inf", int),
                "out_5x5": ("0<,+inf", int),
                "out_pool": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
                "momentum": ("0,1", None),
            }
        )
        self.check_param_ranges()

        self.branch_1x1 = Sequential(
            Convolution2D(in_channels, out_1x1, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_1x1, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_3x3 = Sequential(
            Convolution2D(in_channels, red_3x3, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_3x3, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_3x3, out_3x3, 3, 1, "same", **basic_args),
            BatchNorm2D(out_3x3, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_5x5 = Sequential(
            Convolution2D(in_channels, red_5x5, 1, 1, "valid", **basic_args),
            BatchNorm2D(red_5x5, momentum) if do_batch_norm else None,
            activation(),
            Convolution2D(red_5x5, out_5x5, 5, 1, 2, **basic_args),
            BatchNorm2D(out_5x5, momentum) if do_batch_norm else None,
            activation(),
        )

        self.branch_pool = Sequential(
            Pooling2D(3, 1, "max", "same"),
            Convolution2D(in_channels, out_pool, 1, 1, "valid", **basic_args),
            BatchNorm2D(out_pool, momentum) if do_batch_norm else None,
            activation(),
        )

        super(InceptionBlock, self).__init__()
        self.layers = [
            *self.branch_1x1.layers,
            *self.branch_3x3.layers,
            *self.branch_5x5.layers,
            *self.branch_pool.layers,
        ]

        if optimizer is not None:
            self.set_optimizer(optimizer)

    @override
    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        branch_1x1 = self.branch_1x1(X, is_train)
        branch_3x3 = self.branch_3x3(X, is_train)
        branch_5x5 = self.branch_5x5(X, is_train)
        branch_pool = self.branch_pool(X, is_train)

        out = np.concatenate(
            (branch_1x1, branch_3x3, branch_5x5, branch_pool),
            axis=1,
        )
        return out

    @override
    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        d_out_1x1 = d_out[:, : self.out_1x1, ...]
        d_out_3x3 = d_out[:, self.out_1x1 : self.out_1x1 + self.out_3x3, ...]
        d_out_5x5 = d_out[
            :,
            self.out_1x1 + self.out_3x3 : self.out_1x1 + self.out_3x3 + self.out_5x5,
            ...,
        ]
        d_out_pool = d_out[:, -self.out_pool :, ...]

        dX_1x1 = self.branch_1x1.backward(d_out_1x1)
        dX_3x3 = self.branch_3x3.backward(d_out_3x3)
        dX_5x5 = self.branch_5x5.backward(d_out_5x5)
        dX_pool = self.branch_pool.backward(d_out_pool)

        self.dX = dX_1x1 + dX_3x3 + dX_5x5 + dX_pool
        return self.dX

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_1x1 + self.out_3x3 + self.out_5x5 + self.out_pool,
            height,
            width,
        )


class InceptionBlockV2A(_specials.incep_v2._Incep_V2_TypeA):
    """
    Inception block type-A for Inception V2 network.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_1x1` : int
        Number of output channels for the 1x1 convolution.
    `red_3x3` : int
        Number of output channels for the dimension reduction before
        the 3x3 convolution.
    `out_3x3` : int
        Number of output channels for the 3x3 convolution.
    `red_3x3_db` : int
        Number of output channels for the dimension reduction before
        the double 3x3 convolution.
    `out_3x3_db` : tuple of int
        Number of output channels for the double 3x3 convolutions.
    `out_pool` : int
        Number of output channels for the 1x1 convolution after max pooling.
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    """


class InceptionBlockV2B(_specials.incep_v2._Incep_V2_TypeB):
    """
    Inception block type-B for Inception V2 network.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_1x1` : int
        Number of output channels for the 1x1 convolution.
    `red_7x7` : int
        Number of output channels for the dimension reduction before
        the 7x7 convolution.
    `out_7x7` : int
        Number of output channels for the factorized 7x7 convolution.
    `red_7x7_db` : int
        Number of output channels for the dimension reduction before
        the factorized double 7x7 convolution.
    `out_7x7_db` : tuple of int
        Number of output channels for the factorized double 7x7 convolutions.
    `out_pool` : int
        Number of output channels for the 1x1 convolution after max pooling.
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    """


class InceptionBlockV2C(_specials.incep_v2._Incep_V2_TypeC):
    """
    Inception block type-C for Inception V2 network.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_1x1` : int
        Number of output channels for the 1x1 convolution.
    `red_1x3_3x1` : int
        Number of output channels for the dimension reduction before
        the 1x3 and 3x1 convolution sub-branch.
    `out_1x3_3x1` : tuple of int
        Number of output channels for the 1x3 and 3x1 convolution sub-branch.
    `red_3x3` : int
        Number of output channels for the dimension reduction before
        the 3x3 convolution.
    `out_3x3` : int
        Number of output channels for the 3x3 convolution.
    `out_1x3_3x1_after` tuple of int
        Number of output channels for the 1x3 and 3x1 convolution sub-branch
        after 3x3 convolution.
    `out_pool` : int
        Number of output channels for the 1x1 convolution after max pooling.
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    """


class InceptionBlockV2R(_specials.incep_v2._Incep_V2_Redux):
    """
    Inception block for grid reduction for Inception V2 network.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels.
    `out_1x1` : int
        Number of output channels for the 1x1 convolution.
    `red_3x3` : int
        Number of output channels for the dimension reduction before
        the 3x3 convolution.
    `out_3x3` : int
        Number of output channels for the 3x3 convolution.
    `red_3x3_db` : int
        Number of output channels for the dimension reduction before
        the double 3x3 convolution.
    `out_3x3_db` : tuple of int
        Number of output channels for the double 3x3 convolutions.
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    """


class InceptionBlockV4S(_specials.incep_v4._Incep_V4_Stem):
    """
    Inception block used in Inception V4 network stem part.

    Parameters
    ----------
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    - This block has fixed shape of input and ouput tensors.

        ```py
        Input: Tensor[-1, 3, 299, 299]
        Output: Tensor[-1, 384, 35, 35]
        ```
    """


class InceptionBlockV4A(_specials.incep_v4._Incep_V4_TypeA):
    """
    Inception block type A used in Inception V4 network

    Parameters
    ----------
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    - This block has fixed shape of input and ouput tensors.

        ```py
        Input: Tensor[-1, 384, 35, 35]
        Output: Tensor[-1, 384, 35, 35]
        ```
    """


class InceptionBlockV4B(_specials.incep_v4._Incep_V4_TypeB):
    """
    Inception block type B used in Inception V4 network.

    Parameters
    ----------
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    - This block has fixed shape of input and ouput tensors.

        ```py
        Input: Tensor[-1, 1024, 17, 17]
        Output: Tensor[-1, 1024, 17, 17]
        ```
    """


class InceptionBlockV4C(_specials.incep_v4._Incep_V4_TypeC):
    """
    Inception block type C used in Inception V4 network.

    Parameters
    ----------
    `activation` : FuncType, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of a 4D-array (`Tensor`).

        ```py
        X.shape = (batch_size, height, width, channels)
        ```
    - This block has fixed shape of input and ouput tensors.

        ```py
        Input: Tensor[-1, 1536, 8, 8]
        Output: Tensor[-1, 1536, 8, 8]
        ```
    """
