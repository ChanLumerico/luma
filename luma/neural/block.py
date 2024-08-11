from dataclasses import dataclass
from typing import Literal, Tuple, override
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import TensorLike, ClassType
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural import _specials


__all__ = (
    "ConvBlock1D",
    "ConvBlock2D",
    "ConvBlock3D",
    "DenseBlock",
    "IncepBlock",
    "IncepResBlock",
    "ResNetBlock",
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
class IncepBlockArgs:
    activation: Activation.FuncType
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    random_state: int | None = None


@ClassType.non_instantiable()
class IncepBlock:
    """
    Container class for various Inception blocks.

    References
    ----------
    `Inception V1, V2` :
        [1] Szegedy, Christian, et al. “Going Deeper with Convolutions.”
        Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (CVPR), 2015, pp. 1-9,
        arxiv.org/abs/1409.4842.

    `Inception V4` :
        [2] Szegedy, Christian, et al. “Inception-v4, Inception-ResNet
        and the Impact of Residual Connections on Learning.”
        Proceedings of the Thirty-First AAAI Conference on
        Artificial Intelligence (AAAI), 2017, pp. 4278-4284,
        arxiv.org/abs/1602.07261.

    """

    class V1(_specials.incep_v1._Incep_V1_Default):
        """
        Inception block for Inception V1 network, a.k.a. GoogLeNet.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeA(_specials.incep_v2._Incep_V2_TypeA):
        """
        Inception block type-A for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeB(_specials.incep_v2._Incep_V2_TypeB):
        """
        Inception block type-B for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeC(_specials.incep_v2._Incep_V2_TypeC):
        """
        Inception block type-C for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V2_Redux(_specials.incep_v2._Incep_V2_Redux):
        """
        Inception block for grid reduction for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V4_Stem(_specials.incep_v4._Incep_V4_Stem):
        """
        Inception block used in Inception V4 network stem part.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V4_TypeA(_specials.incep_v4._Incep_V4_TypeA):
        """
        Inception block type A used in Inception V4 network

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V4_TypeB(_specials.incep_v4._Incep_V4_TypeB):
        """
        Inception block type B used in Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1024, 17, 17]
            Output: Tensor[-1, 1024, 17, 17]
            ```
        """

    class V4_TypeC(_specials.incep_v4._Incep_V4_TypeC):
        """
        Inception block type C used in Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1536, 8, 8]
            Output: Tensor[-1, 1536, 8, 8]
            ```
        """

    class V4_ReduxA(_specials.incep_v4._Incep_V4_ReduxA):
        """
        Inception block type A for grid reduction used in
        Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 1024, 17, 17]
            ```
        """

    class V4_ReduxB(_specials.incep_v4._Incep_V4_ReduxB):
        """
        Inception block type B for grid reduction used in
        Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1024, 17, 17]
            Output: Tensor[-1, 1536, 8, 8]
            ```
        """


@ClassType.non_instantiable()
class IncepResBlock:
    """
    Container class for various Inception-ResNet blocks.

    References
    ----------
    `Inception-ResNet V1, V2` :
        [1] Szegedy, Christian, et al. “Inception-v4, Inception-ResNet
        and the Impact of Residual Connections on Learning.”
        Proceedings of the Thirty-First AAAI Conference on
        Artificial Intelligence (AAAI), 2017, pp. 4278-4284,
        arxiv.org/abs/1602.07261.

    """

    class V1_Stem(_specials.incep_res_v1._IncepRes_V1_Stem):
        """
        Inception block used in Inception-ResNet V1 network
        stem part.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 256, 35, 35]
            ```
        """

    class V1_TypeA(_specials.incep_res_v1._IncepRes_V1_TypeA):
        """
        Inception block type A used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 256, 35, 35]
            Output: Tensor[-1, 256, 35, 35]
            ```
        """

    class V1_TypeB(_specials.incep_res_v1._IncepRes_V1_TypeB):
        """
        Inception block type B used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 892, 17, 17]
            Output: Tensor[-1, 892, 17, 17]
            ```
        """

    class V1_TypeC(_specials.incep_res_v1._IncepRes_V1_TypeC):
        """
        Inception block type C used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1792, 8, 8]
            Output: Tensor[-1, 1792, 8, 8]
            ```
        """

    class V1_Redux(_specials.incep_res_v1._IncepRes_V1_Redux):
        """
        Inception block type B for grid reduction used in
        Inception-ResNet V1 network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 896, 17, 17]
            Output: Tensor[-1, 1792, 8, 8]
            ```
        """

    class V2_TypeA(_specials.incep_res_v2._IncepRes_V2_TypeA):
        """
        Inception block type A used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V2_TypeB(_specials.incep_res_v2._IncepRes_V2_TypeB):
        """
        Inception block type B used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1280, 17, 17]
            Output: Tensor[-1, 1280, 17, 17]
            ```
        """

    class V2_TypeC(_specials.incep_res_v2._IncepRes_V2_TypeC):
        """
        Inception block type C used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 2272, 8, 8]
            Output: Tensor[-1, 2272, 8, 8]
            ```
        """

    class V2_Redux(_specials.incep_res_v2._IncepRes_V2_Redux):
        """
        Inception block type B for grid reduction used in
        Inception-ResNet V2 network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1289, 17, 17]
            Output: Tensor[-1, 2272, 8, 8]
            ```
        """


@ClassType.non_instantiable()
class ResNetBlock:
    """
    Container class for building components of ResNet.

    References
    ----------
    `ResNet-(18, 34, 50, 101, 152)` :
        [1] He, Kaiming, et al. “Deep Residual Learning for Image
        Recognition.” Proceedings of the IEEE Conference on Computer
        Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.

    """

    class Basic(_specials.resnet._Basic):
        """
        Basic convolution block used in `ResNet-18` and `ResNet-34`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [1] also for additional information.
        """

    class Bottleneck(_specials.resnet._Bottleneck):
        """
        Bottleneck block used in `ResNet-(50, 101, 152)`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling
        `expansion` : int, default=4
            Expanding factor for the number of output channels.

        See [1] also for additional information.
        """
