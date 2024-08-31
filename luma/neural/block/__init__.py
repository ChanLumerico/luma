"""
`neural.block`
--------------
Neural blocks are modular units in computational systems, each performing 
specific operations like data processing or transformation. They are 
composed of multiple layers or steps and can be combined to build complex 
structures. These blocks simplify the design and functionality of advanced 
systems by breaking down tasks into manageable components.

"""

from dataclasses import dataclass
from typing import Literal, Tuple

from luma.core.super import Optimizer
from luma.interface.typing import ClassType
from luma.interface.util import InitUtil

from luma.neural.block import (
    incep_v1,
    incep_v2,
    incep_v4,
    incep_res_v1,
    incep_res_v2,
    mobile,
    resnet,
    se,
    standard,
    xception,
)


__all__ = (
    "ConvBlock1D",
    "ConvBlock2D",
    "ConvBlock3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConv3D",
    "DenseBlock",
    "SEBlock1D",
    "SEBlock2D",
    "SEBlock3D",
    "IncepBlock",
    "IncepResBlock",
    "ResNetBlock",
    "XceptionBlock",
    "MobileNetBlock",
)


@dataclass
class ConvBlockArgs:
    filter_size: Tuple[int, ...] | int
    activation: callable
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


class ConvBlock1D(standard._ConvBlock1D):
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
    `activation` : callable
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


class ConvBlock2D(standard._ConvBlock2D):
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
    `activation` : callable
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


class ConvBlock3D(standard._ConvBlock3D):
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
    `activation` : callable
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


class SeparableConv1D(standard._SeparableConv1D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    1-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
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
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """


class SeparableConv2D(standard._SeparableConv2D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    2-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
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
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """


class SeparableConv3D(standard._SeparableConv3D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    3-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
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
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """


@dataclass
class DenseBlockArgs:
    activation: callable
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    do_dropout: bool = True
    dropout_rate: float = 0.5
    random_state: int | None = None


class DenseBlock(standard._DenseBlock):
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
    `activation` : callable
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


class SEBlock1D(se._SEBlock1D):
    """
    Squeeze-and-Excitation(SE) block for 1-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 3D-Tensor to 2D-Matrix if set to False.

    """


class SEBlock2D(se._SEBlock2D):
    """
    Squeeze-and-Excitation(SE) block for 2-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 4D-Tensor to 2D-Matrix if set to False.

    """


class SEBlock3D(se._SEBlock3D):
    """
    Squeeze-and-Excitation(SE) block for 3-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 5D-Tensor to 2D-Matrix if set to False.

    """


@dataclass
class BaseBlockArgs:
    activation: callable
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

    class V1(incep_v1._Incep_V1_Default):
        """
        Inception block for Inception V1 network, a.k.a. GoogLeNet.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeA(incep_v2._Incep_V2_TypeA):
        """
        Inception block type-A for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeB(incep_v2._Incep_V2_TypeB):
        """
        Inception block type-B for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeC(incep_v2._Incep_V2_TypeC):
        """
        Inception block type-C for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V2_Redux(incep_v2._Incep_V2_Redux):
        """
        Inception block for grid reduction for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V4_Stem(incep_v4._Incep_V4_Stem):
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

    class V4_TypeA(incep_v4._Incep_V4_TypeA):
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

    class V4_TypeB(incep_v4._Incep_V4_TypeB):
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

    class V4_TypeC(incep_v4._Incep_V4_TypeC):
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

    class V4_ReduxA(incep_v4._Incep_V4_ReduxA):
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

    class V4_ReduxB(incep_v4._Incep_V4_ReduxB):
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

    class V1_Stem(incep_res_v1._IncepRes_V1_Stem):
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

    class V1_TypeA(incep_res_v1._IncepRes_V1_TypeA):
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

    class V1_TypeB(incep_res_v1._IncepRes_V1_TypeB):
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

    class V1_TypeC(incep_res_v1._IncepRes_V1_TypeC):
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

    class V1_Redux(incep_res_v1._IncepRes_V1_Redux):
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

    class V2_TypeA(incep_res_v2._IncepRes_V2_TypeA):
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

    class V2_TypeB(incep_res_v2._IncepRes_V2_TypeB):
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

    class V2_TypeC(incep_res_v2._IncepRes_V2_TypeC):
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

    class V2_Redux(incep_res_v2._IncepRes_V2_Redux):
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

    `ResNet-(200, 269, 1001)` :
        [2] He, Kaiming, et al. “Identity Mappings in Deep Residual
        Networks.” European Conference on Computer Vision (ECCV),
        2016, pp. 630-645.

    """

    class Basic(resnet._Basic):
        """
        Basic convolution block used in `ResNet-18` and `ResNet-34`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [1] also for additional information.
        """

    class Bottleneck(resnet._Bottleneck):
        """
        Bottleneck block used in `ResNet-(50, 101, 152)`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [1] also for additional information.
        """

    class PreActBottleneck(resnet._PreActBottleneck):
        """
        Bottleneck block with pre-activation used in
        `ResNet-(200, 269, 1001)`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [2] also for additional information.
        """


@ClassType.non_instantiable()
class XceptionBlock:
    """
    Container class for building components of XceptionNet.

    References
    ----------
    `XceptionNet` :
        [1] Chollet, François. “Xception: Deep Learning with Depthwise
        Separable Convolutions.” Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2017,
        pp. 1251-1258.

    """

    class Entry(xception._Entry):
        """
        An entry flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 728, 19, 19]
            ```
        """

    class Middle(xception._Middle):
        """
        A middle flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 728, 19, 19]
            Output: Tensor[-1, 728, 19, 19]
            ```
        """

    class Exit(xception._Exit):
        """
        An exit flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 728, 19, 19]
            Output: Tensor[-1, 1024, 9, 9]
            ```
        """


@ClassType.non_instantiable()
class MobileNetBlock:
    """
    Container class for building components of MobileNet series.

    References
    ----------
    `MobileNet V2` :
        [1] Howard, Andrew G., et al. “MobileNets: Efficient
        Convolutional Neural Networks for Mobile Vision Applications.”
        arXiv, 17 Apr. 2017, arxiv.org/abs/1704.04861.

    `MobileNet V3` :
        [2] Howard, Andrew, et al. "Searching for MobileNetV3." Proceedings
        of the IEEE/CVF International Conference on Computer Vision, 2019,
        doi:10.1109/ICCV.2019.00140.

    """

    class InvertedRes(mobile._InvertedRes):
        """
        Inverted Residual Block with depth-wise and point-wise
        convolutions used in MobileNet V2.

        Refer to the figures shown in the original paper[1].
        """

    class InvertedRes_SE(mobile._InvertedRes_SE):
        """
        Inverted Residual Block with depth-wise and point-wise
        convolutions and SE-Block attached used in MobileNet V3.

        Refer to the figures shown in the original paper[2].
        """
