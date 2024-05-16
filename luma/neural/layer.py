from typing import Any, List, Literal, Self, Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import TensorLike
from luma.interface.util import InitUtil, Clone
from luma.neural.base import Layer

from ._layers import (
    _act,
    _conv,
    _drop,
    _linear,
    _norm,
    _pool,
)


__all__ = (
    "Convolution1D",
    "Convolution2D",
    "Convolution3D",
    "Pooling1D",
    "Pooling2D",
    "Pooling3D",
    "Dense",
    "Dropout",
    "Dropout1D",
    "Dropout2D",
    "Dropout3D",
    "Flatten",
    "Activation",
    "BatchNorm1D",
    "BatchNorm2D",
    "BatchNorm3D",
    "LocalResponseNorm",
    "LayerNorm",
    "Sequential",
)


class Convolution1D(_conv._Conv1D):
    """
    Convolutional layer for 1-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels(filters)
    `filter_size`: int
        Length of each filter
    `stride` : int, default=1
        Step size for filters during convolution
    `padding` : {"valid", "same"}, default="same"
        Padding strategies ("valid" for no padding, "same" for zero-padding)
    `initializer` : InitStr, default=None
        Type of weight initializer
    `optimizer` : Optimizer, optional, default=None
        Optimizer for weight update
    `lambda_` : float, default=0.0
        L2-regularization strength
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

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
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            filter_size,
            stride,
            padding,
            initializer,
            optimizer,
            lambda_,
            random_state,
        )


class Convolution2D(_conv._Conv2D):
    """
    Convolutional layer for 2-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels(filters)
    `filter_size`: int
        Length of each filter
    `stride` : int, default=1
        Step size for filters during convolution
    `padding` : {"valid", "same"}, default="same"
        Padding strategies ("valid" for no padding, "same" for zero-padding)
    `initializer` : InitStr, default=None
        Type of weight initializer
    `optimizer` : Optimizer, optional, default=None
        Optimizer for weight update
    `lambda_` : float, default=0.0
        L2-regularization strength
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

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
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0,
        random_state: int = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            filter_size,
            stride,
            padding,
            initializer,
            optimizer,
            lambda_,
            random_state,
        )


class Convolution3D(_conv._Conv3D):
    """
    Convolutional layer for 3-dimensional data.

    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels(filters)
    `filter_size`: int
        Length of each filter
    `stride` : int, default=1
        Step size for filters during convolution
    `padding` : {"valid", "same"}, default="same"
        Padding strategies ("valid" for no padding, "same" for zero-padding)
    `initializer` : InitStr, default=None
        Type of weight initializer
    `optimizer` : Optimizer, optional, default=None
        Optimizer for weight update
    `lambda_` : float, default=0.0
        L2-regularization strength
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

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
        filter_size: int,
        stride: int = 1,
        padding: Literal["valid", "same"] = "same",
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0,
        random_state: int = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            filter_size,
            stride,
            padding,
            initializer,
            optimizer,
            lambda_,
            random_state,
        )


class Pooling1D(_pool._Pool1D):
    """
    Pooling layer for 1-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `filter_size` : int, default=2
        Size of the pooling filter
    `stride` : int, default=2
        Step size of the filter during pooling
    `mode` : {"max", "avg"}, default="max"
        Pooling strategy (i.e., 'max' or 'avg')

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__(filter_size, stride, mode)


class Pooling2D(_pool._Pool2D):
    """
    Pooling layer for 2-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `filter_size` : int, default=2
        Size of the pooling filter
    `stride` : int, default=2
        Step size of the filter during pooling
    `mode` : {"max", "avg"}, default="max"
        Pooling strategy (i.e., 'max' or 'avg')

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__(filter_size, stride, mode)


class Pooling3D(_pool._Pool3D):
    """
    Pooling layer for 3-dimensional data.

    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `filter_size` : int, default=2
        Size of the pooling filter
    `stride` : int, default=2
        Step size of the filter during pooling
    `mode` : {"max", "avg"}, default="max"
        Pooling strategy (i.e., 'max' or 'avg')

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self,
        filter_size: int = 2,
        stride: int = 2,
        mode: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__(filter_size, stride, mode)


class Dense(_linear._Dense):
    """
    A dense layer, also known as a fully connected layer, connects each
    neuron in one layer to every neuron in the next layer. It performs a
    linear transformation followed by a nonlinear activation function,
    enabling complex relationships between input and output. Dense layers
    are fundamental in deep learning models for learning representations from
    data. They play a crucial role in capturing intricate patterns and
    features during the training process.

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `out_features`:  int
        Number of output features
    `initializer` : InitStr, default = None
        Type of weight initializer
    `optimizer` : Optimizer, optional, default=None
        Optimizer for weight update
    `lambda_` : float, default=0.0
        L2-regularization strength
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 2D-array(`Matrix`).

        ```py
        X.shape = (batch_size, n_features)
        ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer | None = None,
        lambda_: float = 0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            initializer,
            optimizer,
            lambda_,
            random_state,
        )


class Dropout(_drop._Dropout):
    """
    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : float, default=0.5
        The fraction of input units to drop during training
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

    Notes
    -----
    - This class applies dropout for every element in the input with any shape.
    """

    def __init__(
        self, dropout_rate: float = 0.5, random_state: int | None = None
    ) -> None:
        super().__init__(dropout_rate, random_state)


class Dropout1D(_drop._Dropout1D):
    """
    Dropout layer for 1-dimensional data.

    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : float, default=0.5
        The fraction of input units to drop during training
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self, dropout_rate: float = 0.5, random_state: int | None = None
    ) -> None:
        super().__init__(dropout_rate, random_state)


class Dropout2D(_drop._Dropout2D):
    """
    Dropout layer for 2-dimensional data.

    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : float, default=0.5
        The fraction of input units to drop during training
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self, dropout_rate: float = 0.5, random_state: int | None = None
    ) -> None:
        super().__init__(dropout_rate, random_state)


class Dropout3D(_drop._Dropout3D):
    """
    Dropout layer for 3-dimensional data.

    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : float, default=0.5
        The fraction of input units to drop during training
    `random_state` : int, optional, default=None
        Seed for various random sampling processes

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self, dropout_rate: float = 0.5, random_state: int | None = None
    ) -> None:
        super().__init__(dropout_rate, random_state)


class Flatten(_linear._Flatten):
    """
    A flatten layer reshapes the input tensor into a 2D array(`Matrix`),
    collapsing all dimensions except the batch dimension.

    Notes
    -----
    - Use this class when using `Dense` layer.
        Flatten the tensor into matrix in order to feed-forward dense layer(s).
    """

    def __init__(self) -> None:
        super().__init__()


class Activation(_act._Activation):
    """
    An Activation Layer in a neural network applies a specific activation
    function to the input it receives, transforming the input to activate
    or deactivate neurons within the network. This function can be linear
    or non-linear, such as Sigmoid, ReLU, or Tanh, which helps to introduce
    non-linearity into the model, allowing it to learn complex patterns.

    Notes
    -----
    - This class is not instantiable, meaning that a solitary use of it
        is not available.

    - All the activation functions are included inside `Activation`.

    Examples
    --------
    >>> # Activation() <- Impossible
    >>> Activation.Linear()
    >>> Activation.ReLU()

    """


class BatchNorm1D(_norm._BatchNorm1D):
    """
    Batch normalization layer for 1-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `momentum` : float, default=0.9
        Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """

    def __init__(
        self, in_features: int, momentum: float = 0.9, epsilon: float = 1e-9
    ) -> None:
        super().__init__(in_features, momentum, epsilon)


class BatchNorm2D(_norm._BatchNorm2D):
    """
    Batch normalization layer for 2-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `momentum` : float, default=0.9
        Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """

    def __init__(
        self, in_features: int, momentum: float = 0.9, epsilon: float = 1e-9
    ) -> None:
        super().__init__(in_features, momentum, epsilon)


class BatchNorm3D(_norm._BatchNorm3D):
    """
    Batch normalization layer for 3-dimensional data.

    Batch normalization standardizes layer inputs across mini-batches to stabilize
    learning, accelerate convergence, and reduce sensitivity to initialization.
    It adjusts normalized outputs using learnable parameters, mitigating internal
    covariate shift in deep networks.

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `momentum` : float, default=0.9
        Momentum for updating the running averages

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """

    def __init__(
        self, in_features: int, momentum: float = 0.9, epsilon: float = 0.00001
    ) -> None:
        super().__init__(in_features, momentum, epsilon)


class LocalResponseNorm(_norm._LocalResponseNorm):
    """
    Local Response Normalization (LRN) is a technique used in neural networks
    to promote competition among neighboring feature maps. By normalizing the
    activities in local regions across channels, LRN helps to enhance generalization
    by suppressing activations that are uniformly large across the entire map and
    boosting those that are uniquely larger in a local neighborhood.

    Parameters
    ----------
    `depth` : int
        Number of adjacent channels to normalize across
    `alpha` : float, default=1e-4
        Scaling parameter for the squared sum
    `beta` : float, default=0.75
        Exponent for the normalization
    `k` : float, default=2
        Offset to avoid division by zero

    Notes
    -----
    - The input `X` must have the form of >=3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, *spatial)
        ```
    """

    def __init__(
        self,
        depth: int,
        alpha: float = 0.0001,
        beta: float = 0.75,
        k: float = 2,
    ) -> None:
        super().__init__(
            depth,
            alpha,
            beta,
            k,
        )


class LayerNorm(_norm._LayerNorm):
    """
    Layer normalization is a technique used in neural networks to normalize the
    inputs across the features for each data sample in a batch independently.
    It stabilizes the learning process by reducing the variance of the inputs
    within a layer, helping to speed up training and improve performance.

    Parameters
    ----------
    `in_shape` : int or tuple of int
        Shape of the input

    Notes
    -----
    - The input `X` must have the form of >=2D-array(`Matrix`).

        ```py
        X.shape = (batch_size, *spatial)
        ```
    """

    def __init__(
        self,
        in_shape: Tuple[int] | int,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__(in_shape, epsilon)


class Sequential(Layer):
    """
    Sequential represents a linear arrangement of layers in a neural network
    model. Each layer is added sequentially, with data flowing from one layer
    to the next in the order they are added. This organization simplifies the
    construction of neural networks, especially for straightforward architectures,
    by mirroring the logical flow of data from input through hidden layers to
    output.

    Parameters
    ----------
    `*layers` : Layer or tuple[str, Layer]
        Layers or layers with its name assigned
        (class name of the layer assigned by default)

    Methods
    -------
    For setting an optimizer of each layer:
    ```py
    def set_optimizer(self, optimizer: Optimizer) -> None
    ```
    To add additional layer:
    ```py
    def add(self, layer: Layer) -> None
    ```
    Specials
    --------
    - You can use `+` operator to add a layer or another instance
        of `Sequential`.

    - Calling its instance performs a single forwarding.

    Notes
    -----
    - Before any execution, an optimizer must be assigned.

    - For multi-class classification, the target variable `y`
        must be one-hot encoded.

    Examples
    --------
    ```py
    model = Sequential(
        ("conv_1", Convolution(3, 6, 3, activation="relu")),
        ("pool_1", Pooling(2, 2, mode="max")),
        ...,
        ("drop", Dropout(0.1)),
        ("flat", Flatten()),
        ("dense_1", Dense(384, 32, activation="relu")),
        ("dense_2", Dense(32, 10, activation="softmax")),
    )
    model.set_optimizer(AnyOptimizer())

    out = model(X, is_train=True) # model.forward(X, is_train=True)
    model.backward(d_out) # assume d_out is the gradient w.r.t. loss
    model.update()
    ```
    """

    def __init__(self, *layers: Layer | tuple[str, Layer]) -> None:
        super().__init__()
        self.layers: List[tuple[str, Layer]] = list()
        for layer in layers:
            self.add(layer)

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        self.input_ = X
        out = X
        for _, layer in self.layers:
            out = layer(out, is_train=is_train)

        return out

    def backward(self, d_out: TensorLike) -> TensorLike:
        for _, layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def update(self) -> None:
        self._check_no_optimizer()
        for _, layer in reversed(self.layers):
            layer.update()

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        self.optimizer = optimizer
        self.optimizer.set_params(**params, ignore_missing=True)

        for _, layer in self.layers:
            cloned_opt = Clone(self.optimizer).get
            if hasattr(layer, "set_optimizer"):
                layer.set_optimizer(cloned_opt)
            else:
                layer.optimizer = cloned_opt

    def _check_no_optimizer(self) -> None:
        if self.optimizer is None:
            raise RuntimeError(
                f"'{self}' has no optimizer! "
                + f"Call '{self}().set_optimizer' to assign an optimizer."
            )

    def add(self, layer: Layer | tuple[str, Layer]) -> None:
        if not isinstance(layer, tuple):
            layer = (str(layer), layer)
        self.layers.append(layer)

        if self.optimizer is not None:
            self.set_optimizer(self.optimizer)

    @override
    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        for _, layer in self.layers:
            w_, b_ = layer.param_size
            w_size += w_
            b_size += b_

        return w_size, b_size

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        for _, layer in self.layers:
            in_shape = layer.out_shape(in_shape)
        return in_shape

    def __add__(self, other: Layer | tuple[str, Layer]) -> Self:
        if isinstance(other, (Layer, tuple)):
            self.add(other)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self).__name__, type(other).__name__
                )
            )

        return self

    def __getitem__(self, index: int) -> Tuple[str, Layer]:
        return self.layers[index]
