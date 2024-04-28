from enum import Enum
from typing import Any, List, Literal, Self, Tuple, Type
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Matrix, Tensor, ClassType
from luma.interface.util import InitUtil, Clone
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer


__all__ = (
    "Convolution",
    "Pooling",
    "Dense",
    "Dropout",
    "Flatten",
    "Activation",
    "Sequential",
)


class Convolution(Layer):
    """
    A convolutional layer in a neural network convolves learnable filters
    across input data, detecting patterns like edges or textures, producing
    feature maps essential for tasks such as image recognition within CNNs.
    By sharing parameters, it efficiently extracts hierarchical representations,
    enabling the network to learn complex visual features.

    Parameters
    ----------
    `in_channels` : Number of input channels
    `out_channels` : Number of output channels(filters)
    `filter_size`: Size of each filter
    `stride` : Step size for filters during convolution
    `padding` : Padding stratagies
    (`valid` for no padding, `same` for typical 0-padding)
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

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
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(
                self.out_channels,
                self.in_channels,
                self.filter_size,
                self.filter_size,
            ),
            b_shape=(1, self.out_channels),
        )
        self.set_param_ranges(
            {
                "in_channels": ("0<,+inf", int),
                "out_channels": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor) -> Tensor:
        self.input_ = X
        batch_size, channels, height, width = X.shape
        if self.in_channels != channels:
            raise ValueError(
                f"channels of 'X' does not match with 'in_channels'! "
                + f"({self.in_channels}!={channels})"
            )

        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        out_height = ((padded_h - self.filter_size) // self.stride) + 1
        out_width = ((padded_w - self.filter_size) // self.stride) + 1

        out: Tensor = np.zeros(
            (
                batch_size,
                self.out_channels,
                out_height,
                out_width,
            )
        )
        self.out_shape = out.shape

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=[2, 3])
        filter_fft = np.fft.rfftn(
            self.weights_,
            s=(padded_h, padded_w),
            axes=[2, 3],
        )

        for i in range(batch_size):
            for f in range(self.out_channels):
                result_fft = np.sum(X_fft[i] * filter_fft[f], axis=0)
                result = np.fft.irfftn(result_fft, s=(padded_h, padded_w))

                sampled_result = result[
                    pad_h : padded_h - pad_h : self.stride,
                    pad_w : padded_w - pad_w : self.stride,
                ]
                out[i, f] = sampled_result[:out_height, :out_width]

        out += self.biases_[:, :, np.newaxis, np.newaxis]
        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        batch_size, channels, height, width = X.shape
        pad_h, pad_w, padded_h, padded_w = self._get_padding_dim(height, width)

        dX_padded = np.zeros((batch_size, channels, padded_h, padded_w))
        self.dW = np.zeros_like(self.weights_)
        self.dB = np.zeros_like(self.biases_)

        X_padded = np.pad(
            X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        X_fft = np.fft.rfftn(X_padded, s=(padded_h, padded_w), axes=[2, 3])
        d_out_fft = np.fft.rfftn(d_out, s=(padded_h, padded_w), axes=[2, 3])

        for f in range(self.out_channels):
            self.dB[:, f] = np.sum(d_out[:, f, :, :])

        for f in range(self.out_channels):
            for c in range(channels):
                filter_d_out_fft = np.sum(
                    X_fft[:, c] * d_out_fft[:, f].conj(),
                    axis=0,
                )
                self.dW[f, c] = np.fft.irfftn(
                    filter_d_out_fft,
                    s=(padded_h, padded_w),
                )[pad_h : pad_h + self.filter_size, pad_w : pad_w + self.filter_size]

        self.dW += 2 * self.lambda_ * self.weights_

        for i in range(batch_size):
            for c in range(channels):
                temp = np.zeros((padded_h, padded_w // 2 + 1), dtype=np.complex128)
                for f in range(self.out_channels):
                    filter_fft = np.fft.rfftn(
                        self.weights_[f, c], s=(padded_h, padded_w)
                    )
                    temp += filter_fft * d_out_fft[i, f]
                dX_padded[i, c] = np.fft.irfftn(temp, s=(padded_h, padded_w))

        self.dX = (
            dX_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
            if pad_h > 0 or pad_w > 0
            else dX_padded
        )
        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, ...]:
        if self.padding == "same":
            pad_h = pad_w = (self.filter_size - 1) // 2
            padded_h = height + 2 * pad_h
            padded_w = width + 2 * pad_w

        elif self.padding == "valid":
            pad_h = pad_w = 0
            padded_h = height
            padded_w = width
        else:
            raise UnsupportedParameterError(self.padding)

        return pad_h, pad_w, padded_h, padded_w


class Pooling(Layer):
    """
    A pooling layer in a neural network reduces the spatial dimensions of
    feature maps, reducing computational complexity. It aggregates neighboring
    values, typically through operations like max pooling or average pooling,
    to extract dominant features. Pooling helps in achieving translation invariance
    and reducing overfitting by summarizing the presence of features in local
    regions. It downsamples feature maps, preserving important information while
    discarding redundant details.

    Parameters
    ----------
    `size` : Size of pooling filter
    `stride` : Step size of filter during pooling
    `mode` : Pooling strategy (i.e. max, average)

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
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

        self.set_param_ranges(
            {
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Tensor) -> Tensor:
        self.input_ = X
        batch_size, channels, height, width = X.shape
        out_height = 1 + (height - self.filter_size) // self.stride
        out_width = 1 + (width - self.filter_size) // self.stride

        out: Tensor = np.zeros((batch_size, channels, out_height, out_width))
        self.out_shape = out.shape

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_height_width(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    out[:, :, i, j] = np.max(window, axis=(2, 3))
                elif self.mode == "avg":
                    out[:, :, i, j] = np.mean(window, axis=(2, 3))
                else:
                    raise UnsupportedParameterError(self.mode)

        return out

    def backward(self, d_out: Tensor) -> Tensor:
        X = self.input_
        _, _, out_height, out_width = d_out.shape
        self.dX = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                h_start, h_end, w_start, w_end = self._get_height_width(i, j)
                window = X[:, :, h_start:h_end, w_start:w_end]

                if self.mode == "max":
                    max_vals = np.max(window, axis=(2, 3), keepdims=True)
                    mask_ = window == max_vals
                    self.dX[:, :, h_start:h_end, w_start:w_end] += (
                        mask_ * d_out[:, :, i : i + 1, j : j + 1]
                    )
                elif self.mode == "avg":
                    self.dX[:, :, h_start:h_end, w_start:w_end] += d_out[
                        :, :, i : i + 1, j : j + 1
                    ] / (self.filter_size**2)

        return self.dX

    def _get_height_width(self, cur_h: int, cur_w: int) -> Tuple[int, int, int, int]:
        h_start = cur_h * self.stride
        w_start = cur_w * self.stride
        h_end = h_start + self.filter_size
        w_end = w_start + self.filter_size

        return h_start, h_end, w_start, w_end


class Dense(Layer):
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
    `in_features` : Number of input features
    `out_features`:  Number of output features
    `initializer` : Type of weight initializer (default `None`)
    `optimizer` : Optimizer for weight update
    `lambda_` : L2-regularization strength
    `random_state` : Seed for various random sampling processes

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
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(self.in_features, self.out_features),
            b_shape=(1, self.out_features),
        )
        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Matrix) -> Matrix:
        self.input_ = X

        out = np.dot(X, self.weights_) + self.biases_
        self.out_shape = out.shape
        return out

    def backward(self, d_out: Matrix) -> Matrix:
        X = self.input_

        self.dX = np.dot(d_out, self.weights_.T)
        self.dW = np.dot(X.T, d_out)
        self.dW += 2 * self.lambda_ * self.weights_
        self.dB = np.sum(d_out, axis=0, keepdims=True)

        return self.dX


class Dropout(Layer):
    """
    Dropout is a regularization technique used during training to prevent
    overfitting by randomly setting a fraction of input units to zero during
    the forward pass. This helps in reducing co-adaptation of neurons and
    encourages the network to learn more robust features.

    Parameters
    ----------
    `dropout_rate` : The fraction of input units to drop during training
    `random_state` : Seed for various random sampling processes

    Notes
    -----
    - During inference, dropout is typically turned off, and the layer behaves
      as the identity function.

    """

    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        else:
            return X

    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out


class Flatten(Layer):
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

    def forward(self, X: Tensor) -> Matrix:
        self.input_ = X
        out = X.reshape(X.shape[0], -1)
        self.out_shape = out.shape
        return out

    def backward(self, d_out: Matrix) -> Tensor:
        dX = d_out.reshape(self.input_.shape)
        return dX


@ClassType.non_instantiable()
class Activation:
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

    type FuncType = Type

    class Linear(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            return X

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out
            return self.dX

    class ReLU(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            self.input_ = X
            return np.maximum(0, X)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out.copy()
            self.dX[self.input_ <= 0] = 0
            return self.dX

    class Sigmoid(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            self.output_ = 1 / (1 + np.exp(-X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * self.output_ * (1 - self.output_)
            return self.dX

    class Tanh(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            self.output_ = np.tanh(X)
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - np.square(self.output_))
            return self.dX

    class LeakyReLU(Layer):
        def __init__(self, alpha: float = 0.01) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor) -> Tensor:
            self.input_ = X
            return np.where(X > 0, X, X * self.alpha)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(self.input_ > 0, 1, self.alpha)
            return self.dX

    class Softmax(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            e_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
            return e_X / np.sum(e_X, axis=-1, keepdims=True)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = np.empty_like(d_out)
            for i, (y, dy) in enumerate(zip(self.output_, d_out)):
                y = y.reshape(-1, 1)
                jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)

                self.dX[i] = np.dot(jacobian_matrix, dy)

            return self.dX

    class ELU(Layer):
        def __init__(self, alpha: float = 1.0) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor) -> Tensor:
            self.input_ = X
            self.output_ = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(
                self.input_ > 0,
                1,
                self.output_ + self.alpha,
            )
            return self.dX

    class SELU(Layer):
        def __init__(
            self,
            lambda_: float = 1.0507,
            alpha: float = 1.67326,
        ) -> None:
            super().__init__()
            self.lambda_ = lambda_
            self.alpha = alpha

        def forward(self, X: Tensor) -> Tensor:
            self.input_ = X
            self.output_ = self.lambda_ * np.where(
                X > 0, X, self.alpha * (np.exp(X) - 1)
            )
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = (
                d_out
                * self.lambda_
                * np.where(
                    self.input_ > 0,
                    1,
                    self.alpha * np.exp(self.input_),
                )
            )
            return self.dX

    class Softplus(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor) -> Tensor:
            self.output_ = np.log1p(np.exp(X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - 1 / (1 + np.exp(self.output_)))
            return self.dX

    class Swish(Layer):
        def __init__(self, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta

        def forward(self, X: Tensor) -> Tensor:
            self.input_ = X
            self.sigmoid = 1 / (1 + np.exp(-self.beta * X))
            self.output_ = X * self.sigmoid
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (
                self.sigmoid + self.input_ * self.sigmoid * (1 - self.sigmoid)
            )
            return self.dX


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
    `*layers` : Layers or layers with its name assigned
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

    @ClassType.non_instantiable()
    class LayerType(Enum):
        ONLY_TRAIN: Tuple[Layer] = (Dropout,)

    def __init__(self, *layers: Layer | Tuple[str, Layer]) -> None:
        self.layers: List[Tuple[str, Layer]] = list()
        for layer in layers:
            self.add(layer)

        self.optimizer = None
        self.loss_func_ = None

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        self.input_ = X
        out = X

        for _, layer in self.layers:
            if Sequential._check_only_train(layer):
                out = layer.forward(out, is_train=is_train)
            else:
                out = layer.forward(out)

        self.out_shape = out.shape
        return out

    def backward(self, d_out: Matrix) -> None:
        for _, layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def update(self) -> None:
        self._check_no_optimizer()
        for _, layer in reversed(self.layers):
            layer.update()

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        self.optimizer = optimizer
        self.optimizer.set_params(**params, ignore_missing=True)

        for _, layer in self.layers:
            layer.optimizer = Clone(self.optimizer).get

    @classmethod
    def _check_only_train(cls, layer: Layer) -> bool:
        return type(layer) in cls.LayerType.ONLY_TRAIN.value

    def _check_no_optimizer(self) -> None:
        if self.optimizer is None:
            raise RuntimeError(
                f"'{self}' has no optimizer! "
                + f"Call '{self}().set_optimizer' to assign an optimizer."
            )

    def add(self, layer: Layer | Tuple[str, Layer]) -> None:
        if not isinstance(layer, tuple):
            layer = (str(layer), layer)
        self.layers.append(layer)

        if self.optimizer is not None:
            self.set_optimizer(self.optimizer)
        if self.loss_func_ is not None:
            self.set_loss(self.loss_func_)

    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        for _, layer in self.layers:
            w_, b_ = layer.param_size
            w_size += w_
            b_size += b_

        return w_size, b_size

    def __call__(self, X: Tensor, is_train: bool = False) -> float:
        return self.forward(X, is_train=is_train)

    def __add__(self, other: Layer | Self) -> Self:
        if isinstance(other, Layer):
            self.add(other)
        elif isinstance(other, Self):
            for layer in other.layers:
                self.add(layer)
        else:
            raise TypeError(
                "Unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self).__name__, type(other).__name__
                )
            )

        return self

    def __getitem__(self, index: int) -> Tuple[str, Layer]:
        return self.layers[index]

    def __str__(self) -> str:
        return super().__str__()
