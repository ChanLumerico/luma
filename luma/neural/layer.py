from typing import Any, List, Literal, Self, Tuple
import numpy as np

from luma.core.super import Optimizer
from luma.interface.typing import Matrix, Tensor
from luma.interface.util import ActivationUtil, InitUtil, Clone
from luma.interface.exception import UnsupportedParameterError
from luma.neural.base import Layer, Loss


__all__ = (
    "Convolution",
    "Pooling",
    "Dense",
    "Dropout",
    "Flatten",
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
    `activation` : Type of activation function
    `initializer` : Type of weight initializer (default 'auto')
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
        activation: ActivationUtil.FuncType = "relu",
        initializer: InitUtil.InitType = "auto",
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
        self.activation = activation
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state

        act = ActivationUtil(self.activation)
        self.act_ = act.activation_type()
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
        out = self.act_.func(out)
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
        self.dX = self.act_.grad(self.dX)
        return self.dX

    def _get_padding_dim(self, height: int, width: int) -> Tuple[int, int, int, int]:
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
    `activation` : Activation function (Use `Sigmoid` for final dense layer)
    `initializer` : Type of weight initializer (default `auto`)
    `optimizer` : Optimizer for weight update (default `SGDOptimizer`)
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
        activation: ActivationUtil.FuncType = "relu",
        initializer: InitUtil.InitType = "auto",
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state

        act = ActivationUtil(self.activation)
        self.act_ = act.activation_type()
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
        out = self.act_.func(out)
        self.out_shape = out.shape
        return out

    def backward(self, d_out: Matrix) -> Matrix:
        X = self.input_
        d_out = self.act_.grad(d_out)

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

    def __init__(self, dropout_rate: float = 0.5, random_state: int = None) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self.mask_: Tensor = None
        self.rs_ = np.random.RandomState(self.random_state)

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        self.input_ = X
        self.out_shape = self.input_.shape

        if is_train:
            self.mask_ = (
                self.rs_.rand(*X.shape) < self.dropout_rate
            ) / self.dropout_rate
            return X * self.mask_
        else:
            return X

    def backward(self, d_out: Tensor) -> Tensor:
        dX = d_out * self.mask_ if self.mask_ is not None else d_out
        return dX


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
    For setting a loss function of the model:
    ```py
    def set_loss(self, loss_func: Loss) -> None
    ```
    To add additional layer:
    ```py
    def add(self, layer: Layer) -> None
    ```
    To compute loss:
    ```py
    def get_loss(y: Matrix, out: Matrix) -> float
    ```
    Specials
    --------
    - You can use `+` operator to add a layer or another instance of `Sequential`.

    - By calling its instance, `forward`, `backward`, and `update`
        is automatically called (single cycle) and the loss is returned.

    - Use `repr()` to print out its structural configuration.

    Notes
    -----
    - Before any execution, an optimizer and a loss function must be assigned.

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
    model.set_loss(AnyLoss())
    ```
    To use automated cyclic run:
    >>> model(X, y, is_train=True)

    Manual run:
    >>> model.forward(X, is_train=True)
    >>> model.backward(d_out)
    >>> model.update()

    """

    trainable: List[Layer] = [Convolution, Dense]
    only_for_train: List[Layer] = [Dropout]

    def __init__(
        self,
        *layers: Layer | Tuple[str, Layer],
        verbose: bool = False,
    ) -> None:
        self.layers: List[Tuple[str, Layer]] = list()
        for layer in layers:
            self.add(layer)

        self.optimizer = None
        self.loss_func_ = None
        self.verbose = verbose

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        self.input_ = X
        out = X

        for name, layer in self.layers:
            if Sequential._check_only_for_train(layer):
                out = layer.forward(out, is_train=is_train)
            else:
                out = layer.forward(out)
            if self.verbose:
                print(f"[Sequential] Feed-forwarded '{name}'")

        self.out_shape = out.shape
        return out

    def backward(self, d_out: Matrix) -> None:
        for name, layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            if self.verbose:
                print(f"[Sequential] Backpropagated '{name}'")

    def update(self) -> None:
        self._check_no_optimizer_loss()
        for name, layer in reversed(self.layers):
            layer.update()
            if self.verbose and Sequential._check_trainable_layer(layer):
                print(f"[Sequential] Updated '{name}'")

    def set_optimizer(self, optimizer: Optimizer, **params: Any) -> None:
        self.optimizer = optimizer
        self.optimizer.set_params(**params, ignore_missing=True)

        for _, layer in self.layers:
            layer.optimizer = Clone(self.optimizer).get

    def set_loss(self, loss_func: Loss) -> None:
        self.loss_func_ = loss_func

    def get_loss(self, y: Matrix, out: Matrix) -> float:
        return self.loss_func_.loss(y, out)

    @classmethod
    def _check_only_for_train(cls, layer: Layer) -> bool:
        return type(layer) in cls.only_for_train

    @classmethod
    def _check_trainable_layer(cls, layer: Layer) -> bool:
        return layer in cls.trainable

    def _check_no_optimizer_loss(self) -> None:
        if self.optimizer is None:
            raise RuntimeError(
                f"'{self}' has no optimizer! "
                + f"Call '{self}().set_optimizer' to assign an optimizer."
            )
        if self.loss_func_ is None:
            raise RuntimeError(
                f"'{self}' has no loss function! "
                + f"Call '{self}().set_loss' to assign a loss function."
            )

    def add(self, layer: Layer | Tuple[str, Layer]) -> None:
        if not isinstance(layer, tuple):
            layer = (str(layer), layer)
        self.layers.append(layer)

    @property
    def param_size(self) -> Tuple[int, int]:
        w_size, b_size = 0, 0
        for _, layer in self.layers:
            w_, b_ = layer.param_size
            w_size += w_
            b_size += b_

        return w_size, b_size

    def __call__(self, X: Tensor, y: Matrix, is_train: bool = False) -> float:
        self._check_no_optimizer_loss()
        out = self.forward(X, is_train=is_train)
        d_out = self.loss_func_.grad(y, out)

        self.backward(d_out)
        self.update()
        return self.get_loss(y, out)

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
        if self.optimizer is not None:
            self.set_optimizer(self.optimizer)
        if self.loss_func_ is not None:
            self.set_loss(self.loss_func_)

        return self

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        rep = f"{type(self).__name__} Configuration\n"
        rep += "-" * 70 + "\n"
        for name, layer in self.layers:
            rep += f"({name}) {repr(layer)}\n"

        w_size, b_size = self.param_size
        rep += f"\nTotal Layers: {len(self.layers)}"
        rep += f"\nTotal Params: ({w_size:,} weights, {b_size:,} biases)"
        rep += f" -> {w_size + b_size:,}\n"
        rep += "-" * 70
        return rep
