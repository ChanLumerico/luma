from typing import Literal

from luma.core.super import Optimizer
from luma.interface.util import InitUtil

from luma.neural.base import Loss
from luma.neural.layer import Activation
from luma.neural.loss import CrossEntropy

from ._models import _simple, _lenet, _imagenet


__all__ = (
    "SimpleMLP",
    "SimpleCNN",
    "LeNet_1",
    "LeNet_4",
    "LeNet_5",
    "AlexNet",
    "ZFNet",
)


class SimpleMLP(_simple._SimpleMLP):
    """
    An MLP (Multilayer Perceptron) is a type of artificial neural network
    composed of at least three layers: an input layer, one or more hidden
    layers, and an output layer. Each layer consists of nodes, or neurons,
    which are fully connected to the neurons in the next layer. MLPs use a
    technique called backpropagation for learning, where the output error
    is propagated backwards through the network to update the weights.
    They are capable of modeling complex nonlinear relationships between
    inputs and outputs. MLPs are commonly used for tasks like classification,
    regression, and pattern recognition.

    Structure
    ---------
    ```py
    (Dense -> Activation -> Dropout) -> ... -> Dense
    ```
    Parameters
    ----------
    `in_features` : Number of input features
    `out_features` : Number of output features
    `hidden_layers` : Numbers of the features in hidden layers
    (`int` for a single layer)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `initializer` : Type of weight initializer
    `activation` : Type of activation function
    `optimizer` : An optimizer used in weight update process
    `loss` : Type of loss function
    `dropout_rate` : Dropout rate
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    Notes
    -----
    - If the data or the target is a 1D-Array(`Vector`), reshape it into a
        higher dimensional array.

    - For classification tasks, the target vector `y` must be
        one-hot encoded.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list[int] | int,
        *,
        activation: Activation,
        optimizer: Optimizer,
        loss: Loss,
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        valid_size: float = 0.1,
        dropout_rate: float = 0.5,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            hidden_layers,
            activation,
            optimizer,
            loss,
            initializer,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            dropout_rate,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class SimpleCNN(_simple._SimpleCNN):
    """
    A Convolutional Neural Network (CNN) is a type of deep neural network
    primarily used in image recognition and processing that is particularly
    powerful at capturing spatial hierarchies in data. A CNN automatically
    detects important features without any human supervision using layers
    with convolving filters that pass over the input image and compute outputs.
    These networks typically include layers such as convolutional layers,
    pooling layers, and fully connected layers that help in reducing the
    dimensions while retaining important features.

    Structure
    ---------
    ```py
    ConvBlock2D -> ... -> Flatten -> DenseBlock -> ... -> Dense
    ```
    Parameters
    ----------
    `in_channels_list` : List of input channels for convolutional blocks
    `in_features_list` : List of input features for dense blocks
    `out_channels` : Output channels for the last convolutional layer
    `out_features` : Output features for the last dense layer
    `filter_size` : Size of filters for convolution layers
    `activation` : Type of activation function
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function
    `initializer` : Type of weight initializer (`None` for dense layers)
    `padding` : Padding strategy (default `same`)
    `stride` : Step size of filters during convolution
    `do_batch_norm` : Whether to perform batch normalization (default `True`)
    `momentum` : Momentum for batch normalization
    `do_pooling` : Whether to perform pooling
    `pool_filter_size` : Size of filters for pooling layers
    `pool_stride` : Step size of filters during pooling
    `pool_mode` : Pooling strategy (default `max`)
    `do_dropout` : Whether to perform dropout
    `dropout_rate` : Dropout rate
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    Notes
    -----
    - Input `X` must have the shape of 4D-array(`Tensor`)

    - For classification tasks, the target vector `y` must be
        one-hot encoded.

    Examples
    --------
    ```py
    model = SimpleCNN(
        in_channels_list=[1, 6],
        in_features_list=[96, 16],
        out_channels=12,
        out_features=10,
        activation=Activation.ReLU(),
        ...,
    )
    ```
    This model has the same structure with:
    ```py
    model = Sequential(
        Convolution(1, 6),  # First convolution block
        Activation.ReLU(),
        Pooling(),

        Convolution(6, 12),  # Second convolution block
        Activation.ReLU(),
        Pooling(),

        Flatten(),
        Dense(96, 16),  # Dense block
        Activation.ReLU(),
        Dropout(),

        Dense(16, 10),  # Final dense layer
    )
    ```
    """

    def __init__(
        self,
        in_channels_list: list[int] | int,
        in_features_list: list[int] | int,
        out_channels: int,
        out_features: int,
        *,
        filter_size: int,
        activation: Activation,
        optimizer: Optimizer,
        loss: Loss,
        initializer: InitUtil.InitStr = None,
        padding: Literal["same", "valid"] = "same",
        stride: int = 1,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        valid_size: float = 0.1,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False
    ) -> None:
        super().__init__(
            in_channels_list,
            in_features_list,
            out_channels,
            out_features,
            filter_size,
            activation,
            optimizer,
            loss,
            initializer,
            padding,
            stride,
            do_batch_norm,
            momentum,
            do_pooling,
            pool_filter_size,
            pool_stride,
            pool_mode,
            do_dropout,
            dropout_rate,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_1(_lenet._LeNet_1):
    """
    LeNet-1 is an early convolutional neural network (CNN) proposed by
    Yann LeCun in 1988, primarily designed for handwritten character
    recognition. It consists of two convolutional layers interleaved
    with subsampling layers, followed by a fully connected layer.
    The network uses convolutions to automatically learn spatial
    hierarchies of features, which are then used for classification
    tasks. LeNet-1 was one of the first successful applications of CNNs,
    laying the groundwork for more complex architectures in image
    processing.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 1, 28, 28]
    ```
    Convolution Layers:
    ```py
    ConvBlock2D(1, 4) -> ConvBlock2D(4, 8)
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(8 * 4 * 4, 10)
    ```
    Output:
    ```py
    Matrix[..., 10]
    ```
    Parameter Size:
    ```txt
    2,180 weights, 22 biases -> 2,202 params
    ```
    Parameters
    ----------
    `activation` : Type of activation function (Default `Tanh`)
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function (Default `CrossEntropy`)
    `initializer` : Type of weight initializer (`None` for dense layers)
    `out_features` : Number of output features (Default `10`)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        super().__init__(
            optimizer,
            activation,
            loss,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_4(_lenet._LeNet_4):
    """
    LeNet-4 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 1, 32, 32]
    ```
    Convolution Layers:
    ```py
    ConvBlock2D(1, 4) -> ConvBlock2D(4, 16)
    ```
    Fully Connected Layers:
    ```py
    Flatten -> DenseBlock(16 * 5 * 5, 120) -> Dense(10)
    ```
    Output:
    ```py
    Matrix[..., 10]
    ```
    Parameter Size:
    ```txt
    50,902 weights, 150 biases -> 51,052 params
    ```
    Parameters
    ----------
    `activation` : Type of activation function (Default `Tanh`)
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function (Default `CrossEntropy`)
    `initializer` : Type of weight initializer (`None` for dense layers)
    `out_features` : Number of output features (Default `10`)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        super().__init__(
            optimizer,
            activation,
            loss,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_5(_lenet._LeNet_5):
    """
    LeNet-5 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 1, 32, 32]
    ```
    Convolution Layers:
    ```py
    ConvBlock2D(1, 6) -> ConvBlock2D(6, 16)
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(16 * 5 * 5, 120) -> DenseBlock(120, 84) -> Dense(84, 10)
    ```
    Output:
    ```py
    Matrix[..., 10]
    ```
    Parameter Size:
    ```txt
    61,474 weights, 236 biases -> 61,710 params
    ```
    Parameters
    ----------
    `activation` : Type of activation function (Default `Tanh`)
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function (Default `CrossEntropy`)
    `initializer` : Type of weight initializer (`None` for dense layers)
    `out_features` : Number of output features (Default `10`)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation = Activation.Tanh(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        super().__init__(
            optimizer,
            activation,
            loss,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class AlexNet(_imagenet._AlexNet):
    """
    AlexNet is a deep convolutional neural network that is designed for
    challenging image recognition tasks and was the winning entry in ILSVRC 2012.
    This architecture uses deep layers of convolutions with ReLU activations,
    max pooling, dropout, and fully connected layers leading to a classification
    output.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 227, 227]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 96) -> LocalResponseNorm(5) ->  # Conv_1
    ConvBlock2D(96, 256) -> LocalResponseNorm(5) ->  # Conv_2

    ConvBlock2D(256, 384, do_pooling=False) -> LocalResponseNorm(5) ->  # Conv_3
    ConvBlock2D(384, 384, do_pooling=False) -> LocalResponseNorm(5) ->  # Conv_4

    ConvBlock2D(384, 256) -> LocalResponseNorm(5) ->  # Conv_5
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(256 * 6 * 6, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    62,367,776 weights, 10,568 biases -> 62,378,344 params
    ```
    Parameters
    ----------
    `activation` : Type of activation function (Default `ReLU`)
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function (Default `CrossEntropy`)
    `initializer` : Type of weight initializer (Default None)
    `out_features` : Number of output features (Default `1000`)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whether to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet
    Classification with Deep Convolutional Neural Networks." Advances in Neural
    Information Processing Systems, 2012.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation = Activation.ReLU(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        super().__init__(
            optimizer,
            activation,
            loss,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ZFNet(_imagenet._ZFNet):
    """
    ZFNet is a refinement of the AlexNet architecture that was specifically
    designed to improve model understanding and performance on image recognition
    tasks. This model was presented by Matthew Zeiler and Rob Fergus in their
    paper and was particularly notable for its improvements in layer configurations
    that enhanced visualization of intermediate activations, aiding in understanding
    the functioning of deep convolutional networks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 227, 227]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 96) -> LocalResponseNorm(5) ->  # Conv_1
    ConvBlock2D(96, 256) -> LocalResponseNorm(5) ->  # Conv_2

    ConvBlock2D(256, 384, do_pooling=False) -> LocalResponseNorm(5) ->  # Conv_3
    ConvBlock2D(384, 384, do_pooling=False) -> LocalResponseNorm(5) ->  # Conv_4

    ConvBlock2D(384, 256) -> LocalResponseNorm(5) ->  # Conv_5
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(256 * 6 * 6, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    58,292,000 weights, 9,578 biases -> 58,301,578 params
    ```
    Parameters
    ----------
    `activation` : Type of activation function (Default `ReLU`)
    `optimizer` : Type of optimizer for weight update
    `loss` : Type of loss function (Default `CrossEntropy`)
    `initializer` : Type of weight initializer (Default None)
    `out_features` : Number of output features (Default `1000`)
    `batch_size` : Size of a single mini-batch
    `n_epochs` : Number of epochs for training
    `learning_rate` : Step size during optimization process
    `valid_size` : Fractional size of validation set
    `lambda_` : L2 regularization strength
    `early_stopping` : Whether to early-stop the training when the valid
    score stagnates
    `patience` : Number of epochs to wait until early-stopping
    `shuffle` : Whether to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding
    Convolutional Networks." European conference on computer vision, 2014.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        activation: Activation = Activation.ReLU(),
        loss: Loss = CrossEntropy(),
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        super().__init__(
            optimizer,
            activation,
            loss,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            learning_rate,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
