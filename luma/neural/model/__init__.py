"""
`neural.model`
--------------
Neural models are computational systems designed to learn patterns and make 
predictions based on data. They consist of multiple neural layers organized 
in a structured way, allowing the model to process information, extract 
features, and generate outputs. These models can handle a wide range of tasks, 
from image recognition to natural language processing, by learning from data 
and improving performance over time through training.

"""

from typing import Literal

from luma.interface.util import InitUtil

from luma.neural.layer import Activation
from luma.neural.model import (
    alex,
    incep,
    lenet,
    mobile,
    resnet,
    simple,
    vgg,
)


__all__ = (
    "SimpleMLP",
    "SimpleCNN",
    "LeNet_1",
    "LeNet_4",
    "LeNet_5",
    "AlexNet",
    "ZFNet",
    "VGGNet_11",
    "VGGNet_13",
    "VGGNet_16",
    "VGGNet_19",
    "Inception_V1",
    "Inception_V2",
    "Inception_V3",
    "Inception_V4",
    "InceptionResNet_V1",
    "InceptionResNet_V2",
    "ResNet_18",
    "ResNet_34",
    "ResNet_50",
    "ResNet_101",
    "ResNet_152",
    "ResNet_200",
    "ResNet_269",
    "ResNet_1001",
    "XceptionNet",
    "MobileNet_V1",
    "MobileNet_V2",
    "MobileNet_V3_Small",
    "MobileNet_V3_Large",
)

MODELS: tuple[str] = __all__
NUM_MODELS: int = len(MODELS)


class SimpleMLP(simple._SimpleMLP):
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
    `in_features` : int
        Number of input features
    `out_features` : int
        Number of output features
    `hidden_layers` : int of list of int
        Numbers of the features in hidden layers (int for a single layer)
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `initializer` : InitStr, default=None
        Type of weight initializer
    `activation` : callable
        Type of activation function
    `dropout_rate` : float, default=0.5
        Dropout rate
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

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
        activation: callable,
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        dropout_rate: float = 0.5,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False
    ) -> None:
        super(SimpleMLP, self).__init__(
            in_features,
            out_features,
            hidden_layers,
            activation,
            initializer,
            batch_size,
            n_epochs,
            valid_size,
            dropout_rate,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class SimpleCNN(simple._SimpleCNN):
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
    `in_channels_list` : int or list of int
        List of input channels for convolutional blocks
    `in_features_list` : int or list of int
        List of input features for dense blocks
    `out_channels` : int
        Output channels for the last convolutional layer
    `out_features` : int
        Output features for the last dense layer
    `filter_size` : int
        Size of filters for convolution layers
    `activation` : callable
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer (None for dense layers)
    `padding` : {"same", "valid"}, default="same"
        Padding strategy
    `stride` : int, default=1
        Step size of filters during convolution
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Size of filters for pooling layers
    `pool_stride` : int, default=2
        Step size of filters during pooling
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy (default `max`)
    `do_dropout` : bool, default=True
        Whether to perform dropout
    `dropout_rate` : float, default=0.5
        Dropout rate
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

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
        activation=Activation.ReLU,
        ...,
    )
    ```
    This model has the same structure with:
    ```py
    model = Sequential(
        Convolution(1, 6),  # First convolution block
        Activation.ReLU,
        Pooling(),

        Convolution(6, 12),  # Second convolution block
        Activation.ReLU,
        Pooling(),

        Flatten(),
        Dense(96, 16),  # Dense block
        Activation.ReLU,
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
        activation: callable,
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
        valid_size: float = 0.1,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False
    ) -> None:
        super(SimpleCNN, self).__init__(
            in_channels_list,
            in_features_list,
            out_channels,
            out_features,
            filter_size,
            activation,
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
            valid_size,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_1(lenet._LeNet_1):
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
    `activation` : callable, default=Activation.Tanh
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.

    """

    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(LeNet_1, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_4(lenet._LeNet_4):
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
    `activation` : callable, default=Activation.Tanh
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """

    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(LeNet_4, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class LeNet_5(lenet._LeNet_5):
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
    `activation` : callable, default=Activation.Tanh
        Type of activation function
        Type of loss function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """

    def __init__(
        self,
        activation: callable = Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(LeNet_5, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class AlexNet(alex._AlexNet):
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
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet
    Classification with Deep Convolutional Neural Networks." Advances in Neural
    Information Processing Systems, 2012.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(AlexNet, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ZFNet(alex._ZFNet):
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
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding
    Convolutional Networks." European conference on computer vision, 2014.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ZFNet, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class VGGNet_11(vgg._VGGNet_11):
    """
    VGG11 is a simplified variant of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 64) -> Pool2D(2, 2, "max") ->  # Conv_1
    ConvBlock2D(64, 128) -> Pool2D(2, 2, "max") ->  # Conv_2

    ConvBlock2D(128, 256, do_pooling=False) ->  # Conv_3
    ConvBlock2D(256, 256) -> Pool2D(2, 2, "max") ->  # Conv_4

    ConvBlock2D(256, 512, do_pooling=False) ->  # Conv_5
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_6

    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_7
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_8
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(512 * 7 * 7, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    132,851,392 weights, 11,944 biases -> 132,863,336 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(VGGNet_11, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class VGGNet_13(vgg._VGGNet_13):
    """
    VGG13 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 64, do_pooling=False) ->  # Conv_1
    ConvBlock2D(64, 64) -> Pool2D(2, 2, "max") ->  # Conv_2

    ConvBlock2D(64, 128, do_pooling=False) ->  # Conv_3
    ConvBlock2D(128, 128) -> Pool2D(2, 2, "max") ->  # Conv_4

    ConvBlock2D(128, 256, do_pooling=False) ->  # Conv_5
    ConvBlock2D(256, 256) -> Pool2D(2, 2, "max") ->  # Conv_6

    ConvBlock2D(256, 512, do_pooling=False) ->  # Conv_7
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_8

    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_9
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_10
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(512 * 7 * 7, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    133,035,712 weights, 12,136 biases -> 133,047,848 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(VGGNet_13, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class VGGNet_16(vgg._VGGNet_16):
    """
    VGG16 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 64, do_pooling=False) ->  # Conv_1
    ConvBlock2D(64, 64) -> Pool2D(2, 2, "max") ->  # Conv_2

    ConvBlock2D(64, 128, do_pooling=False) ->  # Conv_3
    ConvBlock2D(128, 128) -> Pool2D(2, 2, "max") ->  # Conv_4

    ConvBlock2D(128, 256, do_pooling=False) ->  # Conv_5
    ConvBlock2D(256, 256, do_pooling=False) ->  # Conv_6
    ConvBlock2D(256, 256) -> Pool2D(2, 2, "max") ->  # Conv_7

    ConvBlock2D(256, 512, do_pooling=False) ->  # Conv_8
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_9
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_10

    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_11
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_12
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_13
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(512 * 7 * 7, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    138,344,128 weights, 13,416 biases -> 138,357,544 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(VGGNet_16, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class VGGNet_19(vgg._VGGNet_19):
    """
    VGG19 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Convolutional Blocks:
    ```py
    ConvBlock2D(3, 64, do_pooling=False) ->  # Conv_1
    ConvBlock2D(64, 64) -> Pool2D(2, 2, "max") ->  # Conv_2

    ConvBlock2D(64, 128, do_pooling=False) ->  # Conv_3
    ConvBlock2D(128, 128) -> Pool2D(2, 2, "max") ->  # Conv_4

    ConvBlock2D(128, 256, do_pooling=False) ->  # Conv_5
    ConvBlock2D(256, 256, do_pooling=False) ->  # Conv_6
    ConvBlock2D(256, 256, do_pooling=False) ->  # Conv_7
    ConvBlock2D(256, 256) -> Pool2D(2, 2, "max") ->  # Conv_8

    ConvBlock2D(256, 512, do_pooling=False) ->  # Conv_9
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_10
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_11
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_12

    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_13
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_14
    ConvBlock2D(512, 512, do_pooling=False) ->  # Conv_15
    ConvBlock2D(512, 512) -> Pool2D(2, 2, "max") ->  # Conv_16
    ```
    Fully Connected Layers:
    ```py
    Flatten ->
    DenseBlock(512 * 7 * 7, 4096) -> DenseBlock(4096, 4096) ->
    Dense(4096, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    143,652,544 weights, 14,696 biases -> 143,667,240 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(VGGNet_19, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class Inception_V1(incep._Inception_V1):
    """
    Inception v1, also known as GoogLeNet, is a deep convolutional neural network
    architecture designed for image classification. It introduces an "Inception
    module," which uses multiple convolutional filters of different sizes in
    parallel to capture various features at different scales. This architecture
    reduces computational costs by using 1x1 convolutions to decrease the number
    of input channels. Inception v1 achieved state-of-the-art results on the
    ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Introductory Convolutions:
    ```py
    ConvBlock2D(3, 64, filter_size=7, pool_mode="max") ->
    ConvBlock2D(64, 64, filter_size=1, do_pooling=False) ->
    ConvBlock2D(64, 192, filter_size=3, pool_mode="max") ->
    ```
    Inception Blocks:
    ```py
    IncepBlock.V1(192) ->  # Inception_3a
    IncepBlock.V1(256) ->  # Inception_3b
    Pool2D(3, 2, mode="max") ->

    IncepBlock.V1(480) ->  # Inception_4a
    IncepBlock.V1(512) ->  # Inception_4b
    IncepBlock.V1(512) ->  # Inception_4c
    IncepBlock.V1(512) ->  # Inception_4d
    IncepBlock.V1(528) ->  # Inception_4e
    Pool2D(3, 2, mode="max") ->

    IncepBlock.V1(832) ->  # Inception_5a
    IncepBlock.V1(832) ->  # Inception_5b
    GlobalAvgPool2D() ->
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(1024, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    6,990,272 weights, 8,280 biases -> 6,998,552 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(Inception_V1, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class Inception_V2(incep._Inception_V2):
    """
    Inception v2, an improvement of the original Inception architecture,
    enhances computational efficiency and accuracy in deep learning models.
    It introduces the factorization of convolutions and additional
    normalization techniques to reduce the number of parameters and improve
    training stability. These modifications allow for deeper and more
    complex neural networks with improved performance.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Introductory Convolutions:
    ```py
    ConvBlock2D(3, 32, filter_size=3, stride=2) ->
    ConvBlock2D(32, 32, filter_size=3, stride=1) ->
    ConvBlock2D(32, 64, filter_size=3, stride=1) ->
    Pool2D(3, 2, mode="max") ->

    ConvBlock2D(64, 80, filter_size=3, stride=1) ->
    ConvBlock2D(80, 192, filter_size=3, stride=2) ->
    ConvBlock2D(192, 288, filter_size=3, stride=1) ->
    ```
    Inception Blocks:
    ```py
    3x IncepBlock.V2_TypeA(288) ->  # Inception_3
    IncepBlock.V2_Redux(288) ->  # Inception_Rx1

    5x IncepBlock.V2_TypeB(768) ->  # Inception_4
    IncepBlock.V2_Redux(768) ->  # Inception_Rx2

    2x IncepBlock.V2_TypeC([1280, 2048]) ->  # Inception_5
    GlobalAvgPool2D() ->
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(2048, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    24,974,688 weights, 20,136 biases -> 24,994,824 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(Inception_V2, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class Inception_V3(incep._Inception_V3):
    """
    Inception v3, an enhancement of Inception v2, further improves
    computational efficiency and accuracy in deep learning models.
    It includes advanced factorization of convolutions, improved grid
    size reduction techniques, extensive Batch Normalization, and
    label smoothing to prevent overfitting. These modifications enable
    deeper and more complex neural networks with significantly
    enhanced performance and robustness.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Introductory Convolutions:
    ```py
    ConvBlock2D(3, 32, filter_size=3, stride=2) ->
    ConvBlock2D(32, 32, filter_size=3, stride=1) ->
    ConvBlock2D(32, 64, filter_size=3, stride=1) ->
    Pool2D(3, 2, mode="max") ->

    ConvBlock2D(64, 80, filter_size=3, stride=1) ->
    ConvBlock2D(80, 192, filter_size=3, stride=2) ->
    ConvBlock2D(192, 288, filter_size=3, stride=1) ->
    ```
    Inception Blocks:
    ```py
    3x IncepBlock.V2_TypeA(288) ->  # Inception_3
    IncepBlock.V2_Redux(288) ->  # Inception_Rx1

    5x IncepBlock.V2_TypeB(768) ->  # Inception_4
    IncepBlock.V2_Redux(768) ->  # Inception_Rx2

    2x IncepBlock.V2_TypeC([1280, 2048]) ->  # Inception_5
    GlobalAvgPool2D() ->
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(2048, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    25,012,960 weights, 20,136 biases -> 25,033,096 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(Inception_V3, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            smoothing,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class Inception_V4(incep._Inception_V4):
    """
    Inception v4, an enhancement of Inception v3, improves computational
    efficiency and accuracy. It includes sophisticated convolution
    factorization, refined grid size reduction, extensive Batch
    Normalization, and label smoothing. These advancements enable deeper
    and more robust neural networks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Overall:
    ```py
    IncepBlock.V4_Stem() ->  # Stem

    4x IncepBlock.V4_TypeA() ->  # Type A
    IncepBlock.V4_ReduxA(384, (192, 224, 256, 384)) ->  # Redux Type A

    7x IncepBlock.V4_TypeB() ->  # Type B
    IncepBlock.V4_ReduxB() ->  # Redux Type B

    3x IncepBlock.V4_TypeC() ->  # Type C
    GlobalAvgPool2D() ->
    Dropout(0.8) ->
    ```
    Fully Connected Layers:
    ```py
    Flatten() -> Dense(1536, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    42,641,952 weights, 32,584 biases -> 42,674,536 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Inception-v4, Inception-ResNet and the
    Impact of Residual Connections on Learning.” Proceedings of the Thirty-First
    AAAI Conference on Artificial Intelligence, 2017, pp. 4278-4284.
    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(Inception_V4, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            smoothing,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class InceptionResNet_V1(incep._InceptionRes_V1):
    """
    Inception-ResNet v1 combines Inception modules with residual connections,
    improving computational efficiency and accuracy. This architecture uses
    convolution factorization, optimized grid size reduction, extensive
    Batch Normalization, and label smoothing, resulting in deeper and more
    robust neural networks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Overall:
    ```py
    IncepResBlock.V1_Stem() ->  # Stem

    5x IncepResBlock.V1_TypeA() ->  # Type A
    IncepBlock.V4_ReduxA(256, (192, 192, 256, 384)) ->  # Redux Type A

    10x IncepResBlock.V1_TypeB() ->  # Type B
    IncepResBlock.V1_Redux() ->  # Redux Type B

    5x IncepResBlock.V1_TypeC() ->  # Type C
    GlobalAvgPool2D() ->
    Dropout(0.8) ->
    ```
    Fully Connected Layers:
    ```py
    Flatten() -> Dense(1792, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    21,611,648 weights, 33,720 biases -> 21,645,368 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(InceptionResNet_V1, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            smoothing,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class InceptionResNet_V2(incep._InceptionRes_V2):
    """
    Inception-ResNet v2 enhances v1 with a deeper architecture and
    improved residual blocks for better performance. It features refined
    convolution factorization, more extensive Batch Normalization, and
    advanced grid size reduction.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Overall:
    ```py
    IncepBlock.V4_Stem() ->  # Stem

    5x IncepResBlock.V2_TypeA() ->  # Type A
    IncepBlock.V4_ReduxA(384, (256, 256, 384, 384)) ->  # Redux Type A

    10x IncepResBlock.V2_TypeB() ->  # Type B
    IncepResBlock.V2_Redux() ->  # Redux Type B

    5x IncepResBlock.V2_TypeC() ->  # Type C
    GlobalAvgPool2D() ->
    Dropout(0.8) ->
    ```
    Fully Connected Layers:
    ```py
    Flatten() -> Dense(2272, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    34,112,608 weights, 43,562 biases -> 34,156,170 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(InceptionResNet_V2, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            dropout_rate,
            smoothing,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_18(resnet._ResNet_18):
    """
    ResNet-18 is a 18-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    2x ResNetBlock.Basic(64, 64) ->  # conv2
    2x ResNetBlock.Basic(128, 128, stride=2) ->  # conv3
    2x ResNetBlock.Basic(256, 256, stride=2) ->  # conv4
    2x ResNetBlock.Basic(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    11,688,512 weights, 5,800 biases -> 11,694,312 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_18, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_34(resnet._ResNet_34):
    """
    ResNet-34 is a 34-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.Basic(64, 64) ->  # conv2
    4x ResNetBlock.Basic(128, 128, stride=2) ->  # conv3
    6x ResNetBlock.Basic(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.Basic(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    21,796,672 weights, 9,512 biases -> 21,806,184 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_34, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_50(resnet._ResNet_50):
    """
    ResNet-50 is a 50-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.Bottleneck(64, 64) ->  # conv2
    4x ResNetBlock.Bottleneck(128, 128, stride=2) ->  # conv3
    6x ResNetBlock.Bottleneck(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.Bottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    25,556,032 weights, 27,560 biases -> 25,583,592 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_50, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_101(resnet._ResNet_101):
    """
    ResNet-101 is a 101-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.Bottleneck(64, 64) ->  # conv2
    4x ResNetBlock.Bottleneck(128, 128, stride=2) ->  # conv3
    23x ResNetBlock.Bottleneck(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.Bottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    44,548,160 weights, 53,672 biases -> 44,601,832 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_101, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_152(resnet._ResNet_152):
    """
    ResNet-152 is a 152-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.Bottleneck(64, 64) ->  # conv2
    8x ResNetBlock.Bottleneck(128, 128, stride=2) ->  # conv3
    36x ResNetBlock.Bottleneck(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.Bottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    60,191,808 weights, 76,712 biases -> 60,268,520 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_152, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_200(resnet._ResNet_200):
    """
    ResNet-200 is a 200-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.PreActBottleneck(64, 64) ->  # conv2
    24x ResNetBlock.PreActBottleneck(128, 128, stride=2) ->  # conv3
    36x ResNetBlock.PreActBottleneck(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.PreActBottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    64,668,864 weights, 89,000 biases -> 64,757,864 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_200, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_269(resnet._ResNet_269):
    """
    ResNet-269 is a 269-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.PreActBottleneck(64, 64) ->  # conv2
    30x ResNetBlock.PreActBottleneck(128, 128, stride=2) ->  # conv3
    48x ResNetBlock.PreActBottleneck(256, 256, stride=2) ->  # conv4
    8x ResNetBlock.PreActBottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    102,068,416 weights, 127,400 biases -> 102,195,816 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_269, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class ResNet_1001(resnet._ResNet_1001):
    """
    ResNet-1001 is a 1001-layer deep neural network that uses residual
    blocks to improve training by learning residuals, helping prevent
    vanishing gradients and enabling better performance in image
    recognition tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Residual Blocks:
    ```py
    Conv2D(3, 64, filter_size=7, stride=2) ->  # conv1

    3x ResNetBlock.PreActBottleneck(64, 64) ->  # conv2
    33x ResNetBlock.PreActBottleneck(128, 128, stride=2) ->  # conv3
    99x ResNetBlock.PreActBottleneck(256, 256, stride=2) ->  # conv4
    3x ResNetBlock.PreActBottleneck(512, 512, stride=2) ->  # conv5

    AdaptiveAvgPool2D((1, 1)) ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(512 * 4, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    159,884,992 weights, 208,040 biases -> 160,093,032 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    Warnings
    --------
    * This model has highly intensive depth of convolutional layers.
    Please consider your computational power, memory, etc.

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(ResNet_1001, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class XceptionNet(incep._Xception):
    """
    XceptionNet enhances the Inception architecture by replacing standard
    convolutions with depthwise separable convolutions, making it more
    efficient and effective at feature extraction. This design reduces
    the number of parameters and computations while maintaining or
    improving model accuracy on complex tasks.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 299, 299]
    ```
    Separable Convolutions:
    ```py
    XceptionBlock.EntryFlow() ->  # entry flow
    8x XceptionBlock.MiddleFlow() ->  # middle flow
    XceptionBlock.ExitFlow() ->  # exit flow

    SeparableConv2D(1024, 1536, 3) -> Activation.ReLU() ->
    SeparableConv2D(1536, 2048, 3) -> Activation.ReLU() ->

    GlobalAvgPool2D() ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(2048, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    22,113,984 weights, 50,288 biases -> 22,164,272 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Chollet, François. “Xception: Deep Learning with Depthwise
    Separable Convolutions.” Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), 2017, pp.
    1251-1258.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(XceptionNet, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class MobileNet_V1(mobile._Mobile_V1):
    """
    MobileNet-V1 uses depthwise separable convolutions to significantly
    reduce the number of parameters and computational cost, making it
    highly efficient for mobile and embedded devices. It balances
    accuracy and efficiency through adjustable width and resolution
    multipliers.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Separable Convolutions:
    ```py
    Conv2D(3, 32) ->
    SeparableConv2D(32, 64) ->
    SeparableConv2D(64, 128) -> SeparableConv2D(128, 128) ->
    SeparableConv2D(128, 256) -> SeparableConv2D(256, 256) ->
    SeparableConv2D(256, 512) ->

    5x SeparableConv2D(512, 512) ->
    SeparableConv2D(512, 1024) -> SeparableConv2D(1024, 1024) ->

    GlobalAvgPool2D() ->  # avg pool
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(1024, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    4,230,976 weights, 11,944 biases -> 4,242,920 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `width_param` : float, default=1.0
        Width parameter(alpha) of the network
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew G., et al. “MobileNets: Efficient Convolutional
    Neural Networks for Mobile Vision Applications.” arXiv preprint
    arXiv:1704.04861 (2017).

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        width_param: float = 1.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(MobileNet_V1, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            width_param,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class MobileNet_V2(mobile._Mobile_V2):
    """
    MobileNet-V2 builds on the efficiency of its predecessor by introducing
    inverted residuals and linear bottlenecks, further reducing
    computational cost and enhancing performance on mobile and embedded
    devices. It continues to balance accuracy and efficiency while allowing
    for flexible adjustments through width and resolution multipliers.

    Structure
    ---------
    Input:
    ```py
    Tensor[..., 3, 224, 224]
    ```
    Inverted Residual Blocks:
    ```py
    Conv2D(3, 32) -> BatchNorm2D(32) -> ReLU6() ->

    1x MobileNetBlock.InvertedRes(32, 16, 1, 1) ->
    2x MobileNetBlock.InvertedRes(16, 24, 2, 6) ->
    3x MobileNetBlock.InvertedRes(24, 32, 2, 6) ->
    4x MobileNetBlock.InvertedRes(32, 64, 2, 6) ->
    3x MobileNetBlock.InvertedRes(64, 96, 1, 6) ->
    3x MobileNetBlock.InvertedRes(96, 160, 2, 6) ->
    1x MobileNetBlock.InvertedRes(160, 320, 1, 6) ->

    Conv2D(320, 1280) -> BatchNorm2D(1280) -> ReLU6() ->
    GlobalAvgPool2D() ->  # 7x7 avg pool
    Conv2D(1280, 1280, 1) -> BatchNorm2D(1280) -> ReLU6() ->
    ```
    Fully Connected Layers:
    ```py
    Flatten -> Dense(1280, 1000)
    ```
    Output:
    ```py
    Matrix[..., 1000]
    ```
    Parameter Size:
    ```txt
    8,418,624 weights, 19,336 biases -> 8,437,960 params
    ```
    Parameters
    ----------
    `activation` : callable, default=Activation.ReLU6
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `width_param` : float, default=1.0
        Width parameter(alpha) of the network
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear
    Bottlenecks.” Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 4510-4520.

    """

    def __init__(
        self,
        activation: callable = Activation.ReLU6,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0,
        momentum: float = 0.9,
        width_param: float = 1.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        super(MobileNet_V2, self).__init__(
            activation,
            initializer,
            out_features,
            batch_size,
            n_epochs,
            valid_size,
            lambda_,
            momentum,
            width_param,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )


class MobileNet_V3_Small(mobile._Mobile_V3_Small):
    NotImplemented


class MobileNet_V3_Large(mobile._Mobile_V3_Large):
    NotImplemented
