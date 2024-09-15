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

from luma.neural.model import (
    alex,
    dense,
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
    "InceptionRes_V1",
    "InceptionRes_V2",
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
    "SE_ResNet_50",
    "SE_ResNet_152",
    "SE_InceptionRes_V2",
    "DenseNet_121",
    "DenseNet_169",
    "DenseNet_201",
    "DenseNet_264",
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

    """


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

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 28, 28] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    2,180 weights, 22 biases -> 2,202 params
    ```
    Arguments
    ---------
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


class LeNet_4(lenet._LeNet_4):
    """
    LeNet-4 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 32, 32] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    50,902 weights, 150 biases -> 51,052 params
    ```
    Arguments
    ---------
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


class LeNet_5(lenet._LeNet_5):
    """
    LeNet-5 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 32, 32] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    61,474 weights, 236 biases -> 61,710 params
    ```
    Arguments
    ---------
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


class AlexNet(alex._AlexNet):
    """
    AlexNet is a deep convolutional neural network that is designed for
    challenging image recognition tasks and was the winning entry in ILSVRC 2012.
    This architecture uses deep layers of convolutions with ReLU activations,
    max pooling, dropout, and fully connected layers leading to a classification
    output.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    62,367,776 weights, 10,568 biases -> 62,378,344 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class ZFNet(alex._ZFNet):
    """
    ZFNet is a refinement of the AlexNet architecture that was specifically
    designed to improve model understanding and performance on image recognition
    tasks. This model was presented by Matthew Zeiler and Rob Fergus in their
    paper and was particularly notable for its improvements in layer configurations
    that enhanced visualization of intermediate activations, aiding in understanding
    the functioning of deep convolutional networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 227, 227] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    58,292,000 weights, 9,578 biases -> 58,301,578 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class VGGNet_11(vgg._VGGNet_11):
    """
    VGG11 is a simplified variant of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    132,851,392 weights, 11,944 biases -> 132,863,336 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class VGGNet_13(vgg._VGGNet_13):
    """
    VGG13 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    133,035,712 weights, 12,136 biases -> 133,047,848 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class VGGNet_16(vgg._VGGNet_16):
    """
    VGG16 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    138,344,128 weights, 13,416 biases -> 138,357,544 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class VGGNet_19(vgg._VGGNet_19):
    """
    VGG19 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    143,652,544 weights, 14,696 biases -> 143,667,240 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
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


class Inception_V1(incep._Inception_V1):
    """
    Inception v1, also known as GoogLeNet, is a deep convolutional neural network
    architecture designed for image classification. It introduces an "Inception
    module," which uses multiple convolutional filters of different sizes in
    parallel to capture various features at different scales. This architecture
    reduces computational costs by using 1x1 convolutions to decrease the number
    of input channels. Inception v1 achieved state-of-the-art results on the
    ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    6,990,272 weights, 8,280 biases -> 6,998,552 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V1()
    ```
    Arguments
    ---------
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


class Inception_V2(incep._Inception_V2):
    """
    Inception v2, an improvement of the original Inception architecture,
    enhances computational efficiency and accuracy in deep learning models.
    It introduces the factorization of convolutions and additional
    normalization techniques to reduce the number of parameters and improve
    training stability. These modifications allow for deeper and more
    complex neural networks with improved performance.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    24,974,688 weights, 20,136 biases -> 24,994,824 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V2_TypeA(),
    IncepBlock.V2_TypeB(),
    IncepBlock.V2_TypeC(),
    IncepBlock.V2_Redux()
    ```
    Arguments
    ---------
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


class Inception_V3(incep._Inception_V3):
    """
    Inception v3, an enhancement of Inception v2, further improves
    computational efficiency and accuracy in deep learning models.
    It includes advanced factorization of convolutions, improved grid
    size reduction techniques, extensive Batch Normalization, and
    label smoothing to prevent overfitting. These modifications enable
    deeper and more complex neural networks with significantly
    enhanced performance and robustness.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    25,012,960 weights, 20,136 biases -> 25,033,096 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V2_TypeA(),
    IncepBlock.V2_TypeB(),
    IncepBlock.V2_TypeC(),
    IncepBlock.V2_Redux()
    ```
    Arguments
    ---------
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


class Inception_V4(incep._Inception_V4):
    """
    Inception v4, an enhancement of Inception v3, improves computational
    efficiency and accuracy. It includes sophisticated convolution
    factorization, refined grid size reduction, extensive Batch
    Normalization, and label smoothing. These advancements enable deeper
    and more robust neural networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    42,641,952 weights, 32,584 biases -> 42,674,536 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V4_Stem(),
    IncepBlock.V4_TypeA(),
    IncepBlock.V4_TypeB(),
    IncepBlock.V4_TypeC(),
    IncepBlock.V4_ReduxA(),
    IncepBlock.V4_ReduxB()
    ```
    Arguments
    ---------
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


class InceptionRes_V1(incep._InceptionRes_V1):
    """
    Inception-ResNet v1 combines Inception modules with residual connections,
    improving computational efficiency and accuracy. This architecture uses
    convolution factorization, optimized grid size reduction, extensive
    Batch Normalization, and label smoothing, resulting in deeper and more
    robust neural networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    21,611,648 weights, 33,720 biases -> 21,645,368 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepResBlock.V1_Stem(),
    IncepResBlock.V1_TypeA(),
    IncepResBlock.V1_TypeB(),
    IncepResBlock.V1_TypeC(),
    IncepResBlock.V1_Redux(),

    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
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


class InceptionRes_V2(incep._InceptionRes_V2):
    """
    Inception-ResNet v2 enhances v1 with a deeper architecture and
    improved residual blocks for better performance. It features refined
    convolution factorization, more extensive Batch Normalization, and
    advanced grid size reduction.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    34,112,608 weights, 43,562 biases -> 34,156,170 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepResBlock.V2_TypeA(),
    IncepResBlock.V2_TypeB(),
    IncepResBlock.V2_TypeC(),
    IncepResBlock.V2_Redux(),

    IncepBlock.V4_Stem(),
    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
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


class ResNet_18(resnet._ResNet_18):
    """
    ResNet-18 is a 18-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    11,688,512 weights, 5,800 biases -> 11,694,312 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Basic()
    ```
    Arguments
    ---------
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


class ResNet_34(resnet._ResNet_34):
    """
    ResNet-34 is a 34-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    21,796,672 weights, 9,512 biases -> 21,806,184 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Basic()
    ```
    Arguments
    ---------
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


class ResNet_50(resnet._ResNet_50):
    """
    ResNet-50 is a 50-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    25,556,032 weights, 27,560 biases -> 25,583,592 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
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


class ResNet_101(resnet._ResNet_101):
    """
    ResNet-101 is a 101-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    44,548,160 weights, 53,672 biases -> 44,601,832 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
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


class ResNet_152(resnet._ResNet_152):
    """
    ResNet-152 is a 152-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    60,191,808 weights, 76,712 biases -> 60,268,520 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
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


class ResNet_200(resnet._ResNet_200):
    """
    ResNet-200 is a 200-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    64,668,864 weights, 89,000 biases -> 64,757,864 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
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


class ResNet_269(resnet._ResNet_269):
    """
    ResNet-269 is a 269-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    102,068,416 weights, 127,400 biases -> 102,195,816 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
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


class ResNet_1001(resnet._ResNet_1001):
    """
    ResNet-1001 is a 1001-layer deep neural network that uses residual
    blocks to improve training by learning residuals, helping prevent
    vanishing gradients and enabling better performance in image
    recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    159,884,992 weights, 208,040 biases -> 160,093,032 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
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


class XceptionNet(incep._Xception):
    """
    XceptionNet enhances the Inception architecture by replacing standard
    convolutions with depthwise separable convolutions, making it more
    efficient and effective at feature extraction. This design reduces
    the number of parameters and computations while maintaining or
    improving model accuracy on complex tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    22,113,984 weights, 50,288 biases -> 22,164,272 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    XceptionBlock.Entry(),
    XceptionBlock.Middle(),
    XceptionBlock.Exit(),

    SeparableConv2D()
    ```
    Arguments
    ---------
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


class MobileNet_V1(mobile._Mobile_V1):
    """
    MobileNet-V1 uses depthwise separable convolutions to significantly
    reduce the number of parameters and computational cost, making it
    highly efficient for mobile and embedded devices. It balances
    accuracy and efficiency through adjustable width and resolution
    multipliers.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    4,230,976 weights, 11,944 biases -> 4,242,920 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    SeparableConv2D()
    ```
    Arguments
    ---------
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


class MobileNet_V2(mobile._Mobile_V2):
    """
    MobileNet-V2 builds on the efficiency of its predecessor by introducing
    inverted residuals and linear bottlenecks, further reducing
    computational cost and enhancing performance on mobile and embedded
    devices. It continues to balance accuracy and efficiency while allowing
    for flexible adjustments through width and resolution multipliers.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    8,418,624 weights, 19,336 biases -> 8,437,960 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    ```
    Arguments
    ---------
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


class MobileNet_V3_Small(mobile._Mobile_V3_Small):
    """
    MobileNet-V3-Small improves on its predecessors by incorporating
    squeeze-and-excitation (SE) modules and the hard-swish activation,
    specifically designed to further reduce computational cost and optimize
    performance on resource-constrained mobile devices. It strikes a balance
    between accuracy and efficiency, with flexible width and resolution
    adjustments tailored for smaller models.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    32,455,856 weights, 326,138 biases -> 32,781,994 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    MobileNetBlock.InvRes_SE()
    ```
    Arguments
    ---------
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
    `dropout_rate` : float, default=0.2
        Dropout rate
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew, et al. “Searching for MobileNetV3.” Proceedings
    of the IEEE/CVF International Conference on Computer Vision (ICCV),
    2019, pp. 1314-1324.

    """


class MobileNet_V3_Large(mobile._Mobile_V3_Large):
    """
    MobileNet-V3-Large enhances its predecessors by integrating
    squeeze-and-excitation(SE) modules and the hard-swish activation,
    designed to boost performance while minimizing computational cost
    on mobile devices. It provides a balance of accuracy and efficiency,
    with flexible width and resolution adjustments optimized for larger,
    more powerful models.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    167,606,960 weights, 1,136,502 biases -> 168,743,462 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    MobileNetBlock.InvRes_SE()
    ```
    Arguments
    ---------
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
    `dropout_rate` : float, default=0.2
        Dropout rate
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew, et al. “Searching for MobileNetV3.” Proceedings
    of the IEEE/CVF International Conference on Computer Vision (ICCV),
    2019, pp. 1314-1324.

    """


class SE_ResNet_50(resnet._SE_ResNet_50):
    """
    SE-ResNet is a deep neural network that extends the ResNet
    architecture by integrating Squeeze-and-Excitation blocks.
    These blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the
    network.

    ResNet-50 is the base network for this SE-augmented version.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    35,615,808 weights, 46,440 biases -> 35,662,248 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()
    ```
    Arguments
    ---------
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
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class SE_ResNet_152(resnet._SE_ResNet_152):
    """
    SE-ResNet is a deep neural network that extends the ResNet
    architecture by integrating Squeeze-and-Excitation blocks.
    These blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the
    network.

    ResNet-152 is the base network for this SE-augmented version.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    86,504,512 weights, 136,552 biases -> 86,641,064 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()
    ```
    Arguments
    ---------
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
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class SE_InceptionRes_V2(incep._SE_InceptionRes_V2):
    """
    SE-InceptionResNet v2 is a deep neural network that extends the
    Inception-ResNet v2 architecture by integrating Squeeze-and-Excitation (SE)
    blocks. These SE blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the network by
    adaptively recalibrating the channel-wise feature responses.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    58,794,080 weights, 80,762 biases -> 58,874,842 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    # These blocks are SE-augmented
    IncepResBlock.V2_TypeA(),
    IncepResBlock.V2_TypeB(),
    IncepResBlock.V2_TypeC(),
    IncepResBlock.V2_Redux(),

    IncepBlock.V4_Stem(),
    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
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
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class DenseNet_121(dense._DenseNet_121): ...


class DenseNet_169(dense._DenseNet_169): ...


class DenseNet_201(dense._DenseNet_201): ...


class DenseNet_264(dense._DenseNet_264): ...
