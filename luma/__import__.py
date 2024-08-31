from luma.core.main import Luma
from luma.core.base import ModelBase, ParadigmBase, MetricBase, VisualBase
from luma.core.super import Estimator, Transformer, Optimizer, Evaluator, Visualizer
from luma.core.super import Supervised, Unsupervised, Distance

from luma.interface.exception import (
    NotFittedError,
    NotConvergedError,
    UnsupportedParameterError,
    ModelExtensionError,
    InvalidRangeError,
)
from luma.interface.typing import (
    TensorLike,
    Matrix,
    Vector,
    Tensor,
    Scalar,
    ClassType,
    LayerLike,
)
from luma.interface.util import (
    DecisionTreeNode,
    NearestNeighbors,
    SilhouetteUtil,
    DBUtil,
    KernelUtil,
    InitUtil,
    Clone,
    ParamRange,
    TrainProgress,
)

from luma.classifier.discriminant import (
    LDAClassifier,
    QDAClassifier,
    RDAClassifier,
    KDAClassifier,
)
from luma.classifier.logistic import LogisticRegressor, SoftmaxRegressor
from luma.classifier.naive_bayes import GaussianNaiveBayes, BernoulliNaiveBayes
from luma.classifier.svm import SVC, KernelSVC
from luma.classifier.tree import DecisionTreeClassifier
from luma.classifier.neighbors import (
    KNNClassifier,
    AdaptiveKNNClassifier,
    WeightedKNNClassifier,
)

from luma.clustering.kmeans import (
    KMeansClustering,
    KMeansClusteringPlus,
    KMediansClustering,
)
from luma.clustering.kmeans import KMedoidsClustering, MiniBatchKMeansClustering
from luma.clustering.kmeans import FuzzyCMeansClustering
from luma.clustering.hierarchy import AgglomerativeClustering, DivisiveClustering
from luma.clustering.spectral import SpectralClustering, NormalizedSpectralClustering
from luma.clustering.spectral import (
    HierarchicalSpectralClustering,
    AdaptiveSpectralClustering,
)
from luma.clustering.density import DBSCAN, OPTICS, DENCLUE, MeanShiftClustering
from luma.clustering.affinity import AffinityPropagation, AdaptiveAffinityPropagation
from luma.clustering.affinity import KernelAffinityPropagation
from luma.clustering.mixture import GaussianMixture, MultinomialMixture

from luma.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from luma.ensemble.vote import VotingClassifier, VotingRegressor
from luma.ensemble.bagging import BaggingClassifier, BaggingRegressor
from luma.ensemble.boost import AdaBoostClassifier, AdaBoostRegressor
from luma.ensemble.boost import GradientBoostingClassifier, GradientBoostingRegressor
from luma.ensemble.stack import StackingClassifier, StackingRegressor

from luma.neural.base import Layer, Loss, Initializer, Scheduler, NeuralModel
from luma.neural.single import PerceptronClassifier, PerceptronRegressor
from luma.neural.layer import (
    Conv1D,
    Conv2D,
    Conv3D,
    DepthConv1D,
    DepthConv2D,
    DepthConv3D,
    Pool1D,
    Pool2D,
    Pool3D,
    GlobalAvgPool1D,
    GlobalAvgPool2D,
    GlobalAvgPool3D,
    AdaptiveAvgPool1D,
    AdaptiveAvgPool2D,
    AdaptiveAvgPool3D,
    LpPool1D,
    LpPool2D,
    LpPool3D,
    Dense,
    Dropout,
    Dropout1D,
    Dropout2D,
    Dropout3D,
    Flatten,
    Activation,
    BatchNorm1D,
    BatchNorm2D,
    BatchNorm3D,
    LocalResponseNorm,
    LayerNorm,
    Identity,
    Sequential,
)
from luma.neural.optimizer import (
    SGDOptimizer,
    MomentumOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    AdaGradOptimizer,
    AdaDeltaOptimizer,
    AdaMaxOptimizer,
    AdamWOptimizer,
    NAdamOptimizer,
)
from luma.neural.loss import CrossEntropy, BinaryCrossEntropy, MSELoss
from luma.neural.loss import HingeLoss, HuberLoss, KLDivergenceLoss, NLLLoss
from luma.neural.init import KaimingInit, XavierInit
from luma.neural.scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from luma.neural.block import (
    ConvBlock1D,
    ConvBlock2D,
    ConvBlock3D,
    SeparableConv1D,
    SeparableConv2D,
    SeparableConv3D,
    DenseBlock,
    SEBlock1D,
    SEBlock2D,
    SEBlock3D,
    IncepBlock,
    IncepResBlock,
    ResNetBlock,
    XceptionBlock,
    MobileNetBlock,
)
from luma.neural.model import (
    SimpleMLP,
    SimpleCNN,
    LeNet_1,
    LeNet_4,
    LeNet_5,
    AlexNet,
    ZFNet,
    VGGNet_11,
    VGGNet_13,
    VGGNet_16,
    VGGNet_19,
    Inception_V1,
    Inception_V2,
    Inception_V3,
    Inception_V4,
    InceptionResNet_V1,
    InceptionResNet_V2,
    ResNet_18,
    ResNet_34,
    ResNet_50,
    ResNet_101,
    ResNet_152,
    ResNet_200,
    ResNet_1001,
    XceptionNet,
    MobileNet_V1,
    MobileNet_V2,
    MobileNet_V3_Small,
    MobileNet_V3_Large,
)
from luma.neural.autoprop import LayerNode, LayerGraph

from luma.metric.classification import Accuracy, Precision, Recall, F1Score, Specificity
from luma.metric.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
    RSquaredScore,
    AdjustedRSquaredScore,
)
from luma.metric.clustering import SilhouetteCoefficient, DaviesBouldin, Inertia
from luma.metric.distance import Euclidean, Manhattan, Chebyshev, Minkowski
from luma.metric.distance import CosineSimilarity, Correlation, Mahalanobis

from luma.model_selection.split import TrainTestSplit, BatchGenerator
from luma.model_selection.search import GridSearchCV, RandomizedSearchCV
from luma.model_selection.cv import CrossValidator
from luma.model_selection.fold import KFold, StratifiedKFold

from luma.preprocessing.scaler import StandardScaler, MinMaxScaler
from luma.preprocessing.encoder import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    LabelBinarizer,
    LabelSmoothing,
)
from luma.preprocessing.imputer import SimpleImputer, KNNImputer, HotDeckImputer
from luma.preprocessing.outlier import LocalOutlierFactor
from luma.preprocessing.image import (
    ImageTransformer,
    Resize,
    CenterCrop,
    Normalize,
    RandomCrop,
    RandomFlip,
    RandomRotate,
    RandomShift,
)

from luma.reduction.linear import PCA, KernelPCA, LDA, KDA, CCA
from luma.reduction.linear import TruncatedSVD, FactorAnalysis
from luma.reduction.manifold import TSNE, SammonMapping, LaplacianEigenmap
from luma.reduction.manifold import MDS, MetricMDS, LandmarkMDS
from luma.reduction.manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from luma.reduction.manifold import Isomap, ConformalIsomap
from luma.reduction.selection import SBS, SFS, RFE

from luma.regressor.linear import LinearRegressor, LassoRegressor, RidgeRegressor
from luma.regressor.linear import (
    ElasticNetRegressor,
    KernelRidgeRegressor,
    BayesianRidgeRegressor,
)
from luma.regressor.poly import PolynomialRegressor
from luma.regressor.general import PoissonRegressor
from luma.regressor.general import NegativeBinomialRegressor
from luma.regressor.general import GammaRegressor
from luma.regressor.general import BetaRegressor
from luma.regressor.general import InverseGaussianRegressor
from luma.regressor.svm import SVR, KernelSVR
from luma.regressor.tree import DecisionTreeRegressor
from luma.regressor.neighbors import (
    KNNRegressor,
    AdaptiveKNNRegressor,
    WeightedKNNRegressor,
)
from luma.regressor.robust import RANSAC, MLESAC

from luma.pipe.pipeline import Pipeline

from luma.visual.eda import (
    CorrelationBar,
    CorrelationHeatmap,
    JointPlot,
    MissingProportion,
)
from luma.visual.eval import (
    DecisionRegion,
    ClusterPlot,
    ROCCurve,
    PrecisionRecallCurve,
)
from luma.visual.eval import ConfusionMatrix, ResidualPlot, LearningCurve
from luma.visual.eval import ValidationCurve, InertiaPlot

from luma.migrate.port import ModelPorter


if __name__ == "__main__":

    # ------------------- [ luma.core ] ------------------------
    Luma

    ModelBase, ParadigmBase, MetricBase, VisualBase

    Estimator, Transformer, Optimizer, Evaluator, Visualizer,
    Supervised, Unsupervised, Distance

    # ----------------- [ luma.interface ] ---------------------
    NotFittedError, NotConvergedError,
    UnsupportedParameterError, ModelExtensionError,
    InvalidRangeError

    TensorLike, Matrix, Vector, Tensor, Scalar, ClassType,
    LayerLike

    DecisionTreeNode, NearestNeighbors,
    SilhouetteUtil, DBUtil, KernelUtil, InitUtil, Clone,
    ParamRange, TrainProgress

    # ----------------- [ luma.classifier ] --------------------
    LDAClassifier, QDAClassifier, RDAClassifier, KDAClassifier

    LogisticRegressor, SoftmaxRegressor

    GaussianNaiveBayes, BernoulliNaiveBayes

    SVC, KernelSVC

    DecisionTreeClassifier

    KNNClassifier, AdaptiveKNNClassifier, WeightedKNNClassifier

    # ----------------- [ luma.clustering ] --------------------
    KMeansClustering, KMeansClusteringPlus, KMediansClustering,
    KMedoidsClustering, MiniBatchKMeansClustering,
    FuzzyCMeansClustering

    AgglomerativeClustering, DivisiveClustering

    SpectralClustering, NormalizedSpectralClustering,
    HierarchicalSpectralClustering, AdaptiveSpectralClustering

    DBSCAN, OPTICS, DENCLUE, MeanShiftClustering

    AffinityPropagation, AdaptiveAffinityPropagation,
    KernelAffinityPropagation

    GaussianMixture, MultinomialMixture

    # ------------------ [ luma.ensemble ] ---------------------
    RandomForestClassifier, RandomForestRegressor

    VotingClassifier, VotingRegressor

    BaggingClassifier, BaggingRegressor

    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor

    StackingClassifier, StackingRegressor

    # ------------------- [ luma.neural ] ----------------------
    PerceptronClassifier, PerceptronRegressor

    SGDOptimizer, MomentumOptimizer, RMSPropOptimizer,
    AdamOptimizer, AdaGradOptimizer, AdaDeltaOptimizer,
    AdaMaxOptimizer, AdamWOptimizer, NAdamOptimizer

    Layer, Loss, Initializer, Scheduler, NeuralModel

    Conv1D, Conv2D, Conv3D,
    DepthConv1D, DepthConv2D, DepthConv3D,
    Pool1D, Pool2D, Pool3D,
    GlobalAvgPool1D, GlobalAvgPool2D, GlobalAvgPool3D,
    AdaptiveAvgPool1D, AdaptiveAvgPool2D, AdaptiveAvgPool3D,
    LpPool1D, LpPool2D, LpPool3D
    Dropout, Dropout1D, Dropout2D, Dropout3D,
    BatchNorm1D, BatchNorm2D, BatchNorm3D,
    LocalResponseNorm, LayerNorm,
    Dense, Flatten, Activation, Identity,
    Sequential

    ConvBlock1D, ConvBlock2D, ConvBlock3D,
    SeparableConv1D, SeparableConv2D, SeparableConv3D,
    DenseBlock, SEBlock1D, SEBlock2D, SEBlock3D,
    IncepBlock, IncepResBlock, ResNetBlock, XceptionBlock,
    MobileNetBlock

    LayerNode, LayerGraph

    CrossEntropy, BinaryCrossEntropy, MSELoss, HingeLoss,
    HuberLoss, KLDivergenceLoss, NLLLoss

    KaimingInit, XavierInit

    StepLR, ExponentialLR, CosineAnnealingLR, CyclicLR,
    OneCycleLR, ReduceLROnPlateau

    SimpleMLP, SimpleCNN,
    LeNet_1, LeNet_4, LeNet_5,
    AlexNet, ZFNet,
    VGGNet_11, VGGNet_13, VGGNet_16, VGGNet_19,
    Inception_V1, Inception_V2, Inception_V3, Inception_V4,
    InceptionResNet_V1, InceptionResNet_V2, XceptionNet,
    ResNet_18, ResNet_34, ResNet_50, ResNet_101, ResNet_152,
    ResNet_200, ResNet_1001,
    MobileNet_V1, MobileNet_V2, MobileNet_V3_Small,
    MobileNet_V3_Large,

    # ------------------- [ luma.metric ] ----------------------
    Accuracy, Precision, Recall, F1Score, Specificity

    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
    MeanAbsolutePercentageError, RSquaredScore,
    AdjustedRSquaredScore

    SilhouetteCoefficient, DaviesBouldin, Inertia

    Euclidean, Manhattan, Chebyshev, Minkowski,
    CosineSimilarity, Correlation, Mahalanobis

    # --------------- [ luma.module_selection ] ----------------
    TrainTestSplit, BatchGenerator

    GridSearchCV, RandomizedSearchCV

    CrossValidator

    KFold, StratifiedKFold

    # ---------------- [ luma.preprocessing ] ------------------
    StandardScaler, MinMaxScaler

    OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer,
    LabelSmoothing

    SimpleImputer, KNNImputer, HotDeckImputer

    LocalOutlierFactor

    ImageTransformer,
    Resize, CenterCrop, Normalize, RandomCrop, RandomFlip,
    RandomRotate, RandomShift,

    # ----------------- [ luma.reduction ] ---------------------
    PCA, KernelPCA, LDA, KDA, CCA, TruncatedSVD, FactorAnalysis

    TSNE, SammonMapping, LaplacianEigenmap,
    MDS, MetricMDS, LandmarkMDS,
    LLE, ModifiedLLE, HessianLLE, LTSA,
    Isomap, ConformalIsomap

    SBS, SFS, RFE

    # ----------------- [ luma.regressor ] ---------------------
    LinearRegressor, RidgeRegressor, LassoRegressor,
    ElasticNetRegressor, KernelRidgeRegressor,
    BayesianRidgeRegressor

    PolynomialRegressor

    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor,
    BetaRegressor, InverseGaussianRegressor

    SVR, KernelSVR

    DecisionTreeRegressor

    KNNRegressor, AdaptiveKNNRegressor, WeightedKNNRegressor

    RANSAC, MLESAC

    # -------------------- [ luma.pipe ] -----------------------
    Pipeline

    # ------------------- [ luma.visual ] ----------------------
    CorrelationBar, CorrelationHeatmap, JointPlot,
    MissingProportion

    DecisionRegion, ClusterPlot, ROCCurve, PrecisionRecallCurve,
    ConfusionMatrix, ResidualPlot, LearningCurve,
    ValidationCurve, InertiaPlot

    # ------------------ [ luma.migrate ] ----------------------
    ModelPorter
