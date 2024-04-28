from luma.core.main import Luma
from luma.core.base import ModelBase, ParadigmBase, MetricBase, VisualBase
from luma.core.super import (
    Estimator,
    Transformer,
    Optimizer,
    Evaluator,
    Visualizer,
    NeuralModel,
)
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
from luma.neural.single import PerceptronClassifier, PerceptronRegressor
from luma.neural.base import Layer, Loss, Initializer
from luma.neural.layer import (
    Convolution,
    Pooling,
    Dense,
    Dropout,
    Flatten,
    Activation,
    Sequential,
)
from luma.neural.loss import CrossEntropy, BinaryCrossEntropy, MSELoss
from luma.neural.init import KaimingInit, XavierInit

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
from luma.preprocessing.encoder import OneHotEncoder, LabelEncoder, OrdinalEncoder
from luma.preprocessing.encoder import LabelBinarizer
from luma.preprocessing.imputer import SimpleImputer, KNNImputer, HotDeckImputer
from luma.preprocessing.outlier import LocalOutlierFactor

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
from luma.visual.evaluation import (
    DecisionRegion,
    ClusterPlot,
    ROCCurve,
    PrecisionRecallCurve,
)
from luma.visual.evaluation import ConfusionMatrix, ResidualPlot, LearningCurve
from luma.visual.evaluation import ValidationCurve, InertiaPlot

from luma.migrate.port import ModelPorter


if __name__ == "__main__":

    # ------------------- [ luma.core ] ------------------------
    Luma

    ModelBase, ParadigmBase, MetricBase, VisualBase

    Estimator, Transformer, Optimizer, Evaluator, Visualizer,
    Supervised, Unsupervised, Distance, NeuralModel

    # ----------------- [ luma.interface ] ---------------------
    NotFittedError, NotConvergedError,
    UnsupportedParameterError, ModelExtensionError,
    InvalidRangeError

    TensorLike, Matrix, Vector, Tensor, Scalar, ClassType

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

    Layer, Loss, Initializer

    Convolution, Pooling, Dense, Dropout, Flatten, Activation,
    Sequential

    CrossEntropy, BinaryCrossEntropy, MSELoss

    KaimingInit, XavierInit

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

    OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer

    SimpleImputer, KNNImputer, HotDeckImputer

    LocalOutlierFactor

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
