from luma.core.main import LUMA

from luma.interface.exception import NotFittedError, NotConvergedError
from luma.interface.exception import UnsupportedParameterError, ModelExtensionError
from luma.interface.super import Estimator, Transformer, Evaluator, Visualizer
from luma.interface.super import Supervised, Unsupervised, Distance
from luma.interface.util import Matrix, Vector, Scalar, TreeNode, NearestNeighbors
from luma.interface.util import SilhouetteUtil, DBUtil, KernelUtil
from luma.interface.util import Clone

from luma.classifier.logistic import LogisticRegressor, SoftmaxRegressor
from luma.classifier.naive_bayes import GaussianNaiveBayes, BernoulliNaiveBayes
from luma.classifier.svm import SVC, KernelSVC
from luma.classifier.tree import DecisionTreeClassifier
from luma.classifier.neighbors import KNNClassifier, AdaptiveKNNClassifier, WeightedKNNClassifier

from luma.clustering.kmeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering
from luma.clustering.kmeans import KMedoidsClustering, MiniBatchKMeansClustering
from luma.clustering.kmeans import FuzzyKMeansClustering
from luma.clustering.hierarchy import AgglomerativeClustering, DivisiveClustering
from luma.clustering.spectral import SpectralClustering, NormalizedSpectralClustering
from luma.clustering.spectral import HierarchicalSpectralClustering, AdaptiveSpectralClustering
from luma.clustering.density import DBSCAN, OPTICS, DENCLUE, MeanShiftClustering
from luma.clustering.affinity import AffinityPropagation, AdaptiveAffinityPropagation
from luma.clustering.affinity import KernelAffinityPropagation
from luma.clustering.mixture import GaussianMixture, MultinomialMixture

from luma.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from luma.ensemble.vote import VotingClassifier, VotingRegressor
from luma.ensemble.bagging import BaggingClassifier, BaggingRegressor
from luma.ensemble.boost import AdaBoostClassifier, AdaBoostRegressor

from luma.neural.single import PerceptronClassifier, PerceptronRegressor

from luma.metric.classification import Accuracy, Precision, Recall, F1Score, Specificity
from luma.metric.regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from luma.metric.regression import MeanAbsolutePercentageError, RSquaredScore
from luma.metric.clustering import SilhouetteCoefficient, DaviesBouldin
from luma.metric.distance import Euclidean, Manhattan, Chebyshev, Minkowski
from luma.metric.distance import CosineSimilarity, Correlation, Mahalanobis

from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV
from luma.model_selection.cv import CrossValidator

from luma.preprocessing.scaler import StandardScaler, MinMaxScaler
from luma.preprocessing.encoder import OneHotEncoder, LabelEncoder, OrdinalEncoder
from luma.preprocessing.encoder import LabelBinarizer
from luma.preprocessing.imputer import SimpleImputer, KNNImputer, HotDeckImputer
from luma.preprocessing.outlier import LocalOutlierFactor

from luma.reduction.linear import PCA, LDA, TruncatedSVD, FactorAnalysis
from luma.reduction.nonlinear import KernelPCA
from luma.reduction.manifold import TSNE, SammonMapping, LaplacianEigenmap
from luma.reduction.manifold import MDS, MetricMDS, LandmarkMDS
from luma.reduction.manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from luma.reduction.manifold import Isomap, ConformalIsomap
from luma.reduction.select import SBS, SFS, RFE

from luma.regressor.linear import LinearRegressor
from luma.regressor.linear import LassoRegressor, RidgeRegressor, ElasticNetRegressor
from luma.regressor.poly import PolynomialRegressor
from luma.regressor.general import PoissonRegressor
from luma.regressor.general import NegativeBinomialRegressor
from luma.regressor.general import GammaRegressor
from luma.regressor.general import BetaRegressor
from luma.regressor.general import InverseGaussianRegressor
from luma.regressor.svm import SVR, KernelSVR
from luma.regressor.tree import DecisionTreeRegressor
from luma.regressor.neighbors import KNNRegressor, AdaptiveKNNRegressor, WeightedKNNRegressor
from luma.regressor.robust import RANSAC, MLESAC

from luma.pipe.pipeline import Pipeline

from luma.visual.eda import CorrelationBar, CorrelationHeatMap, JointPlot, MissingProportion
from luma.visual.evaluation import DecisionRegion, ClusterPlot, ROCCurve, PrecisionRecallCurve
from luma.visual.evaluation import ConfusionMatrix, ResidualPlot, LearningCurve
from luma.visual.evaluation import ValidationCurve, ValidationHeatmap

from luma.migrate.port import ModelPorter


if __name__ == '__main__':
    
    # ------------------- [ luma.core ] ------------------------
    LUMA
    
    # ----------------- [ luma.interface ] ---------------------
    NotFittedError, NotConvergedError,
    UnsupportedParameterError, ModelExtensionError
    
    Estimator, Transformer, Evaluator, Visualizer,
    Supervised, Unsupervised, Distance
    
    Matrix, Vector, Scalar, TreeNode, NearestNeighbors,
    SilhouetteUtil, DBUtil, KernelUtil,
    Clone
    
    # ----------------- [ luma.classifier ] --------------------
    LogisticRegressor, SoftmaxRegressor
    
    GaussianNaiveBayes, BernoulliNaiveBayes
    
    SVC, KernelSVC
    
    DecisionTreeClassifier
    
    KNNClassifier, AdaptiveKNNClassifier, WeightedKNNClassifier
    
    # ----------------- [ luma.clustering ] --------------------
    KMeansClustering, KMeansClusteringPlus, KMediansClustering,
    KMedoidsClustering, MiniBatchKMeansClustering,
    FuzzyKMeansClustering
    
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
    
    AdaBoostClassifier, AdaBoostRegressor
    
    # ------------------- [ luma.neural ] ----------------------
    PerceptronClassifier, PerceptronRegressor
    
    # ------------------- [ luma.metric ] ----------------------
    Accuracy, Precision, Recall, F1Score, Specificity
    
    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
    MeanAbsolutePercentageError, RSquaredScore
    
    SilhouetteCoefficient, DaviesBouldin
    
    Euclidean, Manhattan, Chebyshev, Minkowski,
    CosineSimilarity, Correlation, Mahalanobis
    
    # --------------- [ luma.module_selection ] ----------------
    TrainTestSplit
    
    GridSearchCV
    
    CrossValidator
    
    # ---------------- [ luma.preprocessing ] ------------------
    StandardScaler, MinMaxScaler
    
    OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer
    
    SimpleImputer, KNNImputer, HotDeckImputer
    
    LocalOutlierFactor
    
    # ----------------- [ luma.reduction ] ---------------------
    PCA, LDA, TruncatedSVD, FactorAnalysis
    
    KernelPCA
    
    TSNE, SammonMapping, LaplacianEigenmap,
    MDS, MetricMDS, LandmarkMDS,
    LLE, ModifiedLLE, HessianLLE, LTSA,
    Isomap, ConformalIsomap
    
    SBS, SFS, RFE
    
    # ----------------- [ luma.regressor ] ---------------------
    LinearRegressor,
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    
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
    CorrelationBar, CorrelationHeatMap, JointPlot,
    MissingProportion
    
    DecisionRegion, ClusterPlot, ROCCurve, PrecisionRecallCurve, 
    ConfusionMatrix, ResidualPlot, LearningCurve,
    ValidationCurve, ValidationHeatmap
    
    # ------------------ [ luma.migrate ] ----------------------
    ModelPorter
