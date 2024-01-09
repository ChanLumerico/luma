from luma.core.main import LUMA

from luma.interface.exception import NotFittedError, NotConvergedError
from luma.interface.exception import UnsupportedParameterError, ModelExtensionError
from luma.interface.super import Estimator, Transformer, Evaluator, Visualizer
from luma.interface.super import Supervised, Unsupervised, Distance
from luma.interface.util import Matrix, Vector, Constant, TreeNode, NearestNeighbors
from luma.interface.util import SilhouetteUtil, DBUtil

from luma.classifier.logistic import LogisticRegressor, SoftmaxRegressor
from luma.classifier.naive_bayes import GaussianNaiveBayes, BernoulliNaiveBayes
from luma.classifier.svm import SVC, KernelSVC
from luma.classifier.tree import DecisionTreeClassifier
from luma.classifier.neighbors import KNNClassifier, AdaptiveKNNClassifier, WeightedKNNClassifier

from luma.clustering.kmeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering
from luma.clustering.kmeans import MiniBatchKMeansClustering
from luma.clustering.hierarchy import AgglomerativeClustering, DivisiveClustering
from luma.clustering.spectral import SpectralClustering, NormalizedSpectralClustering
from luma.clustering.spectral import HierarchicalSpectralClustering, AdaptiveSpectralClustering
from luma.clustering.density import DBSCAN
from luma.clustering.affinity import AffinityPropagation

from luma.ensemble.forest import RandomForestClassifier, RandomForestRegressor

from luma.metric.classification import Accuracy, Precision, Recall, F1Score
from luma.metric.classification import Specificity, AUCCurveROC, Complex
from luma.metric.regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from luma.metric.regression import MeanAbsolutePercentageError, RSquaredScore, Complex
from luma.metric.clustering import SilhouetteCoefficient, DaviesBouldin
from luma.metric.distance import Euclidean, Manhattan, Chebyshev, Minkowski
from luma.metric.distance import CosineSimilarity, Correlation, Mahalanobis

from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV

from luma.preprocessing.scaler import StandardScaler, MinMaxScaler
from luma.preprocessing.encoder import OneHotEncoder, LabelEncoder, OrdinalEncoder
from luma.preprocessing.imputer import SimpleImputer, KNNImputer, HotDeckImputer
from luma.preprocessing.outlier import LocalOutlierFactor

from luma.reduction.linear import PCA, LDA, TruncatedSVD, FactorAnalysis
from luma.reduction.nonlinear import KernelPCA
from luma.reduction.manifold import TSNE, SammonMapping, LaplacianEigenmap
from luma.reduction.manifold import MDS, MetricMDS, LandmarkMDS
from luma.reduction.manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from luma.reduction.manifold import Isomap, ConformalIsomap

from luma.regressor.linear import LinearRegressor
from luma.regressor.linear import LassoRegressor, RidgeRegressor, ElasticNetRegressor
from luma.regressor.polynomial import PolynomialRegressor
from luma.regressor.general import PoissonRegressor
from luma.regressor.general import NegativeBinomialRegressor
from luma.regressor.general import GammaRegressor
from luma.regressor.general import BetaRegressor
from luma.regressor.general import InverseGaussianRegressor
from luma.regressor.svm import SVR, KernelSVR
from luma.regressor.tree import DecisionTreeRegressor
from luma.regressor.neighbors import KNNRegressor, AdaptiveKNNRegressor, WeightedKNNRegressor

from luma.pipe.pipeline import Pipeline

from luma.visual.eda import CorrelationBar, CorrelationHeatMap, JointPlot, MissingProportion
from luma.visual.graph import GraphPlot
from luma.visual.result import DecisionRegion, ClusterPlot

from luma.migrate.port import ModelPorter


if __name__ == '__main__':
    
    # ------------------- [ luma.core ] ------------------------
    LUMA
    
    # ----------------- [ luma.interface ] ---------------------
    NotFittedError, NotConvergedError,
    UnsupportedParameterError, ModelExtensionError
    
    Estimator, Transformer, Evaluator, Visualizer,
    Supervised, Unsupervised, Distance
    
    Matrix, Vector, Constant, TreeNode, NearestNeighbors,
    SilhouetteUtil, DBUtil
    
    # ---------------- [ luma.classifier ] ---------------------
    LogisticRegressor, SoftmaxRegressor
    
    GaussianNaiveBayes, BernoulliNaiveBayes
    
    SVC, KernelSVC
    
    DecisionTreeClassifier
    
    KNNClassifier, AdaptiveKNNClassifier, WeightedKNNClassifier
    
    # ----------------- [ luma.clustering ] --------------------
    KMeansClustering, KMeansClusteringPlus, KMediansClustering,
    MiniBatchKMeansClustering
    
    AgglomerativeClustering, DivisiveClustering
    
    SpectralClustering, NormalizedSpectralClustering,
    HierarchicalSpectralClustering, AdaptiveSpectralClustering
    
    DBSCAN
    
    AffinityPropagation
    
    # ----------------- [ luma.ensemble ] ----------------------
    RandomForestClassifier, RandomForestRegressor
    
    # ------------------ [ luma.metric ] -----------------------
    Accuracy, Precision, Recall, F1Score, Specificity, 
    AUCCurveROC, Complex
    
    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
    MeanAbsolutePercentageError, RSquaredScore, Complex
    
    SilhouetteCoefficient, DaviesBouldin
    
    Euclidean, Manhattan, Chebyshev, Minkowski,
    CosineSimilarity, Correlation, Mahalanobis
    
    # -------------- [ luma.module_selection ] -----------------
    TrainTestSplit
    
    GridSearchCV
    
    # ---------------- [ luma.preprocessing ] ------------------
    StandardScaler, MinMaxScaler
    
    OneHotEncoder, LabelEncoder, OrdinalEncoder
    
    SimpleImputer, KNNImputer, HotDeckImputer
    
    LocalOutlierFactor
    
    # ----------------- [ luma.reduction ] ---------------------
    PCA, LDA, TruncatedSVD, FactorAnalysis
    
    KernelPCA
    
    TSNE, SammonMapping, LaplacianEigenmap,
    MDS, MetricMDS, LandmarkMDS,
    LLE, ModifiedLLE, HessianLLE, LTSA,
    Isomap, ConformalIsomap
    
    # ----------------- [ luma.regressor ] ---------------------
    LinearRegressor,
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    
    PolynomialRegressor
    
    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor,
    BetaRegressor, InverseGaussianRegressor
    
    SVR, KernelSVR
    
    DecisionTreeRegressor
    
    KNNRegressor, AdaptiveKNNRegressor, WeightedKNNRegressor
    
    # -------------------- [ luma.pipe ] -----------------------
    Pipeline
    
    # ------------------- [ luma.visual ] ----------------------
    CorrelationBar, CorrelationHeatMap, JointPlot,
    MissingProportion
    
    GraphPlot
    
    DecisionRegion, ClusterPlot
    
    # ------------------ [ luma.migrate ] ----------------------
    ModelPorter
