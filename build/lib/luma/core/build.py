import sys, time
build_start = time.time()


# ----------------------------------------[ Build Start ]---------------------------------------- #
from luma.core.main import LUMA
from luma.interface.super import Estimator, Transformer, Evaluator, Visualizer
from luma.interface.super import Supervised, Unsupervised, Distance
from luma.interface.exception import NotFittedError, NotConvergedError, UnsupportedParameterError
from luma.interface.util import TreeNode, NearestNeighbors

from luma.preprocessing.scaler import StandardScaler
from luma.preprocessing.scaler import MinMaxScaler

from luma.reduction.linear import PCA, LDA, TruncatedSVD, FactorAnalysis
from luma.reduction.nonlinear import KernelPCA
from luma.reduction.manifold import TSNE, SammonMapping, LaplacianEigenmap
from luma.reduction.manifold import MDS, MetricMDS, LandmarkMDS
from luma.reduction.manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from luma.reduction.manifold import Isomap, ConformalIsomap

from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV

from luma.regressor.linear import RidgeRegressor
from luma.regressor.linear import LassoRegressor
from luma.regressor.linear import ElasticNetRegressor
from luma.regressor.polynomial import PolynomialRegressor
from luma.regressor.general import PoissonRegressor
from luma.regressor.general import NegativeBinomialRegressor
from luma.regressor.general import GammaRegressor
from luma.regressor.general import BetaRegressor
from luma.regressor.general import InverseGaussianRegressor
from luma.regressor.svm import SVR, KernelSVR
from luma.regressor.tree import DecisionTreeRegressor

from luma.classifier.naive_bayes import GaussianNaiveBayes
from luma.classifier.naive_bayes import BernoulliNaiveBayes
from luma.classifier.logistic import LogisticRegressor
from luma.classifier.logistic import SoftmaxRegressor
from luma.classifier.svm import SVC, KernelSVC
from luma.classifier.tree import DecisionTreeClassifier

from luma.clustering.kmeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering

from luma.ensemble.forest import RandomForestClassifier, RandomForestRegressor

from luma.metric.classification import Accuracy, Precision, Recall, F1Score
from luma.metric.classification import Specificity, AUCCurveROC, Complex
from luma.metric.regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from luma.metric.regression import MeanAbsolutePercentageError, RSquaredScore, Complex
from luma.metric.distance import Euclidean, Manhattan, Chebyshev, Minkowski
from luma.metric.distance import CosineSimilarity, Correlation, Mahalanobis

from luma.visual.eda import CorrelationHeatMap, CorrelationBar, JointPlot
from luma.visual.region import DecisionRegion, ClusteredRegion
# ---------------------------------------[ Build End ]------------------------------------------- #


def init():
    
    # core.main
    LUMA

    # interface.super
    Estimator, Transformer, Evaluator, Visualizer
    Supervised, Unsupervised, Distance
    
    # interface.exception
    NotFittedError, NotConvergedError, UnsupportedParameterError
    
    # interface.util
    TreeNode, NearestNeighbors
    
    # preprocessing.scaler
    StandardScaler, MinMaxScaler
    
    # reduction
    PCA, LDA, TruncatedSVD, FactorAnalysis
    KernelPCA
    TSNE, SammonMapping, LaplacianEigenmap
    MDS, MetricMDS, LandmarkMDS
    LLE, ModifiedLLE, HessianLLE, LTSA
    Isomap, ConformalIsomap
    
    # model_selection
    TrainTestSplit
    GridSearchCV
    
    # regression
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    PolynomialRegressor
    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor, 
    BetaRegressor, InverseGaussianRegressor
    SVR, KernelSVR
    DecisionTreeRegressor
    
    # classifier
    GaussianNaiveBayes, BernoulliNaiveBayes
    LogisticRegressor, SoftmaxRegressor
    SVC, KernelSVC
    DecisionTreeClassifier
    
    # clustering
    KMeansClustering, KMeansClusteringPlus, KMediansClustering
    
    # ensemble
    RandomForestClassifier, RandomForestRegressor
    
    # metric
    Accuracy, Precision, Recall, F1Score, Specificity, AUCCurveROC, Complex
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, 
    RootMeanSquaredError, RSquaredScore, Complex
    Euclidean, Manhattan, Chebyshev, Minkowski, 
    CosineSimilarity, Correlation, Mahalanobis
    
    # visual
    DecisionRegion, ClusteredRegion
    CorrelationHeatMap, CorrelationBar, JointPlot


init()
build_end = time.time()
print(f'Build Succeeded! ({build_end - build_start:.3f} sec)', file=sys.stderr)
