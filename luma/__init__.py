from luma.core.main import LUMA

from luma.interface.exception import NotFittedError, NotConvergedError, UnsupportedParameterError
from luma.interface.super import Estimator, Transformer, Evaluator, Visualizer
from luma.interface.super import Supervised, Unsupervised, Distance
from luma.interface.util import TreeNode, NearestNeighbors

from luma.classifier.logistic import LogisticRegressor, SoftmaxRegressor
from luma.classifier.naive_bayes import GaussianNaiveBayes, BernoulliNaiveBayes
from luma.classifier.svm import SVC, KernelSVC
from luma.classifier.tree import DecisionTreeClassifier
from luma.classifier.knn import KNNClassifier

from luma.clustering.kmeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering

from luma.ensemble.forest import RandomForestClassifier, RandomForestRegressor

from luma.metric.classification import Accuracy, Precision, Recall, F1Score
from luma.metric.classification import Specificity, AUCCurveROC, Complex
from luma.metric.regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from luma.metric.regression import MeanAbsolutePercentageError, RSquaredScore, Complex
from luma.metric.distance import Euclidean, Manhattan, Chebyshev, Minkowski
from luma.metric.distance import CosineSimilarity, Correlation, Mahalanobis

from luma.model_selection.split import TrainTestSplit
from luma.model_selection.search import GridSearchCV

from luma.preprocessing.scaler import StandardScaler
from luma.preprocessing.scaler import MinMaxScaler

from luma.reduction.linear import PCA, LDA, TruncatedSVD, FactorAnalysis
from luma.reduction.nonlinear import KernelPCA
from luma.reduction.manifold import TSNE, SammonMapping, LaplacianEigenmap
from luma.reduction.manifold import MDS, MetricMDS, LandmarkMDS
from luma.reduction.manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from luma.reduction.manifold import Isomap, ConformalIsomap

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

from luma.visual.eda import CorrelationBar, CorrelationHeatMap, JointPlot
from luma.visual.region import DecisionRegion, ClusteredRegion


if __name__ == '__main__':
    
    # luma.core
    LUMA
    
    # luma.interface
    NotFittedError, NotConvergedError, UnsupportedParameterError
    Estimator, Transformer, Evaluator, Visualizer
    Supervised, Unsupervised, Distance
    TreeNode, NearestNeighbors
    
    # luma.classifier
    LogisticRegressor, SoftmaxRegressor
    GaussianNaiveBayes, BernoulliNaiveBayes
    SVC, KernelSVC
    DecisionTreeClassifier
    KNNClassifier
    
    # luma.clustering
    KMeansClustering, KMeansClusteringPlus, KMediansClustering
    
    # luma.ensemble
    RandomForestClassifier, RandomForestRegressor
    
    # luma.metric
    Accuracy, Precision, Recall, F1Score, Specificity, AUCCurveROC, Complex
    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
    MeanAbsolutePercentageError, RSquaredScore, Complex
    Euclidean, Manhattan, Chebyshev, Minkowski
    CosineSimilarity, Correlation, Mahalanobis
    
    # luma.module_selection
    TrainTestSplit, GridSearchCV
    
    # luma.preprocessing
    StandardScaler, MinMaxScaler
    
    # luma.reduction
    PCA, LDA, TruncatedSVD, FactorAnalysis
    KernelPCA
    TSNE, SammonMapping, LaplacianEigenmap
    MDS, MetricMDS, LandmarkMDS
    LLE, ModifiedLLE, HessianLLE, LTSA
    Isomap, ConformalIsomap
    
    # luma.regression
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    PolynomialRegressor
    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor
    BetaRegressor, InverseGaussianRegressor
    SVR, KernelSVR
    DecisionTreeRegressor
    
    # luma.visual
    CorrelationBar, CorrelationHeatMap, JointPlot
    DecisionRegion, ClusteredRegion

