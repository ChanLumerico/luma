import sys, time
build_start = time.time()


# ----------------------------------------[ Build Start ]---------------------------------------- #
from LUMA.Core.Main import LUMA
from LUMA.Interface.Super import _Estimator, _Transformer, _Evaluator, _Visualizer
from LUMA.Interface.Super import _Supervised, _Unsupervised, _Distance
from LUMA.Interface.Exception import NotFittedError, UnsupportedParameterError

from LUMA.Preprocessing.Scaler import StandardScaler
from LUMA.Preprocessing.Scaler import MinMaxScaler

from LUMA.Reduction.Linear import PCA, LDA, TruncatedSVD, FactorAnalysis
from LUMA.Reduction.Nonlinear import KernelPCA
from LUMA.Reduction.Manifold import TSNE, SammonMapping, LaplacianEigenmap
from LUMA.Reduction.Manifold import MDS, MetricMDS, LandmarkMDS
from LUMA.Reduction.Manifold import LLE, ModifiedLLE, HessianLLE, LTSA
from LUMA.Reduction.Manifold import Isomap, ConformalIsomap

from LUMA.ModelSelection.Split import TrainTestSplit
from LUMA.ModelSelection.Search import GridSearchCV

from LUMA.Regressor.Linear import RidgeRegressor
from LUMA.Regressor.Linear import LassoRegressor
from LUMA.Regressor.Linear import ElasticNetRegressor
from LUMA.Regressor.Polynomial import PolynomialRegressor
from LUMA.Regressor.General import PoissonRegressor
from LUMA.Regressor.General import NegativeBinomialRegressor
from LUMA.Regressor.General import GammaRegressor
from LUMA.Regressor.General import BetaRegressor
from LUMA.Regressor.General import InverseGaussianRegressor
from LUMA.Regressor.SVM import SVR, KernelSVR

from LUMA.Classifier.NaiveBayes import GaussianNaiveBayes
from LUMA.Classifier.NaiveBayes import BernoulliNaiveBayes
from LUMA.Classifier.Logistic import LogisticRegressor
from LUMA.Classifier.Logistic import SoftmaxRegressor
from LUMA.Classifier.SVM import SVC, KernelSVC
from LUMA.Classifier.Tree import DecisionTreeClassifier

from LUMA.Clustering.KMeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering

from LUMA.Metric.Classification import Accuracy, Precision, Recall, F1Score
from LUMA.Metric.Classification import Specificity, AUCCurveROC, Complex
from LUMA.Metric.Regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from LUMA.Metric.Regression import MeanAbsolutePercentageError, RSquaredScore, Complex
from LUMA.Metric.Distance import Euclidean, Manhattan, Chebyshev, Minkowski
from LUMA.Metric.Distance import CosineSimilarity, Correlation, Mahalanobis

from LUMA.Visual.EDA import CorrelationHeatMap, CorrelationBar, JointPlot
from LUMA.Visual.Region import DecisionRegion, ClusteredRegion
# ---------------------------------------[ Build End ]------------------------------------------- #


def init():
    
    # LUMA.Core.Main
    LUMA
    
    # LUMA.Interface.Super
    _Estimator, _Transformer, _Evaluator, _Visualizer
    _Supervised, _Unsupervised, _Distance
    
    # LUMA.Interface.Exception
    NotFittedError, UnsupportedParameterError
    
    # LUMA.Preprocessing.Scaler
    StandardScaler, MinMaxScaler
    
    # LUMA.Reduction
    PCA, LDA, TruncatedSVD, FactorAnalysis
    KernelPCA
    TSNE, SammonMapping, LaplacianEigenmap
    MDS, MetricMDS, LandmarkMDS
    LLE, ModifiedLLE, HessianLLE, LTSA
    Isomap, ConformalIsomap
    
    # LUMA.ModelSelection
    TrainTestSplit
    GridSearchCV
    
    # LUMA.Regression
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    PolynomialRegressor
    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor, 
    BetaRegressor, InverseGaussianRegressor
    SVR, KernelSVR
    
    # LUMA.Classifier
    GaussianNaiveBayes, BernoulliNaiveBayes
    LogisticRegressor, SoftmaxRegressor
    SVC, KernelSVC
    DecisionTreeClassifier
    
    # LUMA.Clustering
    KMeansClustering, KMeansClusteringPlus, KMediansClustering
    
    # LUMA.Metric
    Accuracy, Precision, Recall, F1Score, Specificity, AUCCurveROC, Complex
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, 
    RootMeanSquaredError, RSquaredScore, Complex
    Euclidean, Manhattan, Chebyshev, Minkowski, 
    CosineSimilarity, Correlation, Mahalanobis
    
    # LUMA.Visual
    DecisionRegion, ClusteredRegion
    CorrelationHeatMap, CorrelationBar, JointPlot


init()
build_end = time.time()
print(f'Build Succeeded! ({build_end - build_start:.3f} sec)', file=sys.stderr)
