import sys, time
build_start = time.time()


# ----------------------------------------[ Build Start ]---------------------------------------- #
from LUMA.Core.Main import LUMA
from LUMA.Interface.Super import _Estimator, _Transformer, _Evaluator, _Visualizer
from LUMA.Interface.Super import _Supervised, _Unsupervised

from LUMA.Preprocessing.Scaler import StandardScaler
from LUMA.Preprocessing.Scaler import MinMaxScaler

from LUMA.Reduction.Linear import PCA, LDA
from LUMA.Reduction.Nonlinear import KernelPCA
from LUMA.Reduction.Manifold import TSNE, MDS
from LUMA.Reduction.Manifold import LLE, ModifiedLLE, HessianLLE

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

from LUMA.Classifier.NaiveBayes import GaussianNaiveBayes
from LUMA.Classifier.NaiveBayes import BernoulliNaiveBayes
from LUMA.Classifier.Logistic import LogisticRegressor
from LUMA.Classifier.Logistic import SoftmaxRegressor

from LUMA.Clustering.KMeans import KMeansClustering, KMeansClusteringPlus, KMediansClustering

from LUMA.Metric.Classification import Accuracy, Precision, Recall, F1Score
from LUMA.Metric.Classification import Specificity, AUCCurveROC, Complex
from LUMA.Metric.Regression import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from LUMA.Metric.Regression import MeanAbsolutePercentageError, RSquaredScore, Complex

from LUMA.Visual.Plotter import DecisionRegion, ClusteredRegion
# ---------------------------------------[ Build End ]------------------------------------------- #


def init():
    
    # LUMA.Core.Main
    LUMA
    
    # LUMA.Interface.Super
    _Estimator, _Transformer, _Evaluator, _Visualizer
    _Supervised, _Unsupervised
    
    # LUMA.Preprocessing.Scaler
    StandardScaler, MinMaxScaler
    
    # LUMA.Reduction
    PCA, LDA
    KernelPCA
    TSNE, MDS
    LLE, ModifiedLLE, HessianLLE
    
    # LUMA.ModelSelection
    TrainTestSplit
    GridSearchCV
    
    # LUMA.Regression
    RidgeRegressor, LassoRegressor, ElasticNetRegressor
    PolynomialRegressor
    PoissonRegressor, NegativeBinomialRegressor, GammaRegressor, BetaRegressor, InverseGaussianRegressor
    
    # LUMA.Classifier
    GaussianNaiveBayes, BernoulliNaiveBayes
    LogisticRegressor, SoftmaxRegressor
    
    # LUMA.Clustering
    KMeansClustering, KMeansClusteringPlus, KMediansClustering
    
    # LUMA.Metric
    Accuracy, Precision, Recall, F1Score, Specificity, AUCCurveROC, Complex
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, RootMeanSquaredError, RSquaredScore, Complex
    
    # LUMA.Visual
    DecisionRegion, ClusteredRegion


init()
build_end = time.time()
print(f'Build Succeeded! ({build_end - build_start:.3f} sec)', file=sys.stderr)
