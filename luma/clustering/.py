import numpy as np


class _GaussianMixture:
    def __init__(self, n_clusters=1, max_iter=100, tol=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None
        self.converged = False
        self.labels_ = None

    def initialize_parameters(self, X):
        np.random.seed(0)
        n_samples, n_features = X.shape
        random_idx = np.random.permutation(n_samples)[:self.n_clusters]
        self.means = X[random_idx]
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_clusters)])
        self.weights = np.full(self.n_clusters, 1 / self.n_clusters)
        
    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            cov_inv = np.linalg.inv(self.covariances[k])
            diff = X - self.means[k]
            exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
            coeff = 1 / np.sqrt(np.linalg.det(self.covariances[k]) * (2 * np.pi)**X.shape[1])
            responsibilities[:, k] = self.weights[k] * coeff * exp_term
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        self.weights = responsibilities.mean(axis=0)
        self.means = responsibilities.T @ X / np.sum(responsibilities, axis=0)[:, np.newaxis]
        for k in range(self.n_clusters):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k, np.newaxis] * diff).T @ diff / responsibilities[:, k].sum()

    def fit(self, X):
        self.initialize_parameters(X)
        for _ in range(self.max_iter):
            responsibilities = self.e_step(X)
            means_old = self.means.copy()
            self.m_step(X, responsibilities)
            if np.linalg.norm(self.means - means_old) < self.tol:
                self.converged = True
                break
        self.labels = np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        return self.e_step(X)

