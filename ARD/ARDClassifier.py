import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

class ARDClassifier(LinearClassifierMixin, BaseEstimator):
    def __init__(
        self,
        max_iterations: int = 1000,
        alpha_bound: float = 1e6,
        weight_bound=1e-6,
        max_w_mp_diff=1e-4,
        tol: float = 1e-4,
        max_newton_cg_steps: int = 100,
    ):
        self.max_iterations = max_iterations
        self.alpha_bound = alpha_bound
        self.max_w_mp_diff = max_w_mp_diff
        self.weight_bound = weight_bound
        self.tol = tol
        self.max_newton_cg_steps = max_newton_cg_steps

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        X_original = X

        self.classes_ = unique_labels(y)
        if self.classes_.size > 2:
            raise ValueError("Multi-classification not supported")
        elif self.classes_.size < 2:
            raise ValueError("Labels contain only one class")

        n, d = X.shape
        t = np.where(y == self.classes_[0], -1.0, 1.0).astype(np.float64).reshape(-1, 1)

        w_mp = np.zeros((d, 1), dtype=np.float64)
        alphas = np.ones((d, 1), dtype=np.float64)

        for iteration in range(self.max_iterations):
            w_mp_old = w_mp.copy()
            informative_features = np.arange(d)[alphas.reshape(-1) < self.alpha_bound]
            m = len(informative_features)
            X = X_original[:, informative_features]
            A = np.diag(alphas[informative_features].ravel())
            w_informative = w_mp[informative_features]

            for i in range(self.max_newton_cg_steps):
                w_informative_new, sigma = self._update_w(w_informative, X, A, t)
                diff = np.abs(w_informative_new - w_informative).max()
                w_informative = w_informative_new
                if diff < self.max_w_mp_diff:
                    break
            w_mp = w_mp.copy()
            w_mp[informative_features] = w_informative

            if np.abs(w_mp - w_mp_old).max() < self.max_w_mp_diff:
                break
            alphas_new = alphas.copy()
            alphas_new[informative_features] = (
                1.0 - np.diag(A).reshape(m, 1) * np.diag(sigma).reshape(m, 1)
            ) / (w_informative ** 2)
            is_feature_informative = (alphas_new < self.alpha_bound) & (
                np.abs(w_mp) > self.weight_bound
            )
            w_mp[~is_feature_informative] = 0.0
            alphas_new[~is_feature_informative] = np.inf
            alphas = alphas_new

        self.n_features_in_ = d
        self.coef_ = w_mp.reshape(1, -1)
        self.intercept_ = 0.0
        self.is_fitted_ = True
        self.informative_features_ = informative_features
        return self

    @staticmethod
    def _score(X, t, w, alphas):
        """Return not normalized logarithm of probability
        P(X, t, w | A) = p(T | X, w, A) * p(w | A).
        """
        score = np.log(expit(t * (X @ w))).sum(axis=0) - 0.5 * (alphas * w ** 2).sum(
            axis=0
        )
        return score

    @staticmethod
    def _update_w(w, X, A, t):
        s = expit(t * (X @ w))
        sigma = np.linalg.inv(X.T @ np.diag((s * (1 - s)).ravel()) @ X + A)
        grad = (X * (1 - s) * t).sum(axis=0).reshape(-1, 1) - A @ w
        w_new = w + sigma @ grad
        return w_new, sigma
