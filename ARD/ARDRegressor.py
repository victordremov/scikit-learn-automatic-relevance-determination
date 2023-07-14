"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class ARDRegressor(BaseEstimator):
    """Automatic relevance determination

    Parameters
    ----------
    n_iterations : int, default = 100
        Number of iterations when converging
    max_alpha: float, default = 100_000
        Maximum value of individual feature regularization parameter.
        When alpha[i] is greater than max_alpha, feature[i] is treated as unrelevant.


    Examples
    --------
    >>> from ARD import ARDRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = ARDRegressor()
    >>> estimator.fit(X, y)
    ARDRegressor()
    """

    def __init__(self, n_iterations: int = 100, max_alpha: float = 100_000):
        self.n_iterations = n_iterations
        self.max_alpha = max_alpha

    def fit(self, X_relevant, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X_relevant : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        assert self.n_iterations > 0
        X_relevant, y = check_X_y(X_relevant, y, accept_sparse=False, dtype=np.float64)
        X_original = np.asarray(X_relevant)
        n, d = X_original.shape

        y = y.astype(np.float64)
        t = y.reshape(n, 1)

        alphas_old = np.ones((d, 1), dtype=np.float64)
        beta_old = 1.0
        for iteration in range(self.n_iterations):
            relevant_features = np.arange(d)[alphas_old.ravel() <= self.max_alpha]
            X_relevant = X_original[:, relevant_features]
            XT_dot_t = X_relevant.T @ t
            XT_dot_X = X_relevant.T @ X_relevant
            # update sigma, w_mp when A, b is fixed
            # as parameters of the posterior distribution
            # p(w | X, T, A, beta) ~ N(w | w_mp, sigma^(-1))
            sigma_relevant = np.linalg.inv(
                beta_old * XT_dot_X + np.diag(alphas_old[relevant_features].reshape(-1))
            )
            w_mp = np.zeros((d, 1), dtype=np.float64)
            w_mp[relevant_features] = beta_old * sigma_relevant @ XT_dot_t

            # update A and beta by maximizing variadic lower estimate for fixed sigma, w_mp
            sigma_relevant_diag = np.diag(sigma_relevant).reshape(-1, 1)
            beta_new = (
                n - (1.0 - alphas_old[relevant_features].T) @ sigma_relevant_diag
            ).item() / np.linalg.norm(t - X_relevant @ w_mp[relevant_features]) ** 2
            alphas_new = alphas_old.copy()
            alphas_new[relevant_features] = (
                1.0 + alphas_old[relevant_features] * sigma_relevant_diag
            ) / w_mp[relevant_features] ** 2
            if not np.isfinite(beta_new):
                # failed to converge
                break
            alphas_old = alphas_new
            beta_old = beta_new

        self.w_ = np.zeros((d, 1), dtype=np.float64)
        self.w_[relevant_features] = w_mp[relevant_features]

        self.sigma_relevant_ = sigma_relevant
        self.alphas_ = alphas_new
        self.beta_ = beta_new
        self.n_features_in_ = d

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        check_is_fitted(self, "is_fitted_")
        return X @ self.w_