"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class ARDRegressor(BaseEstimator):
    """ Automatic relevance determination

    Parameters
    ----------
    n_iterations : int, default = 10
        Number of iterations when converging

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
    def __init__(self, n_iterations: int = 100):
        self.n_iterations = n_iterations

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
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
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        y = y.astype(np.float64)
        n, d = X.shape
        t = y.reshape(n, 1)

        alphas_old = np.ones((d, 1), dtype=np.float64)
        beta_old = 1.

        XT_dot_t = X.T @ t
        XT_dot_X = X.T @ X

        for iteration in range(self.n_iterations):
            # update sigma, w_mp when A, b is fixed
            # as parameters of the posterior distribution
            # p(w | X, T, A, beta) ~ N(w | w_mp, sigma^(-1))
            sigma = np.linalg.inv(beta_old * XT_dot_X + np.diag(alphas_old.reshape(-1)))
            w_mp = beta_old * sigma @ XT_dot_t

            # update A and beta by maximizing variadic lower estimate for fixed sigma, w_mp
            sigma_diag = np.diag(sigma).reshape(d, 1)
            beta_new = (n - (1.0 - alphas_old.T) @ sigma_diag).item() / np.linalg.norm(t - X @ w_mp) ** 2
            alphas_new = (1.0 + alphas_old * sigma_diag) / w_mp ** 2
            if not np.isfinite(beta_new) or not np.isfinite(alphas_new).all():
                break

            alphas_old = alphas_new
            beta_old = beta_new

        self.w_ = w_mp
        self.sigma_ = sigma
        self.alphas_ = alphas_old
        self.beta_ = beta_old
        self.n_features_in_ = d

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

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
        check_is_fitted(self, 'is_fitted_')
        return X @ self.w_


class TemplateClassifier(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class TemplateTransformer(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return np.sqrt(X)
