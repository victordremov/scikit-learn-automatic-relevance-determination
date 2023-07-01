import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from ARD import ARDRegressor

@pytest.fixture
def data():
    np.random.seed(seed=2023)
    n = 1000
    d = 10
    X = np.random.randn(n, d)
    err = np.random.randn(n)
    y = (X[:, 1] + X[:, 3] * 3 + X[:,  5] * 5 + err * 0.1).reshape(n, 1)
    return X, y


def test_ard_regressor(data):
    est = ARDRegressor()
    assert est.n_iterations == 100

    est.fit(*data)
    assert hasattr(est, "is_fitted_")
    for i, alpha in enumerate(est.alphas_):
        if i not in (1, 3, 5):
            assert alpha > 1000
        else:
            assert alpha < 10
    assert_array_equal(True, np.abs(est.w_.reshape(-1) - np.array([0, 1, 0, 3, 0, 5, 0, 0, 0, 0])) < 0.1)

    X = data[0]
    y_pred = est.predict(X)
