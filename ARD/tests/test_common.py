import pytest

from sklearn.utils.estimator_checks import check_estimator

from ARD import ARDRegressor

@pytest.mark.parametrize(
    "estimator",
    [ARDRegressor()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
