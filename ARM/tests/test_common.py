import pytest

from sklearn.utils.estimator_checks import check_estimator

from ARM import TemplateEstimator
from ARM import TemplateClassifier
from ARM import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)