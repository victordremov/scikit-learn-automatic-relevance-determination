.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/ARM/badge/?version=latest
.. _ReadTheDocs: https://ARM.readthedocs.io/en/latest/?badge=latest

Automatic Relevance Determination
============================================================

Bayesian regression techniques can be used to include regularization parameters in the estimation procedure: the regularization parameter is not set in a hard sense but tuned to the data at hand.

This can be done by introducing uninformative priors over the hyper parameters of the model. The $l_2$
 regularization used in Ridge regression and classification is equivalent to finding a maximum a posteriori estimation under a Gaussian prior over the coefficients $w$ with precision $\lambda^{-1}$. Instead of setting $\lambda$ manually, it is possible to treat it as a random variable to be estimated from the data.

To obtain a fully probabilistic model, the output $y$ is assumed to be Gaussian distributed around $Xw$:

$$
    p(y | X, w, \alpha) = N(y | Xw, alpha)
$$

where $alpha$ is again treated as a random variable that is to be estimated from the data.