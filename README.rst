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

This can be done by introducing uninformative priors over the hyper parameters of the model. The :math:`l_2` regularization used in Ridge regression and classification is equivalent to finding a maximum a posteriori estimation under a Gaussian prior over the coefficients :math:`w` with precision :math:`\lambda^{-1}`. Instead of setting :math:`\lambda` manually, it is possible to treat it as a random variable to be estimated from the data.

To obtain a fully probabilistic model, the output :math:`y` is assumed to be Gaussian distributed around :math:`Xw`:

.. math::
    p(y | X, w, \alpha) = N(y | Xw, alpha)

where :math:`\alpha` is again treated as a random variable that is to be estimated from the data.

Usage example
-------
Jupyter notebook with usage example: <https://github.com/victordremov/scikit-learn-automatic-relevance-determination/tree/master/examples/hello-ard.ipynb>

Source code
-------
Source code: <https://github.com/victordremov/scikit-learn-automatic-relevance-determination/tree/master/ARD/ARDRegressor.py>