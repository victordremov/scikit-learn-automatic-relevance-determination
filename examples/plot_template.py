"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`ARD.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from ARD import ARDRegressor

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = ARDRegressor()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
