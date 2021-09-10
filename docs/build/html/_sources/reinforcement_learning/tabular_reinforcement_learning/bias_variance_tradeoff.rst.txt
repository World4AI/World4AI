======================
Bias-Variance Tradeoff
======================

Motivation
==========

The bias-variance tradeoff plays a tremendous role in reinforcement learning and machine learning in general. Ideally we want to reduce both as much as possible, but reducing one means increasing the other at the same time. There is a tradeoff to be made. 

In this section we will look at the definitions of bias and variance and discuss at what side of the spectrum monte carlo and temporal difference methods are.

Bias
====

.. note::

    .. math:: 

        bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta

In statiscics we define the bias of the estimator as the difference between the expected value of the estimator :math:`\mathbb{E}[\hat{\theta}]` and the true parameter :math:`\theta`. The true parameter :math:`\theta` that we are most interested in is the expected value :math:`\mathbb{E}[X]` of some random variable :math:`X`. 

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/bias_variance_tradeoff/random_variable.svg
   :align: center

   Distribution of a random variable X.

In the image above we see the distribution of the random variable :math:`X`. The calculation of the expectation is straightforward when we know the actual probability distribution.

.. math::

    \mathbb{E}[X] = 0.15 * 1 + 0.15 * 2 + 0.3 * 3 + 0.3 * 4 = 2.55

We do not know the true distribution of :math:`X` and can not directly calculate the expected value, therefore we have to use some estimate :math:`\hat{\theta}` as a proxy for the true :math:`\mathbb{E}[X]`. The most straightforward way to estimate the expected value of a random variable is to draw samples from the distribution and to use the individual samples :math:`X` as the estimate :math:`\hat{\theta}`. Is there any bias by using the random samples as an estimate. No, using :math:`X` as an estimate for :math:`\mathbb{E}[X]` intorduces no bias. 

.. math::
    :nowrap:

    \begin{align*}
    & \theta = \mathbb{E}[X] \\
    & \hat{\theta} = X \\
    & bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta = \mathbb{E}[X] - \mathbb{E}[X] = 0
    \end{align*}

If on the other hand we used the number 3 constantly as the estimate of the expected value, then that would definitely introduce a bias. 

.. math::
    :nowrap:

    \begin{align*}
    & \theta = \mathbb{E}[X] \\
    & \hat{\theta} = 3 \\
    & bias(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta = \mathbb{E}[3] - \mathbb{E}[X] = 3 - 2.55 = 0.45
    \end{align*}

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/bias_variance_tradeoff/simple_mdp.svg
   :align: center

   Distribution of a random variable X.

Variance
========

.. note::

    .. math::

        var(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])]