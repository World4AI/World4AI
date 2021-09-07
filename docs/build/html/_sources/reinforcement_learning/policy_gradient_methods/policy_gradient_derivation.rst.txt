==========================
Policy Gradient Derivation
==========================

.. important::

    I have seen several different methods to derive and explain the policy gradient. The first is described in the book by Richard Sutton and Andrew Barto [#]_. The second is covered for example by Pieter Abbeel in his Deep RL Bootcamp lecture on “Policy Gradients and Actor Critic” [#]_. I have always found the second type of derivation more intuitive and clear and therefore I am going to follow the same approach, but some notation and ideas are going to be taken from Sutton’s and Barto’s book. In any case I highly recommend both sources.


Motivation
==========

Before we move to the derivation of the policy gradient, let us discuss why it might be a good idea to use policy gradient methods. Below is a list of points that are often mentioned in the reinforcement learning literature and online lectures.

* Sometimes it is easier to estimate the policy directly, instead of estimating the action value function.

* Q-Learning does not easily allow the use of continuous action spaces, because of the max operation to determine the best action. In policy gradient methods we can sample an action from a continuous distribution.

* It is easy to implement a stochastic policy with policy gradient methods. This avoids the need for an additional exploration strategy, as we can randomly sample from the distribution. Through learning better actions are going to be assigned higher probabilities while bad actions will be unlikely.

* Policy gradient methods have better convergence properties. When in Q-Learning the action which constitutes the greedy action changes due to gradient descent, the change in the shape of the value function is abrupt and might destabilize training. In policy gradient methods the change of the probability distribution of actions is relatively smooth.

* Policy gradient methods have high variance, but improvements can be made to decrease the variance.

Derivation
==========

Let us remember the general interaction cycle between the agent and the environment.

.. figure:: ../../_static/images/reinforcement_learning/policy_gradient_methods/policy_gradient_derivation/interaction_with_policy.svg
   :align: center

   MDP Interaction Cycle.

In policy gradient methods in order to interact with the environment the agent utilizes a parametrized policy :math:`\pi_{\theta}(a|s)`, instead of a value function. The policy provides a distribution of actions that is based on the current state and the learnable parameters :math:`\theta`. 

.. figure:: ../../_static/images/reinforcement_learning/policy_gradient_methods/policy_gradient_derivation/architecture.svg
   :align: center

   Policy Architecture.

In our case the policy is going to be a neural network with weights :math:`\theta`. The neural network will generate a probability distribution that will be sampled to generate an action, :math:`a \sim \pi_{\theta}(. \mid s)`.

The interaction generates trajectories :math:`\tau`, a sequence of tuples consisting of states, actions and rewards :math:`(s_t, a_t, r_t, s_{t+1}, a_{t+1}, r_{t+1}, ... , s_T, a_T, r_T)`. Each of the trajectories has a corresponding return :math:`G`. Sometimes, especially when talking about policy gradients, the return is also defined as :math:`R(\tau)` to indicate that the return is based on the trajectory that was followed.

The general goal of a reinforcement learning agent is to maximize the expected sum of rewards. In policy gradient methods the expected return is also called the *objective function*.

.. math:: 

	J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] = \sum_{\tau}\mathbb{P}(\tau \mid \theta) R(\tau)

The expectation is defined over the trajectories :math:`\tau` that are sampled using a policy :math:`\pi` with parameters :math:`\theta`. Each return that results from the trajectory :math:`\tau` is weighted with the corresponding probability. For us this means that we have to find parameters :math:`\tau` that generate trajectories with the highest expected returns.

.. math::
    \arg\max_{\theta}J(\theta)

To find the parameters that maximize the objective function we are going to use gradient ascent.

.. math:: 

    \theta \leftarrow \theta + \alpha \nabla_{\theta}J(\pi_{\theta})

The objective function :math:`J(\pi_{\theta})` is unknown, because the calculation of the expectation over trajectories would require the knowledge of the dynamics of the model. Therefore it is not that simple to calculate the gradient of the objective function. We need to restate the problem from a different perspective in order to calculate the gradient :math:`\nabla_{\theta}`.

.. note:: 
    
    The likelihood ratio trick utilizes the following identity.

    .. math::

        \nabla_x \log f(\mathbf{x}) = \nabla_x f(\mathbf{x}) \frac{1}{f(\mathbf{x})}
    
    Here we use the chain rule and the derivative of the log function to calculate the derivative.


.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)] \\ 
    & = \nabla_{\theta} \sum_{\tau}\mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau}\nabla_{\theta} \mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau} \frac{\mathbb{P}(\tau \mid \theta)}{\mathbb{P}(\tau \mid \theta)} \nabla_{\theta} \mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \sum_{\tau} \mathbb{P}(\tau \mid \theta) \frac{\nabla_{\theta} \mathbb{P}(\tau \mid \theta)}{\mathbb{P}(\tau \mid \theta)} R(\tau) \\
    & = \sum_{\tau} \mathbb{P}(\tau \mid \theta) \nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau) \\
    & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau)]
    \end{align*}

In the above reformulation we used a couple of mathematical tricks. 

* First, from basic calculus we know that the derivative of a sum is a sum of derivatives. That allows us to bring in the derivative sign inside. We will talk about the huge importance of that step down below. 

* Second, multiplying and dividing by the same number does not change the derivative calculation, because both operations cancel each other. We multiply and divide by the probability of trajectory. Combining the sum over trajectories and the weighting with the probabilities of trajectories gives us an expectation over trajectories.

* Third, we use the likelihood ratio trick to rewrite part of the derivative as a log expression. The log has some nice properties that we are going to apply in a later step.

At this point in time we still do not know the derivative of :math:`\mathbb{P}(\tau \mid \theta)` with respect to :math:`\theta`, because we do not know the exact model of the model of the MDP. We reformulate the probability :math:`\mathbb{P}(\tau \mid \theta)`.

.. math::

    \mathbb{P}(\tau \mid \theta) = \prod_t^H P(S_{t+1} \mid S_t, A_t) \pi_{\theta}(A_t \mid S_t)

The probability of a trajectory depends on one side on the policy of the agent :math:`\pi_{\theta}`, which determines the probability of the action :math:`a_t` based on the current state :math:`s_t`. On the other hand the model calculates the probability of the next state :math:`s_{t+1}` based on the action taken :math:`a_t` and the current state :math:`s_t`. The selection of actions and next states continues until the end of the episode, which is indicated by the horizon :math:`H`. The calculation of the probability of the full trajectory is the product of individual probabilities that are calculated throughout the trajectory. 

.. math::
    :nowrap:

    \begin{align*}
    \nabla_{\theta} \log \mathbb{P}(\tau \mid \theta) & = \nabla_{\theta} \log (\prod_t^H P(S_{t+1} \mid S_t, A_t) \pi_{\theta}(A_t \mid S_t)) \\
    & = \nabla_{\theta} (\sum_t^H \log P(S_{t+1} \mid S_t, A_t) + \sum_t^H \log \pi_{\theta}(A_t \mid S_t)) \\
    & = (\sum_t^H \nabla_{\theta} \log P(S_{t+1} \mid S_t, A_t) + \sum_t^H \nabla_{\theta}  \log \pi_{\theta}(A_t \mid S_t)) \\
    & = \sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) \\
    \end{align*}

It turns out that the gradient of the reformulated problem is an easier problem.

* First, we realize that the log of a product is the sum of the logs, :math:`\log(x*y) = \log x + \log y`. This makes obvious why the reformulation of the problem in terms of logs was a necessary step. This allows us to separate the policy from the model in a powerful way.

* Finally we realize that :math:`\nabla_{\theta} \log P(S_{t+1} \mid S_t, A_t)` is 0. The derivative is with respect to :math:`\theta`, which is the parameter vector of the policy and the policy has no impact on the model. No matter how the policy looks like, the agent can not change the underlying dynamics of the MDP. 


.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\nabla_{\theta} \log\mathbb{P}(\tau \mid \theta) R(\tau)] \\
    & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)]
    \end{align*}

The final gradient depends only on the gradient of the policy :math:`\pi` and the realized return, the knowledge of the dynamics of the model are not required.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) R(\tau^{(i)})
    \end{align*}

Let us also discuss why it was important to push the gradient inside the expectation. The gradient of the expectation :math:`\nabla_{\theta}\mathbb{E}` implies that we have to know the expected value of returns to calculate the gradient, which we don’t. When the expectation is inside we can sample trajectories and estimate the true gradient. The larger the sample size the better the estimate. In practice often the gradient step is taken after a single episode, indicating :math:`m = 1`.

We are not going to implement this naive policy gradient algorithm, as there is high variance due to high noise of returns of individual episodes. Starting with the next chapter we will investigate methods to decrease the variance and implement the algorithm in PyTorch.

Sources
=======

.. [#] Sutton, R. Barto, A. Reinforcement Learning: An Introduction (MIT Press, 2018).
.. [#] Abbeel P. et al. Deep RL Bootcamp. 2017. https://sites.google.com/view/deep-rl-bootcamp