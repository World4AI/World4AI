==================================
Value Approximation With An Oracle
==================================

State Representation
====================

So far for finite MDPs each state was represented by a single number. This number was used as an address in the state-value or action-value lookup table.

.. list-table:: Value Function for finite MDP
    :widths: 25 25
    :header-rows: 1

    * - State
      - Value
    * - 0
      - 1
    * - 1
      - 2
    * - 2
      - 1.5
    * - **3**
      - 3

In the example above to get the the state-value for state 3 for a certain policy :math:`\pi` the agent looked at the value in the lookup table to receive the value of 3.

For complex MDPs that approach is not sustainable, as for most interesting problems the number of states is larger than the number of atoms in the observable universe. Therefore a state is represented by a so-called feature vector. Each number in the vector gives some information about the state. The whole vector is the representation of the state. In many cases the representation is only partial, therefore in approximative methods we are going to use the word *observation* instead of *state* to show the possible limitations of state representations. 

.. math:: 

    \mathbf{x} \doteq (x_1(s), x_2(s), ... , x_d(s))^T

In the Cart Pole environment the feature vector consists of cart position, cart velocity, pole angle and angular velocity.

.. math:: 

    \mathbf{x} \doteq (CartPosition, CartVelocity, PoleAngle, AngularVelocity)^T

Value Representation
====================

The feature vector :math:`\mathbf{x}` is used as an input into the approximative value function and the output is a single state-value or action-value number. 

.. figure:: ../../_static/images/reinforcement_learning/approximative_rl/value_approximation_oracle/value_approximation.svg
   :align: center

   Approximation of a Value Function.

In order for the value function to transform the feature vector into the single number representation an additional vector, called weight vector, :math:`\mathbf{w} \in \mathbb{R}^n` is needed. How exactly the weights are used in the calculation depends on the type of the function.

.. figure:: ../../_static/images/reinforcement_learning/approximative_rl/value_approximation_oracle/value_approximation_weights.svg
   :align: center

   Weights of a Value Function.

There are many different types of function approximators:

* Linear Function Approximators
* Neural Networks (Non-Linear Function Approximators)
* Decision Trees
* ...

Depending on the function approximators the weight vector might play a different role in the calculation of the value function, but the general way to write down function approximators is as follows.

.. math::

    \hat{v}(s, \mathbf{w})

.. math:: 

    \hat{q}(s, a, \mathbf{w})

Where the "^" above the function (read as hat) shows that the function is an approximation and the weight vector :math:`\mathbf{w}` shows that the calculation of state or action values requires that vector.

At the moment of writing most modern reinforcement learning function approximators are neural networks. Linear function approximators are especially useful to introduce the topic of function approximators, as those are easiest to grasp and show some useful mathematical properties. 

In linear function approximators each of the features is “weighted” by the corresponding weight. The individual weighted features are summed up to produce the value. 

.. math:: 

    \hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^T\mathbf{x}(s) \doteq \sum_{i=1}^d w_i x_i(s)


Let us again look at the Cart Pole environment to clarify the linear function approximation. Below is one of the possible initial values for the feature vector. 

:math:`\mathbf{x}` = [0.04371849, -0.04789172, -0.03998533, -0.01820894]
     
In order to calculate the approximate state value for a particular :math:`\pi` for the above state the following equation has to be calculated.

.. math:: 

    \hat{v}(s, \mathbf{w}) = w_1 * 0.04371849 + w_2 * (-0.04789172) + w_3 * (-0.03998533) + w_4 * (-0.01820894)

The same four weights are used for the calculation of the state value for all possible feature vectors.

Neural networks are non-linear function approximators, where each neuron in itself is a non-linear function. The calculation for each neuron is similar to that of the linear function, but the result of the weighted sum is used as an input to a non-linear function :math:`f()`.

.. math:: 

    \hat{v}(s, \mathbf{w}) \doteq f(\mathbf{w}^T\mathbf{x}(s)) \doteq f(\sum_{i=1}^d w_i x_i(s))

General Policy Iteration
========================

Similar to dynamic programming the general idea when using approximative functions is to switch between policy evaluation and policy improvement.

In the policy evaluation step we are going to look for a function :math:`\hat{v}` that is as close as possible to the true value function :math:`v_{\pi}`.

.. math::

    \hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)

In the policy improvement step we are going to utilize :math:`\hat{q}` in order to act greedily and improve our policy. 

.. math:: 

    \hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)

Policy Evaluation
-----------------

Let us as always assume that we have some policy pi and are interested in the true value function of that particular policy. Finding the true value function is out of the question, so we have to deal with an approximation.

.. math::

    \hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)

Generally it might be sufficient for us to find an approximative value function that is just good enough. In this chapter we are going to discuss what constitutes a “good” approximation and how we can find the weight vector :math:`\mathbf{w}` for that “good” approximation . 

To build the theory that is going to be used throughout the rest of the book it is convenient to start the discussion by assuming that we are in a supervised learning setting and that there is an oracle who tells us what the true state-value :math:`v_{\pi}(s)` for the given policy :math:`\pi` and state :math:`s` is. Later the discussion can be extended to reinforcement learning settings where the agent interacts with the environment. 

In supervised learning the goal is to find a weight vector w that produces a function that fits the training data as close as possible. That means that we want weights that reduce the difference between the true state-value and our approximation as much as possible. In reinforcement learning Mean Squared Error (MSE) is used to define the difference between the true value function and the approximate value function.

.. math::

    MSE \doteq \mathbb{E_{\pi}}[(v_{\pi} - \hat{v}(s, \mathbf{w}))^2]

If we find the weight vector :math:`\mathbf{w}` that minimizes the above expression, then we found an approximation that is as close as possible to the true value function given by the oracle.

The common approach to find such a vector is to use stochastic gradient descent. Stochastic gradient descent in a setting with an oracle would work as follows. The agent interacts with the environment using the policy :math:`\pi`. For each of the observations the agent calculates the approximate value and compares the difference between the approximate value and the true value given by the oracle using the mean squared error. In the next step the agent calculates the gradients of MSE with respect to the weights of the value function. Using the gradient the agent reduces the MSE by adjusting the weight vector :math:`\mathbf{w}`. **Stochastic** gradient descent means that the update of the weights is done at each single step.

The update rule for the weight vector is as follows. 

.. math:: 
    :nowrap:

    \begin {align*}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[v_{\pi}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
    & = w_t + \alpha[v_{\pi}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
    \end {align*}



The gradient :math:`\nabla\hat{v}(S_t, \mathbf{w}_t)` is a vector that contains partial derivations of the approximative value function with respect to individual weights. We reduce the weights into the direction of the gradient.

.. math::

    \nabla \hat{v}(s, \mathbf{w}) \doteq (\frac{\partial f(\mathbf{w})}{\partial w_1}, \frac{\partial f(\mathbf{w})}{\partial w_2}, ... , \frac{\partial f(\mathbf{w})}{\partial w_d})^T


Linear functions and neural networks are differentiable, decision Trees are not differentiable functions. That means that for linear functions (and neural networks) it is easy to determine how to adjust the weight vector :math:`\mathbf{w}`. From now on we are primarily going to focus on neural networks. To discuss some of the theoretical properties we will return to linear methods during the next few chapters.

Policy Improvement
------------------

Policy improvement with function approximators utilizes the action-value function instead of a state-value function. 

.. math:: 

    \hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)

Once again we assume to have an oracle that provides the true action-value for a policy :math:`\pi`, given the state and the action. At each time step the agent selects an action using :math:`\epsilon`-greedy. Using the information from the oracle and the approximate estimation, the agent adjusts the weights of the function to get as close as possible to the true action-value function. 

.. math:: 
    :nowrap:

    \begin {align*}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]^2 \\
    & = w_t + \alpha[q_{\pi}(S_t, A_t) - \hat{q}(S_t, A_t, \mathbf{w}_t)]\nabla\hat{q}(S_t, A_t, \mathbf{w}_t)
    \end {align*}

Limitations
-----------

Pure value methods that use action-value functions to determine the policy are still limited to discrete actions spaces. To determine the action the agent needs to take the max over available options and that gets problematic with continuous action spaces. In case of a high number of possible actions it might take a long time to calculate the max and the performance would suffer. For now it is sufficient to know that there are other approximation methods that can deal with these sorts of problems. 
