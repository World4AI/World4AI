==================================
Value Approximation With An Oracle
==================================

Just as in the tabular case of general policy iteration we need to run prediction and improvement sequentially. 

For prediction it is sufficient to approximate the state-value function.

.. math::

    \hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)

For improvement we will need to approximate the action-value function.

.. math:: 

    \hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)


:math:`\hat{v}(s, \mathbf{w})` and :math:`\hat{q}(s, a, \mathbf{w})` are both parameterized functions where :math:`\mathbf{w} \in \mathbb{R}^d` is a weight vector.  That essentially means that in order for the function to calculate the value of the state or state-action pair a vector :math:`\mathbf{w}` (the parameters of the function) is needed for the calculation. Better policy evaluation and policy improvement involve adjusting this weight vector. The novelty is also that the same vector is used for all states (or state-action pairs). That means that adjusting the weights to make better predictions for a certain state affects the predictions of all other states. Finding a balance for different states is going to be of great concern to us. Tweaking the weights for seen states we would like our solution to be general for many other, even unforeseen states. 

The state representation is also different from the tabular case. In the tabular case each state was represented by a single number, which was used as an address in the lookup table. 

With function approximators that approach is not sustainable, as for most interesting problems the number of states is larger than the number of atoms in the observable universe. For that purpose the state is represented by a so-called feature vector. Each number in the vector gives some information about the state. The whole vector is the available representation of the state. 

.. math:: 

    \mathbf{x} \doteq (x_1(s), x_2(s), ... , x_d(s))^T


In many cases the representation is only partial, therefore in approximative methods we are going to use the word observation instead of state to show the possible limitations of state representations. 

Pure value methods that use action-value functions to determine the policy are still limited to discrete actions spaces. To determine the action the agent needs to take the max over available options and that gets problematic with continuous action spaces. In case of a high number of possible actions it might take a long time to calculate the max and the performance would suffer. For now it is sufficient to know that there are other approximation methods that can deal with these sorts of problems. 


There are many different types of function approximators:

* Linear Function Approximators
* Neural Networks (Non-Linear Function Approximators)
* Decision Trees
* ...


Depending on the function approximators the weight vector might play a different role in the calculation of the value function. At the moment of writing most modern reinforcement learning function approximators are neural networks. Linear function approximators are especially useful to introduce the topic of function approximators, as those are easiest to grasp and show some useful mathematical properties. Linear functions and neural networks are differentiable, decision Trees are not differentiable functions. That means that for linear functions (and neural networks) it is easy to determine how to adjust the weight vector w so as to reduce some value of a function.

In linear function approximators each of the features is “weighted” by the corresponding weight. The individual weighted features are summed up to produce the value. 

.. math:: 

    \hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^T\mathbf{x}(s) \doteq \sum_{i=1}^d w_i x_i(s)

Neural networks are non-linear function approximators, where each neuron in itself is a non-linear function. The calculation for each neuron is similar to that of the linear function, but the result of the weighted sum is used as an input to a non-linear function :math:`f()`.

.. math:: 

    \hat{v}(s, \mathbf{w}) \doteq f(\mathbf{w}^T\mathbf{x}(s)) \doteq f(\sum_{i=1}^d w_i x_i(s))


To build the theory that is going to be used throughout the rest of the book it is convenient to start the discussion by assuming that we are in a supervised learning setting and that there is an oracle who tells us what the true state-value :math:`v_{\pi}(s)` for the given policy :math:`\pi` and state :math:`s` is. Later the discussion can be extended to more complex reinforcement learning settings. 

In supervised learning the goal is to find a weight vector w that produces a function that fits the training data as close as possible. That means that we want weights that reduce the difference between the true state-value and our approximation as much as possible. In reinforcement learning Mean Squared Error (MSE) is used to define the difference between the true value function and the approximate value function.

.. math::

    MSE \doteq \mathbb{E_{\pi}}[(v_{\pi} - \hat{v}(s, \mathbf{w}))^2]

Stochastic gradient descent in a setting with an oracle would work as follows. Let us assume we interact with the environment using the policy pi. For each of the observations the oracle tells us the true value of the state v_pi(s), which we can use in stochastic gradient descent to update the weight vector w. **Stochastic** gradient descent means that we use gradient descent at each single observation and do not wait to collect these into a batch. 

The update rule is as follows. 

.. math:: 
    :nowrap:

    \begin {align*}
    w_{t+1} & \doteq w_t - \frac{1}{2}\alpha\nabla[v_{\pi}(S_t) - \hat{v}(S_t,\mathbf{w}_t)]^2 \\
    & = w_t + \alpha[v_{\pi}(S_t) - \hat{v}(S_t, \mathbf{w}_t)]\nabla\hat{v}(S_t, \mathbf{w}_t)
    \end {align*}



The gradient :math:`\nabla\hat{v}(S_t, \mathbf{w}_t)` is a vector that contains partial derivations of the approximative value function with respect to individual weights. We reduce the weights into the direction of the gradient.

.. math::

    \nabla \hat{v}(s, \mathbf{w}) \doteq (\frac{\partial f(\mathbf{w})}{\partial w_1}, \frac{\partial f(\mathbf{w})}{\partial w_2}, ... , \frac{\partial f(\mathbf{w})}{\partial w_d})^T