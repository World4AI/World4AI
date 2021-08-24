==========================
Convergence And Optimality
==========================

Problem Definition
==================

The algorithms that we discussed during the last chapters attempt to find weights that create an approximate function that is as close as possible to the true state or action value function. The measurement of closeness that is used throughout reinforcement learning is the mean squared error (MSE). But in what way does finding the weights that produce the minimal mean squared error contribute to a value function that is close to the optimal function? Therefore before we proceed to the next chapter there are several questions we have to ask ourselves. 

* Can we find the optimal state/action value function?
* What does convergence mean?
* Do the algorithms have convergence guarantees? 
* Towards what value does convergence happen?


Convergence
===========

When we talk about convergence we usually mean that as time moves along and the agent adjusts the weight vector through gradient descent the mean squared error converges towards some value. That does not necessarily mean that the agent finds a weight vector that generates the smallest possible MSE, as gradient descent might get stuck in a local minimum. 

.. list-table:: Prediction Convergence
    :widths: 50 25 25 25
    :header-rows: 1

    * - Algorithm
      - Tabular
      - Linear
      - Non-Linear
    * - *Monte Carlo*
      - Converges
      - Converges
      - Converges
    * - *Sarsa*
      - Converges
      - Converges
      - No Convergence
    * - *Q-Learning*
      - Converges
      - No Convergence
      - No Convergence
  
.. list-table:: Control Convergence
    :widths: 50 25 25 25
    :header-rows: 1

    * - Algorithm
      - Tabular
      - Linear
      - Non-Linear
    * - *Monte Carlo*
      - Converges 
      - Oscilates
      - No Convergence
    * - *Sarsa*
      - Converges
      - Oscilates
      - No Convergence
    * - *Q-Learning*
      - Converges
      - No Convergence
      - No Convergence


Optimality
==========

Conclusion
==========