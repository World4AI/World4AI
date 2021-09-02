==========
Double DQN
==========

Motivation
==========

.. note:: 
    "We first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation [#]_."

.. figure:: ../../_static/images/reinforcement_learning/modern_value_based_approximation/double_dqn/overestimation_nn.svg
   :align: center

   Neural Network Maximization Bias.

The DQN algorithm suffers from a similar maximization bias that is present in tabular Q-Learning. The output of Q-functions is an estimate that might contain some noise. The noise that produces the highest number will be preferred in a max operation, even if the true action values are equal. The researchers at DeepMind showed that applying double learning to the DQN algorithms improves the performance of the agent for Atari games. In this section we are going to focus on simple environments like Lunar Lander in order to account for slower hardware of some readers. 

Theory
======

In the DQN algorithm the target value is calculated by utilizing the neural network with frozen weights :math:`\theta^-`.

.. math::
    
    target = r + \gamma \max_{a'} Q(s', a', \theta^-)

It is noticeble that the same Q-function :math:`Q(s, a, \theta^-)` is used to select the next action :math:`a'` and to calculate the ation-value. This is consistent with the classical definition of Q-Learning. In double Q-Learning two separate acion value functions are used. One is used for action selection while the other is used for the calculation of the target value. Using the same approach in DQN would not be efficient, as it would require the training of two action value function. However the original DQN algorithm already uses two action value funcitons. This will allow us to disentangle the calculate of the target value and to separate action selection and action value calculation.

.. math::
    :nowrap:
    
    \begin{align*}
    & a' = \max_{a}Q(s', a', \theta) \\
    & target = r + \gamma Q(s', a', \theta^-)
    \end{align*}

The action in the next state is selected by utilizing the action value function :math:`Q(s, a, \theta)`, while the calculation of the action-value is performed using the frozen weights :math:`\theta^-`. 

Implementation
==============

The adjustments to DQN that need to be made to implement double learning are minimal. 

.. code:: python

    # replace the below code
    with torch.no_grad():
        target = reward + self.gamma * self.target_network(next_obs).max(dim=1, keepdim=True)[0] * (1 - done)

    # with the following code
    with torch.no_grad():
        next_action = self.online_network(next_obs).max(dim=1, keepdim=True)[1].long()
        target = reward + self.gamma * self.target_network(next_obs).gather(dim=1, index=next_action) * (1 - done)


Sources
=======

.. [#] van Hasselt H., Guez A., Silver D.. Deep Reinforcement Learning with Double Q-learning. https://arxiv.org/abs/1509.06461