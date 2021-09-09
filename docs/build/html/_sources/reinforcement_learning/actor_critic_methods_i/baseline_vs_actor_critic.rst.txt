========================================
Baseline Methods vs Actor-Critic-Methods
========================================

The general shape of the policy gradient can be represented in the following way [#]_. 

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  \Psi_t] 
    \end{align*}

The letter :math:`\Psi` is a placeholder that in practice can be replaced by a number of variables. In the derivation of the policy gradient algorithm :math:`\Psi` corresponded to the return over the whole trajectory :math:`\sum_t^H R_t`. The shape of :math:`\Psi` is what determines if we are talking about pure policy gradient algorithms or about actor-critic methods.

.. figure:: ../../_static/images/reinforcement_learning/actor_critic_methods_i/baseline_vs_ac/actor_critic.svg
   :align: center

   Actor Critic Methods.

An actor-critic algorithm has a policy, called an actor and a value function, called a critic. But not all agents that have separate policy and value functions are defined as actor-critic methods. For example the REINFORCE algorithm with baseline (VPG) does not have a critic even though the baseline is made of the state value function. The key component that is missing is bootstrapping. In VPG the agent takes the action and the action is evaluated by calculating the return that follows and subtucting the expecgted sum of returns :math:`V(S_t)`. The next state :math:`S_{t+1}` that results from the action is not evaluated by the value function, the action is not critiqued directly by the value function. Bootstrapping on the other hand critiques the action by evaluating the outcome of the action through evaluating the next state :math:`V(S_{t+1})`.   

There are several common choices for the calculation of :math:`\Psi` in actor-critic methods.

* :math:`\Psi = \delta = R_t + V(S_{t+1}) - V(S_t)`
* :math:`\Psi = A(S_t, A_t) = Q(S_t, A_t) - V(S_t)`

Both variants are commonly used, but the most powerful and flexible choice for :math:`\Psi` that is used in state of the art systems is going to be covered in a separate section.

.. [#] Schulman J. et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. 2015. https://arxiv.org/abs/1506.02438
