======================================
Generalized Advantage Estimation (GAE)
======================================

Motivation
==========

.. note::

    "We address the first challenge (the large number of samples typically required) by using value functions to substantially reduce the variance of policy gradient estimates at the cost of some bias, with an exponentially-weighted estimator of the advantage function that is analogous to TD(λ) [#]_".

Let us return to the general definition of the policy gradient.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  \Psi_t] 
    \end{align*}

The greek letter psi :math:`\Psi` can be replaced by a variety of options, but in modern reinforcement learning algorithms it is most likely to contain an advantage estimation :math:`A(S_t, A_t)`, making it an advantage actor-critic algorithm.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  A(S_t, A_t)] 
    \end{align*}


Additionally we can define the number of steps :math:`n`, where :math:`n` defines how many returns have to be unrolled following a policy :math:`\pi_{\theta}` before an optimization step is taken. When we want to distinguish the advantage function based on the number of steps, we define the advantage function as :math:`A^{(n)}_t(S_t, A_t)`.


.. math:: 
    :nowrap:

    \begin{align*}
	& \hat{A}^{(1)}_t(S_t, A_t) = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
	& \hat{A}^{(2)}_t(S_t, A_t) = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) - V(S_t) \\
    & ... \\
    & ... \\
    & \hat{A}^{(n)}_t(S_t, A_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t}R_{t'+1} + \gamma^{t+n}V(S_{t+n}) - V(S_t) \\
    \end{align*}

The higher the number n the higher the variance and the lower the bias. The sweetspot is usually not at the extreme ends, where n is either 1 and we end up with a one step temporal difference advantage estimation or n is unbounded and we end up with a full monte carlo estimation. In the A3C algorithm n corresponded to 5, but it is not clear if it was the right choice. 

The generalized advantage estimation allows us to utilize many advantage functions :math:`A^{(n)}_t(S_t, A_t)` with different :math:`n`, to hopefully end up with a more robust advantage estimation.

Theory
======

To come up with a better advantage estimator, each of the :math:`n` estimators is weighted and summed up. 

.. math::
    \hat{A}^{GAE}_t = \sum_{n=1}^\infty w_n \hat{A}_t^{(n)}

To control the strenghts of the weight decay, the authors utilize :math:`\lambda` and create an exponentially-weighted estimator. 

.. math::
    \hat{A}^{GAE}_t = (1 - \lambda)(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + ...)

.. note:: 

    This estimator can be unpacked and reduced to the following form.

    .. math::

        \hat{A}^{GAE}_t = \sum^\infty_{l=0}(\gamma\lambda)^l\delta_{t+l}

    Where :math:`\delta_t` is the temporal difference error at the timestep :math:`t`.

    .. math::
        :nowrap:

        \begin{align*}
        \delta_t & = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
        \delta_{t+1} & = R_{t+2} + \gamma V(S_{t+2}) - V(S_{t+1})
        \end{align*}

    In the implementation below we are going to utilize this reduced form.


Implementation
==============

.. code:: python

    def optimize(self):        
        len_trajectory = len(self.rewards)
        gammas = np.array([self.gamma**exp for exp in range(len_trajectory)])
        taus = np.array([self.tau**exp for exp in range(len_trajectory)])
        
        rewards = np.vstack(self.rewards)
        values = torch.vstack(self.values).detach().cpu().numpy()
        log_probs = torch.vstack(self.log_probs).to(self.device)
        
        # OPTIMIZE
        
        #POLICY OPTIMIZATION
        #-----------------------------------------
        #clear past gradients
        self.pi_optimizer.zero_grad()
        #calculate the gaes
        td_errors = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        gaes = [np.sum(td_errors[l:] * taus[:len_trajectory-1-l] * gammas[:len_trajectory-1-l])
                for l in range(len_trajectory-1)]
        gaes = torch.tensor(gaes, dtype=torch.float32, device=self.device)
        
        #minus to make descent from ascent
        pi_loss = -(gaes.detach() * log_probs).sum()
        #calcualte gradients
        pi_loss.backward()
        #gradient descent
        self.pi_optimizer.step()
        #-----------------------------------------
        #VALUE FUNCTION OPTIMIZATION
        #-----------------------------------------        
        returns = np.array([np.sum(np.array(self.rewards[t:]) * gammas[:len_trajectory-t]) for t in range(len_trajectory)])
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
        values = torch.vstack(self.values).to(self.device)
        
        self.v_optimizer.zero_grad()
        # exclude the last value, because it contains the bootstrapped values
        v_loss = (returns[:-1] - values[:-1]).mul(0.5).pow(2).mean()
        # gradients for the value function
        v_loss.backward()
        #gradient descent
        self.v_optimizer.step()


Sources
=======

.. [#] Schulman J. et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. 2015. https://arxiv.org/abs/1506.02438