=============================
Prioritized Experience Replay
=============================

Motivation
==========

.. note::

    "In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently [#]_."

The experience replay is one of the major DQN components that make the algorithm so efficient. The agent is able to store already seen experiences and to reuse them several times in training before they are discarded. The drawback of the experience replay is the uniform probability with which each stored experience can be drawn. It is reasonable to assume that some experiences are more *important* and therefore better suited to learn from. 

.. figure:: ../../_static/images/reinforcement_learning/modern_value_based_approximation/per/prioritized_experience_replay.svg
   :align: center

   Prioritized Experience Replay.

This is where the prioritized experience replay (PER) comes into play. Each of the experience tuples has a priority assigned to it and the probability with which the tuple is likely to be drawn from the replay buffer and be used in training increases with higher priority. That way more important experiences are used more often and contribute to faster learning of the agent. 

Theory
======

According to the authors of PER, the ideal quantity of priority would be a measure of how much an agent can learn from a given experience. Such a measure is obviously not available and TD error is used as a proxy of priority. 

.. math:: 

    \delta = r + \gamma \max_{a'} Q(s', a', \theta^-) - Q(s, a, \theta)

TD error can be positive or negative, but we are only interested in the magnitude of the error and not in the direction. Therefore the absolute value of the error, :math:`| \delta |`,  is going to be used.

If we sampled only according to the magnitude of TD error, some experiences would not be sampled at all before they are discarded. This is especially problematic when we consider that we bootstrap and have only access to estimates of TD errors. Therefore we generally want to sample experiences with high TD error more often, but still have a non zero probability for experiences with low TD errors. 

If we sampled only according to the magnitude of TD error, some experiences would not be sampled at all before they are discarded. This is especially problematic when we consider that we bootstrap and have only access to estimates of TD errors. Therefore we generally want to sample experiences with high TD error more often, but still have a non zero probability for experiences with low TD errors. DeepMind proposes two approaches to calculate priorities.

* Proportilan priorization: :math:`p_i = |\delta_i| + \epsilon`, where :math:`\epsilon` is a positive constant that makes sure that experience tuples with a TD error of 0 still have a non-zero percent probability of being selected.
* Ranked-based priorization: :math:`p_i = \frac{1}{rank(i)}`, where :math:`rank(i)` is the index number of an experience tuple in a list, in which all absolute TD errors are sorted in descending order.

Ranked-based prioritization is expected to be less sensitive to outliers, therefore this approach is going to be utilized in this chapter.

Measuring TD errors for all experience tuples at each time step would be extremely inefficient, therefore the updates are done only periodically. The TD errors are updated only once they are drawn from the memory buffer and used in the training step. This is due to the fact that TD errors have to be calculated at the training step anyway and no additional computational power is therefore required. The calculations are not done for new experiences therefore each new experience tuple will receive the highest possible priority. 

.. math:: 

    P(i) = \frac{p^{\alpha}_i}{\sum_k p^{\alpha}_k}

The distribution of experience tuples is not only determined by the priority :math:`p_i`, but is additionally controlled by a constant :math:`\alpha`. If :math:`\alpha` is 0 the tuples are uniformly distributed. Higher numbers of :math:`\alpha` correspond to higher importance of priorities. 

If we are not careful and keep using the prioritized experience replay without any adjustment to the update step, we will introduce a bias. Let us assume that we possess the weights of the policy that minimize the mean squared error for the optimal policy. We utilize the policy and interact with the environment to fill the replay buffer. Lastly we want to recreate the weights for the above mentioned policy using the filled replay buffer. If we use the prioritized experience replay we utilize a different distribution than the one that is implied by the optimal weights (the uniform distribution). For example we might see rare experiences more often, which would imply gradient descent steps calculated based on rare experiences more often. On the one hand we want to use important experiences more often, but we would also like to avoid the bias. For that purpose we adjust the gradient descent step by a weight factor.

.. math::

    w_i = (\frac{1}{N} \cdot \frac{1}{P(i)})^\beta

The simplest way to imagine why the adjustment works is to imagine that we have uniform distribution. :math:`\frac{1}{P(i)}` becomes :math:`\frac{1}{\frac{1}{N}}` and the whole expression amounts to 1, indicating that the uniform distribution is already the correct one and we do not need any adjustments. The :math:`\beta` factor is used to control the correction factor. The requirement that we would like to impose is the uniform distribution at the end of the training. Therefore we start with a low :math:`\beta` and allow for stronger updates towards the rare experiences and increase the value over time to make full corrections.


Implementation
==============

.. code:: python

    class PER:
        
        def __init__(self, obs_shape, max_len, batch_size, alpha, beta, beta_increment):
            self.idx = 0
            self.max_len = max_len
            self.current_len = 0
            self.batch_size = batch_size
            self.alpha = alpha
            self.beta = beta
            self.beta_increment = beta_increment
            
            self.obs = np.zeros(shape=(max_len, *obs_shape), dtype=np.float32)
            self.action = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.reward = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.next_obs = np.zeros(shape=(max_len, *obs_shape), dtype=np.float32)
            self.done  = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.priorities  = np.zeros(shape=(max_len, 1), dtype=np.float32)
            
        def __len__(self):
            return self.current_len
        
        def anneal_beta(self):
            self.beta = min(self.beta + self.beta_increment, 1)
        
        def add_experience(self, obs, action, reward, next_obs, done):
            if len(self) == 0:
                priority = 1
            else:
                priority = self.priorities.max()
            
            self.obs[self.idx] = obs
            self.action[self.idx] = action
            self.reward[self.idx] = reward
            self.next_obs[self.idx] = next_obs
            self.done[self.idx] = done
            self.priorities[self.idx] = priority
            
            self.idx = (self.idx + 1) % self.max_len
            self.current_len = min(self.current_len + 1, self.max_len)
        
        def draw_samples(self):
            # rank based approach
            # -1 * is used to create descending order, argsort works in ascending order 
            # +1 at the end is needed to avoid divisions by 0 later
            ranks = np.argsort(-1 * self.priorities[:len(self)], axis=1)+1
            priorities = (1/ranks)**self.alpha
            p = priorities / np.sum(priorities)
            
            # it is easier to calculate the weights here and not during optimization
            weights = (1/len(self) * 1/(p))**self.beta
            idxs = np.random.choice(len(self), self.batch_size, replace=False, p=np.squeeze(p, axis=1))
            
            obs = self.obs[idxs]
            action = self.action[idxs]
            reward = self.reward[idxs]
            next_obs = self.next_obs[idxs]
            done = self.done[idxs]
            weights = weights[idxs]
            
            self.anneal_beta()
            return obs, action, reward, next_obs, done, idxs, weights
        
        def update_priority(self, idxs, td_errors):
            self.priorities[idxs] = np.abs(td_errors)


.. code:: python

    def optimize(self):
        if len(self.memory_buffer) < self.warmup:
            return
        
        self.optimizer.zero_grad()
        obs, action, reward, next_obs, done, idxs, weights = self.batch_memory()
                
        with torch.no_grad():
            target = reward + self.gamma * self.target_network(next_obs).max(dim=1, keepdim=True)[0] * (1 - done)

        
        online = self.online_network(obs).gather(dim=1, index=action)
                
        td_error = target - online
        loss = (weights * td_error).pow(2).mul(0.5).mean()
        loss.backward()
        self.optimizer.step()
        self.memory_buffer.update_priority(idxs, td_error.detach().cpu().numpy())
        
        self.adjust_epsilon()


Sources
=======

.. [#] Schaul T. et al. Prioritized Experience Replay. 2015. https://arxiv.org/abs//1511.05952