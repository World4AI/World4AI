=============================
Prioritized Experience Replay
=============================

Motivation
==========

.. note::

    "In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently [#]_."

Theory
======

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