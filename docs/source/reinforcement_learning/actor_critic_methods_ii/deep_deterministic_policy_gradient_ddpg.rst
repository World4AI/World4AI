=========================================
Deep Deterministic Policy Gradient (DDPG)
=========================================

Motivation
==========

.. note::

    "We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain [#]_."


Theory
======

Implementation
==============

.. code:: python

    import gym
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from copy import deepcopy

.. code:: python

    class MemoryBuffer:
    
        def __init__(self, n_features, n_actions, max_len, batch_size):
            self.idx = 0
            self.max_len = max_len
            self.current_len = 0
            self.batch_size = batch_size
            
            self.obs = np.zeros(shape=(max_len, n_features), dtype=np.float32)
            self.action = np.zeros(shape=(max_len, n_actions), dtype=np.float32)
            self.reward = np.zeros(shape=(max_len, 1), dtype=np.float32)
            self.next_obs = np.zeros(shape=(max_len, n_features), dtype=np.float32)
            self.done  = np.zeros(shape=(max_len, 1), dtype=np.float32)
            
        def __len__(self):
            return self.current_len
        
        def add_experience(self, obs, action, reward, next_obs, done):
            self.obs[self.idx] = obs
            self.action[self.idx] = action
            self.reward[self.idx] = reward
            self.next_obs[self.idx] = next_obs
            self.done[self.idx] = done
            
            self.idx = (self.idx + 1) % self.max_len
            self.current_len = min(self.current_len + 1, self.max_len)
        
        def draw_samples(self):
            
            idxs = np.random.choice(len(self), self.batch_size, replace=False)
            
            obs = self.obs[idxs]
            action = self.action[idxs]
            reward = self.reward[idxs]
            next_obs = self.next_obs[idxs]
            done = self.done[idxs]
            
            return obs, action, reward, next_obs, done

.. code:: python

    class Q(nn.Module):
    
        def __init__(self, n_features, n_actions):
            super(Q, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(in_features=n_features+n_actions, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1)
            )
        
        def forward(self, state, action):
            x = torch.cat((state, action), dim=1)
            return self.model(x)

.. code:: python

    class PI(nn.Module):
    
        def __init__(self, n_features, n_actions, min_actions, max_actions):
            super(PI, self).__init__()
            self.min_actions = min_actions
            self.max_actions = max_actions
            self.model = nn.Sequential(
                nn.Linear(in_features=n_features, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=n_actions),
                nn.Tanh())
            
        def forward(self, state):
            return torch.clamp(self.model(state), self.min_actions, self.max_actions)


.. code:: python

    class Agent:
    
        def __init__(self,
                    env_name,
                    solved_average_reward,
                    batch_size,
                    buffer_size,
                    n_episodes,
                    warmup,
                    q_alpha,
                    pi_alpha,
                    gamma,
                    tau,
                    random_noise,
                    device):
            
            self.device = device
            self.env = gym.make(env_name)
            self.n_features = self.env.observation_space.shape[0]
            self.n_actions = self.env.action_space.shape[0]
            self.max_actions = self.env.action_space.high
            self.min_actions = self.env.action_space.low
            self.solved_average_reward = solved_average_reward
            
            # Two value functions
            self.q_online = Q(self.n_features, self.n_actions).to(self.device)
            self.q_target = deepcopy(self.q_online).to(self.device)
            for param in self.q_target.parameters():
                param.requires_grad = False
            self.q_optimizer = optim.Adam(self.q_online.parameters(), q_alpha)
            
            # Two policies
            self.pi_online = PI(self.n_features, 
                                self.n_actions, 
                                torch.tensor(self.min_actions).to(self.device), 
                                torch.tensor(self.max_actions).to(self.device)).to(self.device)
            self.pi_target = deepcopy(self.pi_online)
            for param in self.pi_target.parameters():
                param.requires_grad = False
            self.pi_optimizer = optim.Adam(self.pi_online.parameters(), pi_alpha)
            
            # memory buffer
            self.memory_buffer = MemoryBuffer(n_features=self.n_features, 
                                            n_actions=self.n_actions, 
                                            max_len=buffer_size, 
                                            batch_size=batch_size)
            
            self.n_episodes = n_episodes
            self.warmup = warmup
            
            self.gamma = gamma
            self.tau = tau
            self.random_noise = random_noise
        
        @torch.no_grad()
        def act(self, obs, noise):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
            action = self.pi_online(obs).squeeze(dim=0).cpu().numpy()
            action = action + noise * np.random.randn(self.n_actions)
            action = np.clip(action, self.min_actions, self.max_actions)
            return action
        
        def store_memory(self, obs, action, reward, next_obs, done):
            self.memory_buffer.add_experience(obs, action, reward, next_obs, done)
        
        def batch_memory(self):
            obs, action, reward, next_obs, done = self.memory_buffer.draw_samples()
            
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)
                    
            return obs, action, reward, next_obs, done
            
        def optimize(self):
            if len(self.memory_buffer) < self.warmup:
                return
            
            obs, action, reward, next_obs, done = self.batch_memory()
            
            #optimize critic
            self.q_optimizer.zero_grad()
            with torch.no_grad():
                next_action = self.pi_target(next_obs)
                target = reward + self.gamma * self.q_target(next_obs, next_action) * (1 - done)
                
            online = self.q_online(obs, action)

            td_error = target - online
            q_loss = td_error.pow(2).mul(0.5).mean()
            q_loss.backward()
            self.q_optimizer.step()
            
            #optimize actor
            self.pi_optimizer.zero_grad()
            pi_loss = -self.q_online(obs, self.pi_online(obs)).mean()
            pi_loss.backward()
            self.pi_optimizer.step()
            
            self.polyak_update()
            
        @torch.no_grad()
        def polyak_update(self):
            # q function
            for target, online in zip(self.q_target.parameters(), self.q_online.parameters()):
                target.data.mul_(1 - self.tau)
                target.data.add_(self.tau * online)
            
            # policy function
            for target, online in zip(self.pi_target.parameters(), self.pi_online.parameters()):
                target.data.mul_(1 - self.tau)
                target.data.add_(self.tau * online)
            
        def learn(self):
            counter = 0
            eval_rewards = []
            eval_rewards_mean = []
            avg_eval_reward_sum = float('-inf')
            
            max_eval_reward_sum = float('-inf')
            max_avg_eval_reward_sum = float('-inf')
            
            for episode in range(self.n_episodes):
                obs = self.env.reset()
                done = False

                while not done:
                    # TRAINING
                    counter += 1
                    action = self.act(obs, self.random_noise)
                    next_obs, reward, done, info = self.env.step(action)
                    self.store_memory(obs, action, reward, next_obs, done)
                    obs = next_obs
                    self.optimize()
                    
                # EVALUATION AND LOGGING
                #-----------------------------------------------------------
                eval_reward_sum = self.evaluate()
                eval_rewards.append(eval_reward_sum)
                
                if eval_reward_sum > max_eval_reward_sum:
                    max_eval_reward_sum = eval_reward_sum

                if len(eval_rewards) > 100:
                    avg_eval_reward_sum = np.mean(eval_rewards[-100:])
                    if avg_eval_reward_sum > max_avg_eval_reward_sum:
                        max_avg_eval_reward_sum = avg_eval_reward_sum
                        
                    eval_rewards_mean.append(avg_eval_reward_sum)
                
                print('--------------------------------')
                print(f'Episode: {episode + 1}')
                print(f'Reward Sum: {eval_reward_sum}')
                print(f'Max Reward Sum: {max_eval_reward_sum}')
                print(f'Avg. Reward Sum: {avg_eval_reward_sum}')
                print(f'Max Avg. Reward Sum: {max_avg_eval_reward_sum}')
                
                if avg_eval_reward_sum > self.solved_average_reward:
                    print('SOLVED')
                    break
    
        def evaluate(self):
            reward_sum = 0
            obs = self.env.reset()
            done = False
            while not done:
                action = self.act(obs, 0)
                next_obs, reward, done, info = self.env.step(action)
                obs = next_obs
                reward_sum += reward
            return reward_sum

.. code:: python

    # PARAMETERS FOR LUNAR LANDER
    ENV_NAME = 'LunarLanderContinuous-v2'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SOLVED_AVERAGE_REWARD = 200
    N_EPISODES = 1000
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    WARMUP = 1000
    Q_ALPHA = 0.00025
    PI_ALPHA = 0.0001
    GAMMA = 0.99
    TAU = 0.001
    RANDOM_NOISE = 0.1

    # create agent
    agent = Agent(
        env_name=ENV_NAME,
        solved_average_reward=SOLVED_AVERAGE_REWARD,
        batch_size=BATCH_SIZE,
        buffer_size=MEMORY_SIZE,
        n_episodes=N_EPISODES,
        warmup=WARMUP,
        q_alpha=Q_ALPHA, 
        pi_alpha=PI_ALPHA,
        gamma=GAMMA,
        tau=TAU,
        random_noise=RANDOM_NOISE,
        device=DEVICE
    )

    agent.learn()


Sources
=======

.. [#] Lillicrap T. et al. Continuous control with deep reinforcement learning". 2015. https://arxiv.org/abs/1509.02971