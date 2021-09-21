=======================
Twin Delayed DDPG (TD3)
=======================

Motivation
==========

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

.. code::

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

.. code::python

    class Q(nn.Module):
        
        def __init__(self, n_features, n_actions):
            super(Q, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(in_features=n_features+n_actions, out_features=400),
                nn.ReLU(),
                nn.Linear(in_features=400, out_features=300),
                nn.ReLU(),
                nn.Linear(in_features=300, out_features=1)
            )
        
        def forward(self, state, action):
            x = torch.cat((state, action), dim=1)
            return self.model(x)

    class PI(nn.Module):
    
        def __init__(self, n_features, n_actions, min_actions, max_actions):
            super(PI, self).__init__()
            self.min_actions = min_actions
            self.max_actions = max_actions
            self.model = nn.Sequential(
                nn.Linear(in_features=n_features, out_features=400),
                nn.ReLU(),
                nn.Linear(in_features=400, out_features=300),
                nn.ReLU(),
                nn.Linear(in_features=300, out_features=n_actions),
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
                    exploration_noise,
                    target_noise,
                    target_noise_lim,
                    policy_update_delay,
                    device):
            
            self.device = device
            self.env = gym.make(env_name)
            self.n_features = self.env.observation_space.shape[0]
            self.n_actions = self.env.action_space.shape[0]
            self.max_actions = self.env.action_space.high
            self.min_actions = self.env.action_space.low
            self.solved_average_reward = solved_average_reward
            
            # Two value functions + two target value functions
            self.q1 = Q(self.n_features, self.n_actions).to(self.device)
            self.q1_target = deepcopy(self.q1).to(self.device)
            for param in self.q1_target.parameters():
                param.requires_grad = False
            self.q1_optimizer = optim.Adam(self.q1.parameters(), q_alpha)
            
            self.q2 = Q(self.n_features, self.n_actions).to(self.device)
            self.q2_target = deepcopy(self.q2).to(self.device)
            for param in self.q2_target.parameters():
                param.requires_grad = False
            self.q2_optimizer = optim.Adam(self.q2.parameters(), q_alpha)
            
            # One policy + one target policy
            self.pi = PI(self.n_features, 
                                self.n_actions, 
                                torch.tensor(self.min_actions).to(self.device), 
                                torch.tensor(self.max_actions).to(self.device)).to(self.device)
            self.pi_target = deepcopy(self.pi)
            for param in self.pi_target.parameters():
                param.requires_grad = False
            self.pi_optimizer = optim.Adam(self.pi.parameters(), pi_alpha)
            
            # memory buffer
            self.memory_buffer = MemoryBuffer(n_features=self.n_features, 
                                            n_actions=self.n_actions, 
                                            max_len=buffer_size, 
                                            batch_size=batch_size)
            
            self.n_episodes = n_episodes
            self.warmup = warmup
            
            self.gamma = gamma
            self.tau = tau
            self.exploration_noise = exploration_noise
            self.target_noise = target_noise
            self.target_noise_lim = target_noise_lim
            self.policy_update_delay = policy_update_delay
        
        @torch.no_grad()
        def act(self, obs, noise):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
            action = self.pi(obs).squeeze(dim=0).cpu().numpy()
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
            
        def optimize(self, counter):
            if len(self.memory_buffer) < self.warmup:
                return
            
            obs, action, reward, next_obs, done = self.batch_memory()
            
            #optimize critics
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            
            with torch.no_grad():
                next_action = self.pi_target(next_obs)
                epsilon = torch.randn_like(next_action) * self.target_noise
                epsilon = torch.clamp(epsilon, -self.target_noise_lim, self.target_noise_lim)
                next_action += epsilon
                next_action = torch.clamp(next_action, 
                                        torch.tensor(self.min_actions).to(self.device),
                                        torch.tensor(self.max_actions).to(self.device))
                
                q1_target = self.q1_target(next_obs, next_action)
                q2_target = self.q2_target(next_obs, next_action)
                q_target = torch.min(q1_target, q2_target)
                target = reward + self.gamma * q_target * (1 - done)
                
            q1 = self.q1(obs, action)
            q2 = self.q2(obs, action)

            td_error1 = target - q1
            td_error2 = target - q2
            q1_loss = td_error1.pow(2).mul(0.5).mean()
            q2_loss = td_error2.pow(2).mul(0.5).mean()
            q1_loss.backward()
            q2_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()
            
            
            if counter % self.policy_update_delay == 0:
                
                #optimize actor
                self.pi_optimizer.zero_grad()
                pi_loss = -self.q1(obs, self.pi(obs)).mean()
                pi_loss.backward()
                self.pi_optimizer.step()

                self.update_target_q()
                self.update_target_pi()
            
        
        @torch.no_grad()
        def update_target_q(self):
            # q function
            for q1_target, q1 in zip(self.q1_target.parameters(), self.q1.parameters()):
                q1_target.data.mul_(1 - self.tau)
                q1_target.data.add_(self.tau * q1)
                
            for q2_target, q2 in zip(self.q2_target.parameters(), self.q2.parameters()):
                q2_target.data.mul_(1 - self.tau)
                q2_target.data.add_(self.tau * q2)
        
        @torch.no_grad()
        def update_target_pi(self):
            # policy function
            for pi_target, pi in zip(self.pi_target.parameters(), self.pi.parameters()):
                pi_target.data.mul_(1 - self.tau)
                pi_target.data.add_(self.tau * pi)
            
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
                    action = self.act(obs, self.exploration_noise)
                    next_obs, reward, done, info = self.env.step(action)
                    self.store_memory(obs, action, reward, next_obs, done)
                    obs = next_obs
                    self.optimize(counter)
                    
                    
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
    BATCH_SIZE = 100
    MEMORY_SIZE = 10000
    WARMUP = 1000
    Q_ALPHA = 1e-3
    PI_ALPHA = 1e-3
    GAMMA = 0.99
    TAU = 5e-3
    EXPLORATION_NOISE = 0.1
    TARGET_NOISE = 0.2
    TARGET_NOISE_LIM = 0.5
    POLICY_UPDATE_DELAY = 2

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
        exploration_noise=EXPLORATION_NOISE,
        target_noise=TARGET_NOISE,
        target_noise_lim=TARGET_NOISE_LIM,
        policy_update_delay=POLICY_UPDATE_DELAY,
        device=DEVICE
    )

    agent.learn()

Sources
=======