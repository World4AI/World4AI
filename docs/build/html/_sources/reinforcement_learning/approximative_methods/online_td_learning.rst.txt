==================
Online TD Learning
==================

.. code:: python

    import gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np


.. code:: python

    class Q(nn.Module):
    
        def __init__(self, observation_len, action_len):
            super(Q, self).__init__()
            self.fc_1 = nn.Linear(observation_len, 64)
            self.fc_2 = nn.Linear(64, 128)
            self.fc_3 = nn.Linear(128, action_len)
        
        def forward(self, observation):
            x = F.relu(self.fc_1(observation))
            x = F.relu(self.fc_2(x))
            x = self.fc_3(x)
            return x


.. code:: python

    class Agent():

        def __init__(self, 
                    observation_len, 
                    action_len,
                    alpha=0.01, 
                    epsilon_start=1.0, 
                    epsilon_end=0.1, 
                    epsilon_step=0.0001, 
                    gamma=0.95):
            
            self.q = Q(observation_len, action_len)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.q.to(self.device)
            self.optimizer = optim.RMSprop(self.q.parameters(), alpha)
            self.loss = nn.MSELoss()
            
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_step = epsilon_step
            self.gamma = gamma
            
            
            self.action_len = action_len
            
        def decrease_epsilon(self):
            self.epsilon -= self.epsilon_step
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end
        
        def select_action(self, observation):
            
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_len)
            else:
                with torch.no_grad():
                    observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
                    action = self.q(observation).argmax().item()
            
            return action
        
        def learn(self, obs, action, reward, next_obs, done):
            self.optimizer.zero_grad()
            
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward, dtype = torch.float32).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                target = reward + self.gamma * self.q(next_obs).max() * (not done)
            
            online = self.q(obs)[action]
            
            loss = self.loss(target, online).to(self.device)
            loss.backward()
            
            self.optimizer.step()
            self.decrease_epsilon()

.. code:: python

    num_episodes = 10000
    env = gym.make('LunarLander-v2')
    action_len = env.action_space.n
    observation_len = env.observation_space.shape[0]
    agent = Agent(observation_len, action_len)


.. code:: python

    reward_tracking = []
    for episode in range(num_episodes):
        
        obs = env.reset()
        done = False
        reward_sum = 0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            reward_sum += reward
        
        reward_tracking.append(reward_sum)
        if (episode + 1) % 100 == 0:
            print(f'Episode: {episode+1}, Reward: {reward_sum}, Mean: {np.array(reward_tracking[-100:]).mean()}')