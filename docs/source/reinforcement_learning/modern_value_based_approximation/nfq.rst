=========================
Neural Fitted Q Iteration
=========================

The naive implementation of Q-Learning with neural networks from a previous chapter was an online learning implementation. The algorithm used an experience once and threw the information away. 

.. note:: 

    An algorithm **A** is said to be more sample efficient than algorithm **B** if algorithm **A** requires less samples to learn a task. 

In that sense approximate online q-learning was not particularly sample efficient. Neural Fitted Q Iteration (NFQ) takes a different approach [#]_. Neural Fitted Q Iteration (NFQ) takes a different approach. The experience is collected in a sample set :math:`\mathcal{D}` and the whole set is used to reduce the mean squared error. The training is similar to supervised learning, where the same data set is iterated over and over again, until the desired convergence is achieved. 

There are several variants regarding how the collection of experiences can be handled. The first variant, that is also going to be used in our implementation below, samples the whole experience set by performing random actions before any training is done. An additional and probably a better variant is to fill the sample set gradually, by iterating between collection and training. This variant makes especially a lot of sense when certain observations can not be reached through random actions. 

If you study the paper by Riedmiller, you will notice that the algorithms we are using are deviating from the source material. The original paper uses cost (negative reward), instead of reward. This is less common in modern literature and we are going to stick to rewards. Keep this in mind when you notice expressions where q-values need to be minimized and not maximized. 

Algorithm
=========

.. math::
    :nowrap:
  
    \begin{algorithm}[H]
        \caption{Neural Fitted Q Iteration}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: Environment for interaction, $env$
        \STATE Input: Neural Network initialized randomly (q-function) $\hat{q}$
        \STATE Input: Number of episodes for data collection
        \STATE Input: Number of epochs
        \STATE Input: Batch Size
        \STATE Input: Learning Rate alpha $\alpha$
        \STATE Input: Discount Rate gamma $\gamma$
        \FOR{$episode=0$ to number of episodes}
          \STATE Initialize $env$ and get initial state $S$
          \REPEAT
            \STATE Select action $A$ based on $S$ using $\epsilon$-greedy action selection
            \STATE Observe reward $R$ and state $S'$
            \STATE Add $S, A, R, S'$ to $\mathcal{D}$
            \STATE $S \leftarrow S'$
          \UNTIL{State $S'$ is terminal}
        \ENDFOR
        \FOR{$episode=0$ to number of epochs}
            \FOR{batch in $\mathcal{D}$}
                \STATE Update weights $\mathbf{w}$ for a whole batch
                \STATE $\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma\max_a\hat{q}(S',a, \mathbf{w}) - \hat{q}(S, A, \mathbf{w})]\nabla\hat{q}(S, A, \mathbf{w})$
            \ENDFOR
        \ENDFOR
    \end{algorithmic}
    \end{algorithm}
  

PyTorch Implementation
======================

.. code:: python

    import numpy as np
    import gym

    import math
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

.. code:: python

    class Q(nn.Module):
    
        def __init__(self):
            super(Q, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(5, 10),
                nn.Sigmoid(),
                nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, 10),
                nn.Sigmoid(),
                nn.Linear(10, 1)
            )
            
        def forward(self, state, action):
            x = torch.cat((state, action), dim=1)
            return self.model(x)

.. code:: python

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

.. code:: python

    # Creating a dataset
    class TrainDataset(Dataset):
        
        def __init__(self, obss, actions, rewards, next_obss, dones):
            self.obss = obss
            self.actions = actions
            self.rewards = rewards
            self.next_obss = next_obss
            self.dones = dones
        
        def __getitem__(self, idx):
            obs = self.obss[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            next_obs = self.next_obss[idx]
            done = self.dones[idx]
            
            return torch.tensor(obs), torch.tensor(action), torch.tensor(reward), torch.tensor(next_obs), torch.tensor(done)
        
        def __len__(self):
            return len(self.obss)

.. code:: python

    class Agent:
    
        def __init__(self, env, q_function, num_episodes, num_epochs, batch_size, alpha, gamma):
            self.env = env
            self.q_function = q_function
            self.num_episodes = num_episodes
            self.num_epochs = num_epochs
            self.gamma = gamma
            
            self.optimizer = optim.RMSprop(q_function.parameters(), alpha)
            self.criterion = nn.MSELoss()
            
            # Collect Experiences
            obss, actions, rewards, next_obss, dones = self.collect_experiences()
            train_dataset = TrainDataset(obss, actions, rewards, next_obss, dones)
            self.D = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            
            self.num_batches = math.ceil(len(train_dataset) / batch_size)
            #SOME DATA STRUCTURES FOR LOGGING
            # save sum of rewards after each evaluation episode
            self.rewards_list = []
            # save average reward for each 100 episodes
            self.avg_rewards_list = []
            # max sum of rewards seen so far in a single game
            self.max_reward = 0
            # max average sum of rewards over 100 games seen so far
            self.max_avg_rewards = 0

        
        def collect_experiences(self):
            obss = []
            actions = []
            rewards = []
            next_obss = []
            dones = []

            for episode in range(self.num_episodes):
                done = False
                obs = self.env.reset()
                while not done:
                    action = self.env.action_space.sample()
                    next_obs, reward, done, info = self.env.step(action)

                    obss.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    next_obss.append(next_obs)
                    dones.append(done)

                    obs = next_obs

            # transform the list into numpy arrays        
            obss = np.array(obss, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_obss = np.array(next_obss, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            # add additional axes e.g. (5, ) to (5, 1)
            actions = np.expand_dims(actions, axis=1)
            rewards = np.expand_dims(rewards, axis=1)
            dones = np.expand_dims(dones, axis=1)

            return obss, actions, rewards, next_obss, dones 
        
        def evaluate(self, epoch, batch):
            done = False
            obs = self.env.reset()
            reward_sum = 0
            while not done:
                with torch.no_grad():
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0).to(DEVICE)
                    action_left = torch.tensor([[0]], dtype=torch.float32).to(DEVICE)
                    action_right = torch.tensor([[1]], dtype=torch.float32).to(DEVICE)

                    if self.q_function(obs, action_left) > self.q_function(obs, action_right):
                        action = 0
                    else:
                        action = 1

                next_obs, reward, done, _ = self.env.step(action)
                reward_sum += reward
                obs = next_obs
                
            self.rewards_list.append(reward_sum)
            if reward_sum > self.max_reward:
                self.max_reward = reward_sum
            
            if len(self.rewards_list) >= 100:
                avg_reward = np.mean(self.rewards_list[-100:])
                if avg_reward > self.max_avg_rewards:
                    self.max_avg_rewards = avg_reward
                self.avg_rewards_list.append(avg_reward)
                print('--------------------------------')
                print(f'Episode: {epoch}/{self.num_epochs}')
                print(f'Episode: {batch}/{self.num_batches}')
                print(f'Reward Sum: {reward_sum}')
                print(f'Max Reward Sum: {self.max_reward}')
                print(f'Avg. Reward Sum: {avg_reward}')
                print(f'Max Avg. Reward Sum: {self.max_avg_rewards}')

        
        def learn(self):
        
            for epoch in range(self.num_epochs):
                for batch, (obss, actions, rewards, next_obss, dones) in enumerate(self.D):
                    obss = obss.to(DEVICE)
                    actions = actions.to(DEVICE)
                    rewards = rewards.to(DEVICE)
                    next_obss = next_obss.to(DEVICE)
                    dones = dones.to(DEVICE)

                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        action_left = torch.zeros((rewards.shape[0], 1), dtype=torch.float32).to(DEVICE)
                        action_right = torch.ones((rewards.shape[0], 1), dtype=torch.float32).to(DEVICE)

                        target_left = rewards + GAMMA * self.q_function(next_obss, action_left) * (1 - dones)
                        target_right = rewards + GAMMA * self.q_function(next_obss, action_right) * (1 - dones)
                        target = torch.maximum(target_left, target_right)

                    online = self.q_function(obss, actions)

                    loss = self.criterion(online, target)
                    loss.backward()

                    self.optimizer.step()                
                    self.evaluate(epoch+1, batch+1)


.. code:: python

    # CONSTANTS
    ENV = gym.make('CartPole-v1')
    NUM_EPISODES = 1000
    ALPHA = 0.01
    GAMMA = 0.95
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    Q_FUNCTION = Q()
    Q_FUNCTION.to(DEVICE)

    agent = Agent(env=ENV, 
              q_function=Q_FUNCTION, 
              num_episodes=NUM_EPISODES, 
              num_epochs=NUM_EPOCHS, 
              batch_size=BATCH_SIZE, 
              alpha=ALPHA, 
              gamma=GAMMA)

    agent.learn()

Sources
=======
.. [#] Riedmiller M. (2005) Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method. In: Gama J., Camacho R., Brazdil P.B., Jorge A.M., Torgo L. (eds) Machine Learning: ECML 2005. ECML 2005. Lecture Notes in Computer Science, vol 3720. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11564096_32