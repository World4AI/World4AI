=========
REINFORCE
=========

Motivation
==========

The policy gradient algorithm developed in the previous section shows high variance and thus requires a high number of trajectories. In this chapter we are going to start to develop methods that reduce variance and implement the first variant of an algorithm with less variance called REINFORCE.

Theory
======

Temporal Decomposition
----------------------

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) R(\tau^{(i)}) \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_t^H R_t \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) [\sum_{k=0}^{t} R_k + \sum_{k=t+1}^{H} R_k]
    \end{align*}

The policy gradient is calculated by multiplying the gradient of the log probability of an action :math:`A_t` given the state :math:`S_t` with the return from the whole trajectory :math:`\sum_t^H R_t`. The return of the trajectory can be decomposed into two distinct parts. The return that was already realized and can not be changed :math:`\sum_{k=0}^{t} R_k` and the return that is still to be earned and is going to be generated from the next step onward :math:`\sum_{k=t+1}^{H} R_k`.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} R_k
    \end{align*}

There is no need to multiply the log probability of an action with the past return. The action has no impact on the past. It turns out that ignoring past returns reduces the variance.

Discounting
-----------

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k
    \end{align*}

The second adjustment to the policy gradient algorithm that we are going to make is the introduction of discounting. Discounting that we have already seen with dynamic programming and q-learning accounts for the time value of rewards. Additionally discounting reduces variance in policy gradients methods. 

Implementation
==============

.. code:: python

    import gym
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions.categorical import Categorical

.. code:: python

    class PI(nn.Module):
    
        def __init__(self, num_features, num_actions):
            super(PI, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(in_features=num_features, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=num_actions)
            )
        
        def forward(self, state):
            x = self.model(state)
            distribution = Categorical(logits=x)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action.cpu().item(), log_prob

.. code:: python

    class Agent():
    
        def __init__(self,
                    env,
                    num_episodes,
                    solved_average_reward,
                    pi_function,
                    alpha,
                    gamma,
                    device):
            
            self.env = env
            self.num_episodes = num_episodes
            self.solved_average_reward = solved_average_reward
            self.device = device
            self.pi_function = pi_function.to(self.device)
            self.optimizer = optim.Adam(pi_function.parameters(), alpha)
            self.gamma = gamma
            
            self.reset()
            
        def reset(self):
            self.log_probs = []
            self.rewards = []
        
        def optimize(self):        
            len_trajectory = len(self.rewards)
            gammas = np.array([self.gamma**exp for exp in range(len_trajectory)])
            
            returns = np.array([np.sum(np.array(self.rewards[t:]) * gammas[:len_trajectory-t]) for t in range(len_trajectory)])
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
            log_probs = torch.vstack(self.log_probs).to(self.device)

            # optimize
            
            #clear past gradients
            self.optimizer.zero_grad()
            #minus to make descent from ascent
            loss = -(returns * log_probs).sum()
            #calcualte gradients
            loss.backward()
            #gradient descent
            self.optimizer.step()
            
        
        def learn(self):
            eval_rewards = []
            eval_rewards_mean = []
            avg_eval_reward_sum = float('-inf')
            
            max_eval_reward_sum = float('-inf')
            max_avg_eval_reward_sum = float('-inf')

            
            for episode in range(self.num_episodes):
                eval_reward_sum = 0
                obs = self.env.reset()
                done = False
                self.reset()

                while not done:
                    obs = torch.tensor(obs).unsqueeze(dim=0).to(self.device)
                    action, log_prob = self.pi_function(obs)
                    next_obs, reward, done, _ = self.env.step(action)

                    self.rewards.append(reward)
                    self.log_probs.append(log_prob)
                    obs = next_obs
                    
                    eval_reward_sum += reward
                    

                # EVALUATION AND LOGGING
                #-----------------------------------------------------------
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


                # policy gradient 
                self.optimize()


.. code:: python

    # PARAMETERS FOR LUNAR LANDER
    ENV_NAME = 'LunarLander-v2'
    ENV = gym.make(ENV_NAME)
    NUM_FEATURES = ENV.observation_space.shape[0]
    NUM_ACTIONS = ENV.action_space.n
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SOLVED_AVERAGE_REWARD = 200
    NUM_EPISODES = 5000
    ALPHA = 0.0005
    GAMMA = 0.99
    PI_FUNCTION = PI(NUM_FEATURES, NUM_ACTIONS)