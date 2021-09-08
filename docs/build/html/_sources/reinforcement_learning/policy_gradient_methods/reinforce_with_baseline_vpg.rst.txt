========================================================
REINFORCE With Baseline | Vanilla Policy Gradient  (VPG)
========================================================

Motivation
==========

The reinforce algorithm implemented in the last chapter is the first viable variant of a policy gradient method, yet not the one with the least variance. In this chapter we introduce an improvement to REINFORCE. The algorithm is called REINFORCE with baseline, but sometimes the name vanilla policy gradient (VPG) is also used. 

Theory
======

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k
    \end{align*}

The gradient calculation for REINFORCE can be interpreted as follows. The log probability of actions that generate high returns is increased, while the log probability of actions that generate negative returns is decreased.

.. figure:: ../../_static/images/reinforcement_learning/policy_gradient_methods/baseline/no_baseline.svg
   :align: center

   Policy Gradient Without Baseline.

But what if all returns are positive or clustered like in the image above? The probability of actions with highest returns will increase more than those with lower returns. But the process is slow.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - b
    \end{align*}

The next improvement we are going to introduce is the baseline. The baseline :math:`b` is deducted from the return, which reduces the variance. 

.. figure:: ../../_static/images/reinforcement_learning/policy_gradient_methods/baseline/baseline.svg
   :align: center

   Policy Gradient With Baseline.

Intuitively some of the positive returns might stay positive, while others are pushed below the zero line. That makes the gradient positive for returns above the baseline and negative for returns below the baseline. There are different choices for the baseline :math:`b`, which has an impact on how much variance and bias the algorithm has. 

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - V(S_t^{(i)})
    \end{align*}

REINFORCE with baseline uses the state value function :math:`V(S_t)` as the baseline. This makes perfect sense as only the probability of those actions are increased that generate returns that are above the expected sum of rewards. In our implementation  :math:`V(S_t)` is going to be a learned neural network, meaning that we have two separate functions, one for the policy and one for the state value. In VPG the policy and value functions are updated with monte carlo simulations, therefore this algorithm is only suited for episodic tasks.

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

    class V(nn.Module):
    
        def __init__(self, num_features):
            super(V, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        def forward(self, state):
            x = self.model(state)
            return x

.. code:: python

    class Agent():
    
        def __init__(self,
                    env,
                    num_episodes,
                    solved_average_reward,
                    pi_function,
                    v_function,
                    pi_alpha,
                    v_alpha,
                    gamma,
                    device):
            
            self.env = env
            self.num_episodes = num_episodes
            self.solved_average_reward = solved_average_reward
            self.device = device
            self.pi_function = pi_function.to(self.device)
            self.v_function = v_function.to(self.device)
            self.pi_optimizer = optim.Adam(pi_function.parameters(), pi_alpha)
            self.v_optimizer = optim.Adam(v_function.parameters(), v_alpha)
            self.gamma = gamma
            
            self.reset()
            
        def reset(self):
            self.log_probs = []
            self.rewards = []
            self.values = []
        
        def optimize(self):        
            len_trajectory = len(self.rewards)
            gammas = np.array([self.gamma**exp for exp in range(len_trajectory)])
            
            returns = np.array([np.sum(np.array(self.rewards[t:]) * gammas[:len_trajectory-t]) for t in range(len_trajectory)])
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
            log_probs = torch.vstack(self.log_probs).to(self.device)
            values = torch.vstack(self.values).to(self.device)

            # OPTIMIZE
            
            #POLICY OPTIMIZATION
            #-----------------------------------------
            #clear past gradients
            self.pi_optimizer.zero_grad()
            #calculate the advantages
            advantages = returns - values
            #minus to make descent from ascent
            pi_loss = -(advantages.detach() * log_probs).sum()
            #calcualte gradients
            pi_loss.backward()
            #gradient descent
            self.pi_optimizer.step()
            #-----------------------------------------
            #VALUE FUNCTION OPTIMIZATION
            #-----------------------------------------
            self.v_optimizer.zero_grad()
            v_loss = advantages.mul(0.5).pow(2).mean()
            # gradients for the value function
            v_loss.backward()
            #gradient descent
            self.v_optimizer.step()
        
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
                    value = self.v_function(obs)
                    next_obs, reward, done, _ = self.env.step(action)

                    self.rewards.append(reward)
                    self.log_probs.append(log_prob)
                    self.values.append(value)
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
    PI_ALPHA = 0.0005
    V_ALPHA = 0.0005
    GAMMA = 0.99

    PI_FUNCTION = PI(NUM_FEATURES, NUM_ACTIONS)
    V_FUNCTION = V(NUM_FEATURES)