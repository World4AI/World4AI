=========================================
Asynchronous Advantage Actor-Critic (A3C)
=========================================

Motivation
==========

    .. note::

        "The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU [#]_."

Policy gradient algorithms are on-policy algorithms. To see why this is the case let us look at the policy gradient.

.. math::
    :nowrap: 

    \begin{align*}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t)  \Psi_t] 
    \end{align*}

In order to caluculate the gradient, the expectation has to be estimated based on trajectories that are sampled with the help of the policy :math:`\pi_{\theta}`. But each time we perform a gradient descent step the weights :math:`\theta` are adjusted, which changes the policy and therefore changes the expectation. Therefore after each optimization step old experience tuples have to be thrown away and can not be reused in an off-policy fashion.

Because of its off-policy nature the experience replay that was used with value based reinforcement learning algorithms can not be used in the same way with policy gradient algorithms. But the memory buffer tried to solve the problem of highly correlated data that is generated in reinforcement learning. The asynchronous advantage actor-critic (A3C) tries to solve the problem by running the same algorithm in parallel on different processor cores. Each core has a distinct agent that interacts with its own instance of the environment and updates the policy periodically. It is reasonable to assume that each agent faces different states and rewards when interacting with the environment, thus reducing the correlation problem.

Theory
======

General Structure
-----------------

.. figure:: ../../_static/images/reinforcement_learning/actor_critic_methods_i/a3c/structure.svg
   :align: center

   A3C Structure.

At the core of the A3C algorithm as an asynchronous learning mechanism. The value and the policy function (implemented through a neural network) is initialized with shared weights that can be accessed by other processes on presumably different cores. Each agent that is spawned on a different process copies the shared weights into the local functions before each interaction sequence. After a certain number of steps or when the agent encounters the terminal state, the agent runs the gradient descent step on the shared weights. This process continues with the copying of the global shared weights. The copying and updating is done in an asynchronous, non blocking, way. That means that at the same time several agents can copy the same shared weights and update them based on the same policy. Sometimes the gradients that are copied back into the shared value and policy network might be overridden by a different agent. This algorithm, called Hogwild, still manages to work well, even when the agents that live on different processes do not communicate with each other. Many optimization algorithms like RMSprop and Adam have internal variables that are calculated based on previous optimization steps. These variables are also shared among all different agents. 

The A3C algorithm uses only different cores of the CPU and avoids the use of the GPU altogether. Still, according to the authors the algorithm outperforms the classical DQN algorithms in all Atari games in terms of computational speed and score performance. 

n-steps as :math:`\Psi`
-----------------------

The A3C algorithm calculates the advantage by using up to n interaction steps. If the environment terminates before n steps could be taken, then the optimization is done with the data that is available. In the original paper the authors use 5 interaction steps before optimization. The rest of the trajectory return is calculated through bootstrapping, which makes the algorithm an actor-critic algorithm. 

.. math::
   :nowrap:

   \begin{align*}
   & \Psi_{t+0} = R_{t+1} + R_{t+2} + R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t}) \\
   & \Psi_{t+1} = R_{t+2} + R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+1}) \\
   & \Psi_{t+2} = R_{t+3} + R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+2}) \\
   & \Psi_{t+3} = R_{t+4} + R_{t+5} + V(S_{t+5}) - V(S_{t+3}) \\
   & \Psi_{t+4} = R_{t+5} + V(S_{t+5}) - V(S_{t+4}) \\
   \end{align*}

After 5 steps the agent has 5 consecutive rewards at its disposal and can therefore use up to 5 rewards for the calculation of the advantage. At the end the agent can only use a single reward to estimate the advantage.

Entropy
-------

Entropy measures how much uncertainty is inherent in a probability distribution. The more certain the actions become the lower the entropy gets. If we assume for example a policy with two actions, then a policy which tends to select the same action for a given state with almost 100% probability will have a very low entropy. If on the other hand both actions will be selected with 50% probability for a given state, then the entropy will be high. 

In the A3C the authors add an entropy term to the gradient ascent step of the policy gradient algorithm.

.. math:: 

   \beta \nabla_{\theta} H(\pi_{\theta}(S_t))

:math:`H` calculates entropy and beta :math:`\beta` is used to measure the importance of the entropy. The reason for using entropy is to encourage exploration. The general idea of the policy gradient algorithm is to maximize actions with the highest advantage, but if the convergence to certain actions happen too soon, then it is possible that the agent misses on more favorable actions. Higher entropy forces the policy function to contain more uncertainty and therefore explore more.


Architecture
------------

.. figure:: ../../_static/images/reinforcement_learning/actor_critic_methods_i/a3c/architecture.svg
   :align: center

   A3C Architecture.

So far we have used separate networks for the agent and the critic. A3C uses the same neural network for the initial layers and only the last layer is separated into the policy and the value outputs. This approach is especially useful when several convolutional layers have to be trained in order to evaluate images and the weight sharing might facilitate training. 

In our implementation below we use shared weights, but also add additional separate layers at the end of the network. This approach  produced better results in tests with simple environments like Lunar Lander.  


Implementation
==============

.. code:: python

   import gym
   import numpy as np
   from numpy.core.shape_base import block

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.nn.modules.linear import Linear
   import torch.optim as optim
   import torch.multiprocessing as mp
   from torch.multiprocessing import Value, Queue
   from torch.distributions.categorical import Categorical

.. code:: python

   class SharedRMSprop(optim.RMSprop):
      def __init__(self, 
                  params, 
                  lr=0.01, 
                  alpha=0.99, 
                  eps=1e-08, 
                  weight_decay=0, 
                  momentum=0, 
                  centered=False):
         super(SharedRMSprop, self).__init__(
               params, 
               lr=lr, 
               alpha=alpha, 
               eps=eps, 
               weight_decay=weight_decay, 
               momentum=momentum, 
               centered=centered)
         
         for group in self.param_groups:
               for p in group['params']:
                  state = self.state[p]
                  if len(state) == 0:
                     state['step'] = 0
                     state['shared_step'] = torch.zeros(1).share_memory_()
                     state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format).share_memory_()
                     if group['momentum'] > 0:
                           state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format).share_memory_()
                     if group['centered']:
                           state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format).share_memory_()
                           
      def step(self):
         for group in self.param_groups:
               for p in group['params']:
                  if p.grad is None:
                     continue
                  state = self.state[p]
                  state['steps'] = state['shared_step'].item()
                  state['shared_step'] += 1
         
         super().step()

.. code:: python

   # the actor-critic shares the first layers
   class ActorCritic(nn.Module):
      
      def __init__(self, n_features, n_actions):
         super(ActorCritic, self).__init__()
         self.shared_model = nn.Sequential(
               nn.Linear(in_features=n_features, out_features=64),
               nn.ReLU(),
               nn.Linear(in_features=64, out_features=128),
               nn.ReLU(),
         )
         self.v = nn.Sequential(
               nn.Linear(in_features=128, out_features=128),
               nn.ReLU(),
               nn.Linear(in_features=128, out_features=1)
               )

         self.pi = nn.Sequential(
               nn.Linear(in_features=128, out_features=128),
               nn.ReLU(),
               nn.Linear(in_features=128, out_features=n_actions)
               )
      
      def forward(self, x):
         x = self.shared_model(x)
         v = self.v(x)
         logits = self.pi(x)
         
         distribution = Categorical(logits=logits)
         action = distribution.sample()
         log_prob = distribution.log_prob(action)
         entropy = distribution.entropy()
         
         return v, action.cpu().item(), log_prob, entropy

.. code:: python

   class Agent():
      def __init__(self,
                  agent_id, 
                  env_name,
                  n_features,
                  n_actions,
                  actor_critic_function,
                  max_episodes,
                  solved_average_reward,
                  shared_counter,
                  shared_return_tracker,
                  shared_actor_critic,
                  shared_optimizer,
                  n_step,
                  beta,
                  gamma):
         
         print(f'NEW AGENT WITH ID {agent_id} CREATED')
         self.agent_id = agent_id
         self.env = gym.make(env_name)
         self.local_actor_critic = actor_critic_function(n_features, n_actions)
         self.max_episodes = max_episodes
         self.solved_average_reward = solved_average_reward
         self.shared_counter = shared_counter
         self.shared_return_tracker = shared_return_tracker
         print(self.shared_return_tracker)
         self.shared_actor_critic = shared_actor_critic
         self.shared_optimizer = shared_optimizer
         self.n_step = n_step
         self.beta = beta
         self.gamma = gamma
      
      def reset(self):
         self.rewards = []
         self.values = []
         self.log_probs = []
         self.entropies = []
         
         self.local_actor_critic.load_state_dict(self.shared_actor_critic.state_dict())
         
      def optimize(self):
         len_trajectory = len(self.rewards)
         gammas = np.array([self.gamma**exp for exp in range(len_trajectory)])
         
         returns = np.array([np.sum(np.array(self.rewards[t:]) * gammas[:len_trajectory-t]) for t in range(len_trajectory-1)])
         returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(dim=1)
         log_probs = torch.vstack(self.log_probs)
         values = torch.vstack(self.values)
         entropies = torch.vstack(self.entropies)

         # OPTIMIZE
         
         #clear past gradients
         self.shared_optimizer.zero_grad()

         #OPTIMIZATION
         #-----------------------------------------
         #calculate the advantages
         advantages = returns - values
         #minus to make descent from ascent
         pi_loss = -(advantages.detach() * log_probs).mean()
         entropy_loss = - self.beta * entropies.mean()
         v_loss = advantages.mul(0.5).pow(2).mean()
         
         loss = pi_loss + entropy_loss + v_loss
         
         #calcualte gradients
         loss.backward()
         
         # push gradients into the shared network from the local network
         for local_param, shared_param in zip(self.local_actor_critic.parameters(), self.shared_actor_critic.parameters()):
               if shared_param.grad is None:
                  shared_param._grad = local_param.grad


         #gradient descent
         self.shared_optimizer.step()

      
      def learn(self):
         obs = self.env.reset()
         obs = torch.tensor(obs).unsqueeze(dim=0)
         self.reset()
         step = 0
         reward_sum = 0

         while True:
               
               value, action, log_prob, entropy = self.local_actor_critic(obs)
               next_obs, reward, done, _ = self.env.step(action)
               reward_sum+=reward
               step += 1

               self.rewards.append(reward)
               self.log_probs.append(log_prob)
               self.values.append(value)
               self.entropies.append(entropy)
               obs = next_obs
               obs = torch.tensor(obs).unsqueeze(dim=0)
               
               if done or step >= self.n_step:
                  # Bootstrapped value for the target
                  # We pack the value in the reward list to make the calculations easier
                  with torch.no_grad():
                     reward, _, _, _ = self.local_actor_critic(obs)
                     reward = reward.cpu().detach().item()
                  if done:
                     reward = 0
                  self.rewards.append(reward)
                  # gradient descent
                  self.optimize()
                  self.reset()
                  step = 0

                  if done:
                     self.shared_counter.value += 1
                     returns = self.shared_return_tracker.get()
                     returns.append(reward_sum)
                     reward_mean = 0
                     if len(returns) >= 100:
                           reward_mean = np.mean(returns[-100:])
                     self.shared_return_tracker.put(returns)

                     if self.shared_counter.value >= self.max_episodes:
                           break

                     if reward_mean > self.solved_average_reward:
                           print("SOLVED")
                           break

                     print(f'Game Nr: {self.shared_counter.value} Agent Nr: {self.agent_id} achieved reward of {reward_sum}. Average Reward: {reward_mean}')

                     obs = self.env.reset()
                     obs = torch.tensor(obs).unsqueeze(dim=0)
                     reward_sum = 0
                    
                    
.. code:: python

   # PARAMETERS FOR LUNAR LANDER
   ENV_NAME = 'LunarLander-v2'
   ENV = gym.make(ENV_NAME)
   N_FEATURES = ENV.observation_space.shape[0]
   N_ACTIONS = ENV.action_space.n
   SOLVED_AVERAGE_REWARD = 200
   MAX_EPISODES = 5000
   N_STEP = 5
   ALPHA = 0.0005
   BETA = 0.01 
   GAMMA = 0.99
   ACTOR_CRITIC_FUNCTION = ActorCritic
   NUM_WORKERS = 8

   # shared variables
   COUNTER = mp.Value('i', 0)
   RETURN_TRACKER = Queue()
   RETURN_TRACKER.put([])
   ACTOR_CRITIC = ActorCritic(N_FEATURES, N_ACTIONS).share_memory()
   ACTOR_CRITIC_OPTIMIZER = SharedRMSprop(ACTOR_CRITIC.parameters(), ALPHA)


.. code:: python

   def spawn_agent(agent_id, shared_counter, shared_return_tracker, shared_actor_critic, shared_optimizer):

      agent = Agent(
                  agent_id=agent_id,
                  env_name=ENV_NAME,
                  n_features=N_FEATURES,
                  n_actions=N_ACTIONS,
                  actor_critic_function=ACTOR_CRITIC_FUNCTION,
                  max_episodes=MAX_EPISODES,
                  solved_average_reward=SOLVED_AVERAGE_REWARD,
                  shared_counter=shared_counter,
                  shared_return_tracker=shared_return_tracker,
                  shared_actor_critic=shared_actor_critic,
                  shared_optimizer=shared_optimizer,
                  n_step=N_STEP,
                  beta=BETA,
                  gamma=GAMMA)

      agent.learn()
         
   if __name__ == '__main__':
      print('-----------------------------------------------------------------------')

      workers = [mp.Process(target=spawn_agent, args=(i+1, COUNTER, RETURN_TRACKER, ACTOR_CRITIC, ACTOR_CRITIC_OPTIMIZER)) for i in range(NUM_WORKERS)]
      [w.start() for w in workers] 
      [w.join() for w in workers]

      print('FINISHED')

Sources
=======

.. [#] Mnih V. et al. Asynchronous Methods for Deep Reinforcement Learning. 2016. https://arxiv.org/abs/1602.01783

