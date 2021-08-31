===================
Monte Carlo Methods
===================

Motivation
==========

.. note:: 
    Monte Carlo methods are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results [#]_.

If we look at any definition of Monte Carlo methods, there is a high chance that the definition contains random sampling. 

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/monte_carlo/paths.svg
   :align: center

   Paths Generated Through Monte Carlo.

When we apply Monte Carlo methods to reinforcement learning we sample episode paths, also called trajectories. The agent interacts with the environment and collects experience tuples that consist of states, actions and rewards. 

Monte Carlo methods are similar in spirit to bandit methods. The state-value and action-value functions can be estimated by taking the sampled trajectories and building averages. Unlike in bandits though, Monte Carlo methods are able to deal with environments where several non terminal states exist.

Estimations can only be made once the trajectory is complete when the episode finishes, which means that Monte Carlo methods only work for episodic tasks. 

Generalized Policy Iteration
============================

The Monte Carlo algorithm will follow general policy iteration. We alternate between policy evaluation and policy improvement to find the optimal policy.

Policy Estimation
-----------------

Theory
######

Policy estimation deals with finding the true value function of a given policy :math:`\pi`. Mathematically speaking we are looking for the expected sum of discounted rewards (also called returns) when the agent follow the policy :math:`\pi`. 

:math:`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`

A natural way to estimate the expected value of a random variable is to get samples from a distribution and to use the average as an estimate. In reinforcement learning the agent can estimate the expected value of returns for a policy :math:`\pi` by interacting with the environment, generating trajectories over and over again and building averages over the returns of the trajectories. 

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/monte_carlo/mc_backup.svg
   :align: center

   Monte Carlo Trajectories.

The above image shows different trajectories that were created by following a policy pi. The large circles represent a certain state, the smaller black circles represent the actions and the boxes are the terminal states. To estimate the state value for the grey state located at the left we calculate the (discounted) return that is generated after the grey state and build a simple average. The same process is repeated for the yellow, green and all the other states. 

Generally there are two methods to calculate the averages. Each time the agent faces a state during an episode is called a visit. In the “First Visit” Monte Carlo method only the return from the first visit to that state until the end of the episode is calculated. If the state is visited several times during an episode, the additional visits are not considered in the calculation. While in the “Every Visit” method each visit is counted. The “First Visit” method is more popular and generally more straightforward and is going to be covered in this section, but the algorithms can be easily adjusted to account for the “Every Visit” method. 

To make the calculations of the averages computationally efficient we are going to use the incremental implementation that we already used for n-armed bandits.

.. math::

    NewEstimate \leftarrow OldEstimate + StepSize[Target - OldEstimate]

Algorithm
#########

The algorithm is divided into two steps. 

* In the first step using the policy :math:`\pi` (or :math:`\mu` if the policy is deterministic) the agent generates a trajectory.
* In the second step the agent improves the estimation for the state value function :math:`V(s)`. For that purpose the agent loops over the previously generated trajectory and for each experience tuple he determines if he deals with a first visit to that state :math:`s`. If he does he calculates the discounted sum of rewards from that point on to the terminal state :math:`G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t`. Finally the agent performs an update step by using the incremental average calculation :math:`V(s) = V(s) + \alpha [G_{t:T} - V(s)]`.


.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Monte Carlo Prediction: First Visit}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: environment $env$, policy $\mu$, state set $\mathcal{S}$, number of episodes, learning rate $\alpha$, discount factor $\gamma$
        \STATE Initialize: 
        \STATE $V(s)$ for all $s \in \mathcal{S}$ with zeros
        \FOR{$i=0$ to number of episodes}
            \STATE ...................................................................................................................................
            \STATE (1) INTERACTION WITH THE ENVIRONMENT
            \STATE create $trajectory$ as empty list [...]
            \REPEAT
                \STATE Generate experience $tuple$ $(State, Action, Reward)$ using policy $\mu$ and MDP $env$ 
                \STATE Push $tuple$ into $trajectory$
            \UNTIL{state is terminal}
            \STATE ...................................................................................................................................
            \STATE (2) ESTIMATION OF VALUE FUNCTION
            \STATE Create Visited(s) = $False$ for all $s \in \mathcal{S}$
            \FOR{$t=0$ to number of tuples in $trajectory$ $T$}
                \STATE $s \leftarrow$ state from $trajectory[t]$
                \IF {Visited($s$) is True}
                    \STATE go to next tuple
                \ELSE
                    \STATE Visited($s$) = True 
                \ENDIF
                \STATE $G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t$
                \STATE $V(s) = V(s) + \alpha [G_{t:T} - V(s)]$
            \ENDFOR
        \ENDFOR
        \STATE
        \STATE RETURN V(s)
    \end{algorithmic}
    \end{algorithm}

Implementation
##############

Once again we are going to utilize the already discussed frozen lake environment to demonstrate the implementation of the algorithms.

.. code:: python

    import gym
    import numpy as np

.. code:: python

    env = gym.make('FrozenLake-v0')


.. code:: python

    S = [x for x in range(env.observation_space.n)]
    A = [x for x in range(env.action_space.n)]

The below policy is going to be evaluated.  

.. code:: python

    def policy(state):
    #     LEFT = 0 
    #     DOWN = 1 
    #     RIGHT = 2 
    #     UP = 3 

        mu = {
            0: 2,
            1: 2,
            2: 1,
            3: 0,
            4: 1, 
            5: 1,
            6: 1,
            7: 1,
            8: 2,
            9: 1,
            10: 1,
            11: 1,
            12: 2,
            13: 2,
            14: 2,
            15: 2   
        }
        
        return mu[state]

The following code shows a relatively straightforward implementation of Monte Carlo evaluation. To get better results we could still increase the number of episodes and reduce the learning rate :math:`\alpha` over time.

.. code:: python

    def mc_prediction(env, policy, S, num_episodes=100000, alpha=0.001, gamma=0.99):
    
        V = np.zeros(len(S))
        
        for episode in range(num_episodes):
            
            # generate episode
            episode = []
            state = env.reset()
            done = False
            
            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                experience = (state, action, reward)
                episode.append(experience)
                state = next_state
            
            # makes an array with nr. timesteps as rows and 3 columns
            episode = np.array(episode, dtype=np.int32)
            
            time_steps = len(episode)
            
            # discounts
            discounts = np.array([gamma**time_step for time_step in range(time_steps)])
            
            # update state-value function
            visited = np.zeros(len(S), dtype=np.bool_)
            for time_step, (state, _, _) in enumerate(episode):
                
                if visited[state]:
                    continue
                else:
                    visited[state] = True
                
                returns = episode[time_step:, 2]
                remain_steps = len(returns)
                G = np.sum(returns * discounts[:remain_steps])
                
                V[state] = V[state] + alpha * (G - V[state])
                    
        return V

The following state-value function is calculated for the policy :math:`\mu(s)`::

    0.03475|0.02148|0.04602|0.02590|
    0.05337|0.00000|0.09429|0.00000|
    0.11221|0.28622|0.30213|0.00000|
    0.00000|0.46326|0.64936|0.00000|

Policy Improvement and Control
------------------------------

Theory
######

The value iteration algorithm that we applied in the dynamic programming section used the following update step.

.. math::
    :nowrap:

    \begin{align*}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
    \end{align*}


This exact update step is not going to work with Monte Carlo methods, because that would require the full knowledge of the model. We would have to know the transition probabilities from state :math:`s` to state :math:`s'` and the corresponding reward.

If we look closely at the above expression, we should notice that we can rewrite the update rule in terms of an action-value function?

.. math::
    :nowrap:

    \begin{align*}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    & = \max_a q_k(s, a) \\
    \end{align*}


With those rewrites we do not require the knowledge of the model, but it becomes obvious that the key is to estimate the action-value function and not the state-value function. Having an estimate of an action-value function allows the agent to select better actions by acting greedily and to gradually improve the policy towards the optimal policy. To estimate the action-value function we will still generate episodes and compute averages, but the averages are not going to be for a state, but for a state-action pair. 

There is still one problem that we face without the knowledge of the model of the MDP though. If our policy is fully deterministic and thus avoids some state-action pairs by design, then we can not compute a good estimate for certain state-action pairs and thus might not arrive at the optimal policy. The solution is to use an :math:`\epsilon`-greedy policy, meaning that with a probability of :math:`\epsilon` we take a random action and with probability of :math:`1-\epsilon` we take the greedy action. That way we are guaranteed that all state-action pairs are going to be visited.

.. note::

    **On-policy** methods improve the same policy that is used to generate the trajectory.

    **Off-policy** methods improve a policy that is different from the one that is used to generate the trajectory.

Before we move on to the implementation of the Monte Carlo control algorithm it is important to discuss the difference between on-policy and off-policy methods. Once the need arises to explore the environment we could ask ourselves, “Do we need to improve the same policy that is used to generate actions or can we learn the optimal policy while using the data that was produced by a different policy?”. To frame the question differently “Is it possible to learn the optimal policy while only selecting random actions?”. That depends on the design of the algorithm. On-policy methods improve the same policy that is also used to generate the actions, while off-policy methods improve a policy that is not the one that is used to generate the trajectories. The algorithm that is covered below is an on-policy algorithm. 

Algorithm
#########

Once again the algorithm follows a two step approach.

* In the first step the agent generates the trajectory using the :math:`\epsilon`-greedy policy.
* In the second step the agent iterates over the trajectory and uses Monte Carlo and incremental improvement to estimate the action-values. 


.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Monte Carlo Control: First Visit}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: environment $env$, state set $\mathcal{S}$, action set $\mathcal{A}$, number of episodes, learning rate $\alpha$, discount factor $\gamma$, epsilon $\epsilon$
        \STATE Initialize: 
        \STATE $Q(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$ with zeros 
        \STATE
        \STATE policy $\pi(a \mid s)$ for all $a \in \mathcal{A}$, where $A \sim \pi(. \mid s)$
        \STATE $r \leftarrow$ random number
        \IF {$r < \epsilon$}
            \STATE $A \leftarrow$ random action
        \ELSE
        \STATE $A \leftarrow \arg\max_aQ(a)$
        \ENDIF
        \STATE
        \FOR{$i=0$ to number of episodes}
            \STATE ...................................................................................................................................
            \STATE (1) INTERACTION WITH THE ENVIRONMENT
            \STATE create $trajectory$ as empty list [...]
            \REPEAT
                \STATE Generate experience $tuple$ $(State, Action, Reward)$ using policy $\pi$ and MDP $env$ 
                \STATE Push $tuple$ into $trajectory$
            \UNTIL{state is terminal}
            \STATE ...................................................................................................................................
            \STATE (2) ESTIMATION OF VALUE FUNCTION
            \STATE Create Visited(s) = $False$ for all $s \in \mathcal{S}$
            \FOR{$t=0$ to number of tuples in $trajectory$ $T$}
                \STATE $s \leftarrow$ state from $trajectory[t]$
                \STATE $a \leftarrow$ action from $trajectory[t]$
                \IF {Visited($s$) is True}
                    \STATE go to next tuple
                \ELSE
                    \STATE Visited($s$) = True 
                \ENDIF
                \STATE $G_{t:T} = \sum_{k=t}^T \gamma^{k-t}R_t$
                \STATE $Q(s, a) = Q(s, a) + \alpha [G_{t:T} - Q(s, a)]$
            \ENDFOR
        \ENDFOR
        \STATE
        \STATE RETURN policy, Q(s, a)
    \end{algorithmic}
    \end{algorithm}

Implementation
##############

The python implementation follows directly from the above described algorithm.

.. code::

    def mc_control(env, S, A, num_episodes=200000, alpha=0.001, gamma=0.99, epsilon=0.1):
    
        Q = np.zeros(shape=(len(S), len(A)))
        
        # generate policy
        def policy(state):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state].argmax()

            return action
        
        for episode in range(num_episodes):

            # generate episode
            episode = []
            state = env.reset()
            done = False

            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                experience = (state, action, reward)
                episode.append(experience)
                state = next_state

            # makes an array with nr. timesteps as rows and 3 columns
            episode = np.array(episode, dtype=np.int32)

            time_steps = len(episode)

            # discounts
            discounts = np.array([gamma**time_step for time_step in range(time_steps)])

            # update action-value function
            visited = np.zeros(shape=(len(S), len(A)), dtype=np.bool_)
            for time_step, (state, action, _) in enumerate(episode):
                
                if visited[state][action]:
                    continue
                else:
                    visited[state][action] = True
                
                returns = episode[time_step:, 2]
                remain_steps = len(returns)
                G = np.sum(returns * discounts[:remain_steps])
                
                Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])
    
        policy_mapping = np.argmax(Q, axis=1)
        policy = lambda x: policy_mapping[x]

        return policy, Q


Sources
=======

.. [#] https://en.wikipedia.org/wiki/Monte_Carlo_method