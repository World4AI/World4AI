===================
Monte Carlo Methods
===================

Motivation
==========

Monte Carlo methods are similar in spirit to bandit methods, as in both the action-value function can be estimated through continuous trial and error and averaging. Unlike in bandits though, monte carlo methods are able to deal with problems where decisions are based on different states. 

In reinforcement learning the goal of the agent is to maximize the expected sum of discounted rewards.  

:math:`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`

:math:`q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]`

In dynamic programming we were allowed to access the model, which allowed us to use the update rule for value functions with the knowledge of transition probabilities and expected rewards. 

In Reinforcement Learning on the other hand the agent has no access to the model and needs to learn the expected value through other means. The most straightforward way to accomplish that is to generate discounted returns through continuous interaction. The estimates of value functions can be calculated by averaging over the generated returns. 

Just as with dynamic programming the algorithm is going to be based on general policy iteration, which means that policy estimation and policy improvement will succeed each other. 

Policy Estimation
=================

Let us recall once again that the idea of policy estimation is to find the state-value and action-value functions for a corresponding policy.

:math:`v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]`

:math:`q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]`


A natural way to estimate the expected value of a random variable is to sample random variables and to use the average as an estimate. Therefore using a policy the agent can interact with the environment and generate full episodes. The returns can be used to build averages to estimate state-value and action-value functions. For each state or state-action pair we could keep a list or a table respectively where the estimate is stored and updated. 

Generally there are two methods to calculate the averages. Each time the agent faces a state (or state-action pair) during an episode is called a visit. In the “First Visit” monte carlo method only the return from the first visit to that state (state-action pair) until the end of the episode is calculated. If the state (state-action) is visited several times during an episode, the additional visits are not considered in the calculation. While in the “Every Visit” method each visit is counted. The “First Visit” method is more popular and is going to be covered in this section, but the algorithms can be easily adjusted to account for the “Every Visit” method.


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

.. code:: python

    import gym
    import numpy as np

.. code:: python

    env = gym.make('FrozenLake-v0')


.. code:: python

    S = [x for x in range(env.observation_space.n)]
    A = [x for x in range(env.action_space.n)]

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

Policy Improvement and Control
==============================

To find the optimal policy with a Monte Carlo method we are going to use general policy iteration by applying policy evaluation and policy improvement back-to-back. The approach is going to be similar to value iteration, as only one step of policy evaluation is going to be used. 


.. math::
    :nowrap:

    \begin{align*}
    v_{k+1}(s) & \doteq \max_a \mathbb{E}[R_{t+1} + \gamma v_k (S_{t+1}) \mid S_t = s, A_t = a] \\
    & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
    \end{align*}


The approach from value iteration where we estimated and improved the state-value function is not going to work with Monte Carlo methods, as those steps required the knowledge of the model. 

But what if we rewrite the above improvement step as follows?

.. math::

    v_{k+1}(s) \doteq \max_a q_k(s, a) \\
    \mu_{k+1}(s) \doteq \arg\max_a q_k(s, a)

With those rewrites we do not require the knowledge of the model, but it becomes obvious that the key is to estimate the action-value function and not the state-value function. Having an estimate of an action-value function allows the agent to select better actions and to gradually improve the policy towards the optimal policy.  In the first step we need to estimate the action-value function by calculating averages. In the second step we can calculate a new greedy policy by taking the argmax over q(s,a). 

There is still one problem that we face without the knowledge of the model of the MDP though. If our policy is fully deterministic and thus avoids some state-action pairs by design, then we can not compute a good estimate for certain state-action pairs and thus might not arrive at the optimal policy. The solution is as with bandits is to use an :math:`\epsilon`-greedy policy, meaning that with a probability of :math:`\epsilon` we take a random action and with probability of :math:`1-\epsilon` we take the greedy action. That way we are guaranteed that all state-action pairs are going to be visited.
    

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