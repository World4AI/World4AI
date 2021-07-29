================
Policy Iteration
================

The policy iteration algorithm is an iterative method. Iterative methods start with initial (usually random or 0) values as approximations and improve the subsequent approximations with each iteration using the previous approximations as input. The policy iteration algorithm consists of two steps. The policy evaluation step calculates the value function for a given policy. The policy improvement step improves the given policy. Both steps run after each other to form the policy iteration algorithm. 

Policy Evaluation
=================

The goal of policy evaluation is to find the true value function :math:`v_{\pi}` of the policy :math:`\pi`. 

.. math::
    :nowrap:

    \begin{align*}
    v_{\pi}(s) & \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{align*}

The algorithm uses the mathematical definition of the value function and turns it into an iterative algorithm.

.. math::
    v_{k+1}(s) \doteq \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{k}(s')]


At each iteration the approximate value for each state is calculated using the old value from in the Bellman equation. The old value is then substituted by the new value. At this point it should become apparent why the Bellman equation is useful. Only the reward from the next time step is required to improve the approximation, as all subsequent rewards are already condensed into the value function from the next time step. That allows the algorithm to use the model to look only one step into the future for the reward and use the approximated value function for the next time step. By repeating the update step over and over again the rewards are getting embedded into the value function and the approximation gets better and better. It can be shown mathematically that if the update step is repeated an unlimited number of times, the approximate value function approaches the true value function of the policy. In practice the improvement is done as long as the value function between two iterations is large enough. 

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Iterative Policy Evaluation}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: policy $\mu$, model $p$, state set $\mathcal{S}$, stop criterion $\theta > 0$, discount factor $\gamma$
        \STATE Initialize: $V(s)$ and $V_{old}(s)$, for all $s \in \mathcal{S}$ with zeros
        \REPEAT
            \STATE $\Delta \leftarrow 0$
            \STATE $V_{old}(s) = V(s)$ for all $s \in \mathcal{S}$
            \FORALL{$s \in \mathcal{S}$}
                \STATE $V(s) \leftarrow \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma V_{old}(s')]$
                \STATE $\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)$
            \ENDFOR
        \UNTIL{$\Delta < \theta$}
    \end{algorithmic}
    \end{algorithm}


In order to calculate the value function the algorithm needs 4 inputs.

* The deterministic policy :math:`mu` is a function that gets a state as an input and generates an action as an output. We are not going to deal with stochastic agents yet, therefore :math:`\pi(a \mid s) = 1` in the update step. :math:`a = \mu (s)` 
* The model p will take the current state and action as input and return the next state, the next reward, the terminal flag and the corresponding probability. :math:`probability, next state, reward, terminal = p(s, a)` 
* :math:`\mathcal{S}` is the state set of the MDP
* Theta :math:`\theta`  is the criterion to stop the algorithm once the difference between the old value function :math:`V_{old}` and new value function :math:`V` is less than :math:`\theta`
* Gamma :math:`\gamma` is the discount factor for the update step in the Bellman equation

We are going to keep two versions of the value function, :math:`V_{old}` and :math:`V`. During an update iteration we move through the states one at a time and it is more intuitive in my opinion to update all the states with old values before updating the value function as a whole. If we used only one value function we would update some of the states with the already updated values and some with old values. Both versions of updates are valid, but the “inplace” version is not going to be used in this chapter. 

The imports consist only of Gym and NumPy. Gym for the MDP and NumPy to make calculations more efficient. 

.. code:: python

    import gym
    import numpy as np

We are going to calculate the value function for the Frozen Lake environment, but the algorithm is general and can be applied to many different MDPs.

.. code:: python

    env = gym.make('FrozenLake-v0')


The policy function contains internally a mapping table that maps states to actions deterministically. 

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

The model returns the list of possible next states, rewards and corresponding probabilities given the current state and action.

.. code::

    def model(state, action):
        return env.P[state][action]


.. code::

    S = [x for x in range(env.observation_space.n)]

The below code covers the actual policy evaluation algorithm. 

.. code::

    def policy_evaluation(policy, model, S, theta=0.00001, gamma=0.99):
        # initialize value functions with zeros
        V = np.zeros(len(S))
        V_old = np.zeros(len(S))
        
        while True:
            delta = 0
            V_old = V.copy()
            for state in S:
                # we avoid the loop over the actions as the policy is deterministic
                action = policy(state)
        
                value = 0
                for prob, next_state, reward, done in model(state, action):
                    value += prob * (reward + gamma * V_old[next_state] * (not done))
                V[state] = value
                
            # check for stop criterion and break if necessary
            max_diff = np.max(np.abs(V - V_old))
            if max_diff < theta:
                break
                
        return V

.. code::

    policy_evaluation(policy, model, S)

Policy Improvement
==================


