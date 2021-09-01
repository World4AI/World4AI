=================
Double Q-Learning
=================

Motivation
==========

.. note::
    "In some stochastic environments the well-known reinforcement learning algorithm Q-learning performs very poorly. This poor performance is caused by large overestimations of action values. These overestimations result from a positive bias that is introduced because Q-learning uses the maximum action value as an approximation for the maximum expected action value" [#]_.

The update rule in Q-learning involves a maximization operation. This can lead to what is known as maximization bias.

.. math:: 

    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/double_q_learning/maximization_bias.svg
   :align: center

   Maximization Bias.

The above example of a Q-Table exemplifies the problem. We are making the assumption that the true action value of all actions for state 1 is 0. Yet the samples that were collected so far produced action values that are slightly above or below 0. If the next state :math:`S_{t+1}` corresponds to state 1 the maximization operation will pick the action-value of action 8. If the true action-values all correspond to 0 for state 1 then it should not matter for the agent which action should be picked next. Yet due to estimation errors the agent will select the action with the maximum estimated value. This is called maximization bias. Double Q-Learning aims to alleviate the problem.

Generalized Policy Iteration
============================

Theory
------

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/double_q_learning/two_value_functions.svg
   :align: center

   Two Value Functions.

To reduce maximization bias double Q-Learning introduces a second action value function. 

.. figure:: ../../_static/images/reinforcement_learning/tabular_rl/double_q_learning/double_q.svg
   :align: center

   Double Q-Learning.

Even if one of the functions has a bias towards one of the actions, it is unlikely that the second function will have the same exact bias. In the above example the first action value function prefers the action number 8 while the second prefers the very first action. Combining both Q-functions should make Q-Learning converge faster.

The below update step updates one of the Q-functions while using the other function in the calculation of the target value. Which one of the functions is treated as function 1 and which one as function 2 is decided using a random variable. Usually the probability corresponds to a fair coin toss.

.. math:: 

    Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_aQ_1(S_{t+1}, a)) - Q_1(S_t, A_t)]

Algorithm
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Double Q-Learning}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: environment $env$, state set $\mathcal{S}$, action set $\mathcal{A}$, number of episodes, learning rate $\alpha$, discount factor $\gamma$
        \STATE Initialize: 
        \STATE $Q_1(s, a)$ and $Q_2(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$ with zeros
        \STATE $\epsilon$-greedy policy $\pi(a \mid s)$ with $Q=Q_1+Q_2$ for all $a \in \mathcal{A}$, where $A \sim \pi(. \mid s)$
        \STATE
        \FOR{$i=0$ to number of episodes}
            \STATE Reset state $S$
            \REPEAT
                \STATE Generate experience $tuple$ $(A,R,S')$ using policy $\pi$ and MDP $env$
                \STATE EITHER 
                \STATE $Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_2(S_{t+1}, \arg\max_aQ_1(S_{t+1}, a)) - Q_1(S_t, A_t)]$
                \STATE OR
                \STATE $Q_2(S_t, A_t) \leftarrow Q_2(S_t, A_t) + \alpha [R_{t+1} + \gamma Q_1(S_{t+1}, \arg\max_aQ_2(S_{t+1}, a)) - Q_2(S_t, A_t)]$
                \STATE $S \leftarrow S'$
            \UNTIL{state is terminal}
        \ENDFOR
        \STATE
        \STATE RETURN policy, Q(s,a)
    \end{algorithmic}
    \end{algorithm}


Implementation
--------------

.. code:: python

    def double_q_learning(env, S, A, num_episodes=100000, alpha=0.01, gamma=0.99, epsilon=0.1):
        Q_a = np.zeros(shape=(len(S), len(A)))
        Q_b = np.zeros(shape=(len(S), len(A)))
        
        # an epsilon greedy policy
        def policy(state):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                Q = Q_a + Q_b
                action = Q[state].argmax()
            return action
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                
                # decide which Q function to update and which to use for target
                if np.random.rand() < 0.5:
                    Q_1 = Q_a
                    Q_2 = Q_b
                else:
                    Q_1 = Q_b
                    Q_2 = Q_a
                
                next_action = Q_1[state].argmax()
                Q_1[state, action] += alpha * (reward + Q_2[next_state][next_action] * (not done) - Q_1[state, action])
                state = next_state
        
        Q = Q_a + Q_b
        policy_mapping = np.argmax(Q, axis=1)
        policy = lambda x: policy_mapping[x]

        return policy, Q

Sources
=======
.. [#] van Hasselt, H. (2010). Double Q-learning. In Advances in Neural Information Processing Systems 23, pp. 2613–2621. Curran Associates, Inc.