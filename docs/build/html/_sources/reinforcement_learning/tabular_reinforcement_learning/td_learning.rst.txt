============================
Temporal Difference Learning
============================

TD Prediction
=============

.. math::
    :nowrap:

    \begin{align*}
    & V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] \\
    & V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
    \end{align*}


.. code:: python

    def td_prediction(env, policy, S, num_episodes=100000, alpha=0.01, gamma=0.99):
    
        V = np.zeros(len(S))
        
        for episode in range(num_episodes):
            
            state = env.reset()
            done = False
            
            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
                state = next_state
        
        return V


SARSA
=====

.. math:: 

    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]


.. code:: python

    def sarsa(env, S, A, num_episodes=100000, alpha=0.01, gamma=0.99, epsilon=0.1):
    
        Q = np.zeros(shape=(len(S), len(A)))
        
        # an epsilon greedy policy
        def policy(state):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state].argmax()
            return action
        
        for episode in range(num_episodes):
            state = env.reset()
            action = policy(state)
            done = False
            
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = policy(next_state)
                Q[state, action] += alpha * (reward + Q[next_state, next_action] * (not done) - Q[state, action])
                state, action = next_state, next_action
        
        policy_mapping = np.argmax(Q, axis=1)
        policy = lambda x: policy_mapping[x]

        return policy, Q

Q-Learning
==========

.. math:: 

    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]


.. code:: python

    def q_learning(env, S, A, num_episodes=100000, alpha=0.01, gamma=0.99, epsilon=0.1):
    
        Q = np.zeros(shape=(len(S), len(A)))
        
        # an epsilon greedy policy
        def policy(state):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state].argmax()
            return action
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = policy(state)
                next_state, reward, done, _ = env.step(action)
                Q[state, action] += alpha * (reward + Q[next_state].max() * (not done) - Q[state, action])
                state = next_state
        
        policy_mapping = np.argmax(Q, axis=1)
        policy = lambda x: policy_mapping[x]

        return policy, Q