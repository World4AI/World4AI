============
Introduction
============

So far we dealt with tabular solution methods for finite MDPs. 

.. list-table:: Q-Table
    :widths: 25 25 25 25 25
    :header-rows: 1

    * - State
      - Value Action 1
      - Value Action 2 
      - Value Action 3
      - Value Action 4
    * - 0
      - 1
      - 2
      - 1
      - 1 
    * - 1
      - 1
      - 2
      - 3
      - 1 
    * - 2
      - 1
      - 2
      - 3
      - 2 
    * - 3
      - 2
      - 2
      - 3
      - 4 

The number of rows (4 states) and columns (4 actions) in the Q-Table was finite. This allowed us to loop over all states and actions and use Monte Carlo or TD methods. Given enough iterations we were guaranteed to find the optimal value functions and thus policies.

Most interesting reinforcement learning problems do not have such nice properties. In case state or action sets are infinite or extremely large it becomes impossible to store the value function as a table. 

.. list-table:: Infinite Q-Table
    :widths: 25 25 25 25 25
    :header-rows: 1

    * - State
      - Value Action 1
      - Value Action 2 
      - Value Action 3
      - Value Action 4
    * - 0
      - 1
      - 2
      - 1
      - 1 
    * - 1
      - 1
      - 2
      - 3
      - 1 
    * - 2
      - 1
      - 2
      - 3
      - 2 
    * - 3
      - 2
      - 2
      - 3
      - 4 
    * - .
      - .
      - .
      - .
      - .
    * - 1,000,000,000
      - 2
      - 1
      - 2
      - 3

The above table shows action-values for 1,000,000,000 discrete states. Even if we possessed a computer which could efficiently store a high amount of states, we still need to loop over all these states and thus convergence would be extremely slow. 

Q-Tables become almost impossible when the agent has to deal with continuous variables. When the agent sees a continuous observation for example there is a chance that the same exact value will not be seen again. Yet the agent will need to learn how to deal with future unseen observations. We expect from the agent to find a policy that is “good”  across many different observations. The key word that we are looking for is generalization. 

.. list-table:: Q-Table
    :widths: 25 25 25
    :header-rows: 1

    * - State Representation
      - Action Value 1
      - Action Value 2
    * - 1.1
      - 1
      - 2
    * - 1.3
      - 1.2
      - 1.8
    * - 1.5
      - 1.5
      - 1.2
    * - **1.7**
      - **?**
      - **?**
    * - 2.1
      - 1.8
      - 1.3
    * - 2.2
      - 1.7
      - 1.8
    * - 2.5
      - 1.5
      - 2


The example above shows how generalization might look like. The state is represented by 1 single continuous variable and there are only 2 discrete actions available (left and right). If you look at the state representation with the value 1.7 could you approximate the action-values and determine which action has the higher value? You will probably not get the exact correct answer but the value for the left action should probably be somewhere between 1.5 and 1.8 and therefore larger than the value between 1.2 and 1.3. Real reinforcement learning tasks might be a lot more complex, but I think it is a good mental model to imagine the agent interpolating between the states he has already seen and learned from. 

In the case when the state/action sets are extremely large and/or continous, lookup tables for each state-action pair become impossible. As the number of states and/or actions is possibly infinite, the function has to generalize and we can assume that it is impossible to create a function that generates optimal values for each state-action pair.  Thus it becomes necessary to create value functions that are not exact, but approximative, meaning that the value function does not return the true value of a policy but a value that is hopefully close enough. Finding the optimal policy and value function is often not possible, but the general idea is to find a policy that still performs in a way that generates a relatively high expected sum of rewards. 

We can look at it this way. Humans most definitely don’t have optimal policies for complex tasks they have to perform. If they did, then chess for example would be extremely boring and would not have survived for many hundred years. Still you can appreciate the complexity of the game and the extremely high level of professional players. We are going to have similar expectations for our agents. Even though we will not be able to find optimal value functions for some of the environments we are going to generate extremely impressive results.
