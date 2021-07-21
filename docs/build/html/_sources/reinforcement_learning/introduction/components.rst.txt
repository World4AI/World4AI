================================
Agent and Environment Components
================================


.. figure:: ../../_static/images/reinforcement_learning/introduction/components/components.svg
   :align: center
   
   Components of the agent and the environment

When the agent receives the state or the environment receives the action both use their respective internal components to transform the received data into a form that can be sent back to the originator of the data. 

.. figure:: ../../_static/images/reinforcement_learning/introduction/components/function.svg
   :align: center
   
   A mathematical function 

.. note::
   The components of the agent and the environment are basically mathematical functions that take input and generate output. Oftentimes the word mapping is used in that context. A function that takes x as input and outputs y is said to map x to y.     


Components of the Environment
=============================

.. note::
   The model of the environment regulates the transition probabilities from the current state into the next state and the calculation of rewards using the current state and the action taken by the agent as input. 
    
The environment has basically one single component called the model. 

.. figure:: ../../_static/images/reinforcement_learning/introduction/components/model.svg
   :align: center
   
   The model function of the environment

.. note::
   The process of changing the state of the environment is called transitioning (into a new state).

How exactly the model looks depends on the environment. 

.. figure:: ../../_static/images/reinforcement_learning/introduction/components/grid_state.svg
   :align: center
   
   A gridworld with the correspondig state values

.. list-table:: Model of a deterministic gridworld
   :header-rows: 1
   
   * - Current State 
     - Action North (1)
     - Action East (2) 
     - Action South (3)
     - Action West (4)
   * - 1
     - Probability 100%, Next State 6, Reward -1
     - Probability 100%, Next State 2, Reward -1
     - Probability 100%, Next State 1, Reward -1
     - Probability 100%, Next State 1, Reward -1
   * - 2
     - Probability 100%, Next State 7, Reward -1
     - Probability 100%, Next State 3, Reward -1
     - Probability 100%, Next State 2, Reward -1
     - Probability 100%, Next State 1, Reward -1
   * - 
     - 
     - 
     - 
     -
   * - 25
     - Probability 100%, Next State 25, Reward -1
     - Probability 100%, Next State 25, Reward -1
     - Probability 100%, Next State 20, Reward -1
     - Probability 100%, Next State 24, Reward -1

Sometimes a simple table is all that is required. For a gridworld with 25 possible states and  4 possible actions a table with 25 rows and 5 columns could be used to represent the model. The inner cells at the interaction between the current state and the action would have the probabilities to transition into the next state and the reward.

More complex environments like the atari games would have their game engine and game logic that would calculate the transitions and rewards.  

In reinforcement learning the model of the environment is usually not something that the agent has access to. The agent has to learn to navigate in an environment where the rules of the game are not known. 

In most cases reinforcement learning practitioners do not deal with the creation of new environments. There are already hundreds of ready made environments that they can access. This reduces development speed and allows comparisons among different researchers and algorithms.


Components of the Agent
=======================

The agent has up to three main components. The policy function, the value function and a model. Generally only the policy is actually required for the agent to work. Nevertheless, the model and the value function are major parts of many modern reinforcement learning algorithms. Especially the value function is often considered to be a necessary component.  

Policy
------

.. note::
   The policy of the agent maps states to actions.

.. figure:: ../../_static/images/reinforcement_learning/introduction/components/policy.svg
   :align: center
   
   The policy function of the environment

The policy has the purpose to calculate the action given the current state of the environment as the input. 

.. list-table:: Policy in a deterministic gridworld
   :header-rows: 1
   
   * - Current State 
     - Action
   * - 1
     - East (2)
   * - 2
     - North (1)
   * - 
     - 
   * - 25
     - West (4)

For very simple environments the policy function might also be a table that contains all possible states and for each state there is a corresponding action. In more complex environments it is not possible to construct a mapping table like the one above, as the number of states is extremely high. In that case other solutions like neural networks are used. 

Value Function
--------------

.. note::
    The value function of the agent maps states to values.
 
.. figure:: ../../_static/images/reinforcement_learning/introduction/components/value.svg
   :align: center
    
   The value function of the environment

The second component is the so-called value function. The value function gets a state as an input and generates a single scalar value. The higher the value, the better the state.


.. figure:: ../../_static/images/reinforcement_learning/introduction/components/bad_state.svg
   :align: center
    
   A relatively bad state

.. figure:: ../../_static/images/reinforcement_learning/introduction/components/good_state.svg
   :align: center
    
   A relatively good state


The two images above show different states in the gridworld. In the first image the circle is in the bottom right corner. In the second image the circle is almost at the goal. Which of the two states is more preferable for the agent? Intuitively speaking the second one, as the agent is close to getting a positive reward and has already gotten all the negative rewards. To transform this intuition into actual numeric values the value function is used. 

.. list-table:: A thought up value function in a deterministic gridworld
   :header-rows: 1

   * - Current State 
     - Value
   * - 1
     - 1
   * - 2
     - 1.5
   * - 
     - 
   * - 25
     - 5

Similar to the policy for simple environments the value function can be calculated with the help of a table or in more complex environments using a neural network. 


Model
-----

.. note::
   The model of the agent is an approximation of the true model of the environment.

The third and last component is the model. The model of the environment is something that the agent generally has no access to, but the agent can theoretically learn about the model by interacting with the environment. So essentially the agent creates some sort of an approximation of the true model of the environment. Each interaction allows the agent to improve his knowledge regarding the transition probabilities from one state to the next and the corresponding reward. The model can then for example be used to improve the policy. This is especially useful when interacting with the environment is for some reason costly. 
