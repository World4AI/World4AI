<h1>Agent and Environment Components</h1>

<p>
    When the agent receives the state or the environment receives the action both use their respective internal components to transform the received data into a form that can be sent back to the originator of the data.
</p> 

<p>
    .. note::
    The components of the agent and the environment are basically mathematical functions that take input and generate output. Oftentimes the word mapping is used in that context. A function that takes x as input and outputs y is said to map x to y.
</p>     


<h2>Components of the Environment</h2>

<p>
    .. note::
    The model of the environment regulates the transition probabilities from the current state into the next state and the calculation of rewards using the current state and the action taken by the agent as input.
</p> 

<p>The environment has basically one single component called the model.</p> 

<p>
.. note::
   The process of changing the state of the environment is called transitioning (into a new state).
</p>

<p>How exactly the model looks depends on the environment.</p> 

<p>
    Sometimes a simple table is all that is required. For a gridworld with 25 possible states and  4 possible actions a table with 25 rows and 5 columns could be used to represent the model. The inner cells at the interaction between the current state and the action would have the probabilities to transition into the next state and the reward.
</p>

<p>
    More complex environments like the atari games would have their game engine and game logic that would calculate the transitions and rewards.  
</p>

<p>
    In reinforcement learning the model of the environment is usually not something that the agent has access to. The agent has to learn to navigate in an environment where the rules of the game are not known. 
</p>

<p>
    In most cases reinforcement learning practitioners do not deal with the creation of new environments. There are already hundreds of ready made environments that they can access. This reduces development speed and allows comparisons among different researchers and algorithms.
</p>

<h2>Components of the Agent</h2>

<p>
    The agent has up to three main components. The policy function, the value function and a model. Generally only the policy is actually required for the agent to work. Nevertheless, the model and the value function are major parts of many modern reinforcement learning algorithms. Especially the value function is often considered to be a necessary component.
</p>

<h3>Policy</h3>

<p>
    The policy has the purpose to calculate the action given the current state of the environment as the input.
</p> 

<p>
.. note::
   The policy of the agent maps states to actions.
</p>

<p>
    For very simple environments the policy function might also be a table that contains all possible states and for each state there is a corresponding action. In more complex environments it is not possible to construct a mapping table like the one above, as the number of states is extremely high. In that case other solutions like neural networks are used. 
</p>

<h3>Value Function</h3>

<p>
.. note::
    The value function of the agent maps states to values.
</p>
 
<p>
    The second component is the so-called value function. The value function gets a state as an input and generates a single scalar value. The higher the value, the better the state.
</p>

<p>
    The two images above show different states in the gridworld. In the first image the circle is in the bottom right corner. In the second image the circle is almost at the goal. Which of the two states is more preferable for the agent? Intuitively speaking the second one, as the agent is close to getting a positive reward and has already gotten all the negative rewards. To transform this intuition into actual numeric values the value function is used. 
</p>

<p>
    Similar to the policy for simple environments the value function can be calculated with the help of a table or in more complex environments using a neural network. 
</p>

<h3>Model</h3>

<p>
    .. note::
    The model of the agent is an approximation of the true model of the environment.
</p>

<p>
    The third and last component is the model. The model of the environment is something that the agent generally has no access to, but the agent can theoretically learn about the model by interacting with the environment. So essentially the agent creates some sort of an approximation of the true model of the environment. Each interaction allows the agent to improve his knowledge regarding the transition probabilities from one state to the next and the corresponding reward. The model can then for example be used to improve the policy. This is especially useful when interacting with the environment is for some reason costly. 
</p>