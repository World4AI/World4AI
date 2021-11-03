<svelte:head>
    <title>World4AI | Reinforcement Learning | Value, Policy, Model</title>
    <meta name="description" content="In reinforcement learning the value function, the policy and the model are essential components of agent. The environment has only one component, the model.">
</svelte:head>

<h1>Value, Policy, Model</h1>

<p><em>What are the respective components of the agent and the environment?</em></p>

<div class="separator"></div>

<p>The interaction between the agent and the environment involves processing the input data and transforming it into the output that can be sent back. For example when the agent receives the current state of the environment, it needs to transform that state data into the action and when the environment receives the action it needs to transorm that action and the current state into the new state and the corresponding reward.</p>

<p>For that purpose the agent and the environment utilize their internal components. The environment uses the <strong>model</strong>, while the agent might use up to three components. The <strong>value function</strong>, the <strong>policy</strong> and the <strong>model</strong>.</p>

<p>These components are basically mathematical functions that take inputs and generate outputs. Oftentimes the word mapping is used in that context. A function that takes x as input and outputs y is said to map x to y.</p>

<p class="info">The environment has one component called model, while the agent might have a value function, a policy and a model as components.</p>


<div class="separator"></div>

<h2>Model of the Environment</h2>

<p>The model (sometimes called the dynamics) is the only component of the environment, but it is often divided into two separate parts for convenience.</p>

<p>The first part is responsible for calculating the next state based on the current state of the environment and the action chosen by the agent. This subcomponent contains the transition probabilities of the model and it is responsible for transitioning into the next state.</p>

<p class="info">The process of changing the state of the environment is called transitioning (into a new state).</p>

<p>The second part is responsible for calculating the reward based on the current state and the action.</p>

<p class="info">The model consists of the transition function and the reward function.</p>
    
<p>How exactly the model looks depends on the environment. Sometimes a simple table is all that is required. For a gridworld with 25 possible states and  4 possible actions a table with 25 rows and 5 columns could be used to represent the model. The inner cells at the interaction between the current state and the action would have the probabilities to transition into the next state and the reward.</p>

<p>More complex environments like the atari games would have their game engine and game logic that would calculate the transitions and rewards.</p>

<p>In reinforcement learning the model of the environment is usually not something that the agent has access to. The agent has to learn to navigate in an environment where the rules of the game are not known.</p>

<p>In most cases reinforcement learning practitioners do not deal with the creation of new environments. There are already hundreds of ready made environments that they can access. This reduces development speed and allows comparisons among different researchers and algorithms.</p>

<div class="separator"></div>

<h2>Components of the Agent</h2>

<p>The agent has up to three main components. The policy function, the value function and a model. Generally only the policy is actually required for the agent to work. Nevertheless, the model and the value function are major parts of many modern reinforcement learning algorithms. Especially the value function is often considered to be a necessary component of a successful agent.</p>

<div class="separator"></div>

<h3>Policy</h3>

<p>The first component is the policy. The policy calculates the action directly based the current state of the environment.</p> 

<p class="info">The policy of the agent maps states to actions.</p>

<p>For very simple environments the policy function might also be a table that contains all possible states and for each state there is a corresponding action. In more complex environments it is not possible to construct a mapping table like the one above, as the number of states is extremely high. In that case other solutions like neural networks are used. </p>

<div class="separator"></div>

<h3>Value Function</h3>
 
<p>The second component is the so-called value function. The value function gets a state as an input and generates a single scalar value. The higher the value, the better the state.</p>

<p class="info">The value function of the agent maps states to values.</p>

<p>The two images above show different states in the gridworld. In the first image the circle is in the bottom right corner. In the second image the circle is almost at the goal. Which of the two states is more preferable for the agent? Intuitively speaking the second one, as the agent is close to getting a positive reward and has already gotten all the negative rewards. To transform this intuition into actual numeric values the value function is used.</p>

<p>Similar to the policy for simple environments the value function can be calculated with the help of a table or in more complex environments using a neural network.</p>

<div class="separator"></div>

<h3>Model</h3>

<p>The third and last component is the model. The model of the environment is something that the agent generally has no access to, but the agent can theoretically learn about the model by interacting with the environment.  Essentially the agent creates some sort of an approximation of the true model of the environment. Each interaction allows the agent to improve his knowledge regarding the transition probabilities from one state to the next and the corresponding rewards. The model can for example be used to improve the policy. This is especially useful when interacting with the environment is for some reason costly. Additionally the model can connect to the policy in generate better actions.</p>

<p class="info">The model of the agent is an approximation of the true model of the environment.</p>

<div class="separator"></div>
