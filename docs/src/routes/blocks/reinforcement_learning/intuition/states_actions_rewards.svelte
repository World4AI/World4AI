<script>

import { action_destroyer } from "svelte/internal";

</script>
<svelte:head>
    <title>World4AI | Reinforcement Learning | States, Actions, Rewards</title>
    <meta name="description" content="In Reinforcement learning states, actions and rewards are send between the agent and the environment.">
</svelte:head>

<h1>States, Actions, Rewards</h1>
<h2>How does the interaction in reinforcement learning look like?</h2>

<h3>Interaction</h3>

<p>The agent and the environment interact continuously, each reacting to the data sent by the other. </p>
<p>In reinforcement learning the sequential information flow between the agent and the environment is called <strong>interaction</strong>.</p>

<p>In essence interaction means that there is a communication channel between the agent and the environment, where data flows sequentially between the two.</p>

<p>In order for the data to be communicated there have to be three consecutive steps.</p>

<ol>
    <li>The agent / environment receives the data</li>     
    <li>The data is processed by the receiver</li>
    <li>The processed data is sent back to the originator</li>
</ol>

<p>What is important to mention is that this stream of data is sent in a strictly sequential way. When the environment sends the data for example, it has to wait until it receives the response from the agent. Only then a new batch of data can be sent to the agent again.</p>  

<p>The middle part of the three steps, the processing of the data by the **agent** is the main part of reinforcement learning. The processing will be covered in detail in later chapters. In this chapter I will only describe the types of data.</p>

<p>In general in reinforcement learning there are just 3 types of data that need to be sent between the agent and the environment: **state** data, **action** data and **reward** data.</p>

<p>The agent receives the current state of the environment and based on the state sends the action he would like to take. The environment sends the reward for that action and transitions into a new state taking the action of the agent and the current state into account. The agent then uses the reward to learn to make decisions based on states.</p>

<h3>Math Sidenote</h3>

<p>
    Following are some mathematical definitions to be able to follow the following section.

    .. note:: 
    Discrete variables are integers like :math:`1, 2, 3, 42`

    Continuous Variables are floating point numbers like :math:`3.14, 22.2, 14.992`
</p>

<h3>State</h3>

<p>.. note::The state is the representation of the current condition of the environment.</p>
<p>The state describes how the environment actually looks like. It is the current situation the agent faces and based on the state the agent has to make his decisions. The state can be represented by a scalar, a vector, a matrix or a tensor and can be either discrete or continuous.</p>

<p>In this simple gridworld example all the agent needs to know to make the decisions is the location of the circle in the environment. So in the starting position the state would be 1. The one to the right 2 and so on. Based on the position the agent can choose the path towards the triangle.</p>

<p>There are of course more complex environments where the state is for example represented by a tensor containing rgb values.</p>

<h3>Action</h3>

<p>.. note::The action is the representation of the decision/behaviour of the agent.</p>

<p>The action is the behaviour the agent chooses based on the state of the environment. [#]_ Like the state the action can be a scalar, a vector, a matrix or a tensor of discrete or continuous values.</p>

<p>In the above gridworld example the agent can move north, east, south and west. Each action is encoded by a discrete scalar value.</p>

<ul>
    <li>North = 1</li>
    <li>East = 2</li>
    <li>South = 3</li>
    <li>West = 4</li>
</ul>

<p>One of the above scalar values is sent back to the environment.</p>
 
<h3>Reward</h3>

<p>
.. note::
   The reward is the signal to reinforce certain behaviour of the agent to achieve the goal of the environment.
</p> 

<p>The reward is what the agent receives from the environment for an action. It is the value that the environment uses to reinforce a behaviour to solve an environment and it is the value that the agent uses to improve his behaviour.</p>

<p>Unlike the action or the state the reward has to be a scalar, one single number, it is not possible for the reward to be a vector, matrix or tensor. As expected larger numbers represent larger or better rewards so that the reward of 1 is higher than the reward of -1.</p>

<p>In this gridworld example the agent receives a reward of -1 for each step taken with the exception of taking a step towards the triangle, where the agent receives a reward of 1.</p> 
   
<h3>Timestep</h3>

<p>.. note:: In reinforcement learning each iteration of exchanging state+reward and action is called a timestep.</p>

<p>Reinforcement learning works in (mostly) discrete timesteps. Each iteration where the environment and the agent each have sent their data constitutes a timestep.</p>

<h3>Notes</h3>

<p>
    .. [#] RGB stands for red, green, blue and is a common way to represent images. 
    .. [#] Theoretically the agent can make random decisions, but to maximize the sum of rewards agents should base decisions on the state.
</p>