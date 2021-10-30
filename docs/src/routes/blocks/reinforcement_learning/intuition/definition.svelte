<script>
    import { RandomAgent} from '$lib/reinforcement_learning/common/RandomAgent';
    import Grid from '$lib/reinforcement_learning/intuition/applications/Grid.svelte';
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | Definition</title>
    <meta name="description" content="Reinforcement learning is defined as learning through trial and error and delayed rewards.">
</svelte:head>

<h1>Definition Of Reinforcement Learning</h1>
<h2>What are the key characteristics of reinforcement learning?</h2>

<p>There are probably dozens of formal definitions of reinforcement learning. These definitions do not necessarily contradict each other, but rather explain something similar when we look a little deeper at what the definitions are trying to convey. In this section we are going to look at the one definition that should capture the essence of reinforcement learning in a very clear way.</p>

<p class="info">Reinforcement Learning is Learning through Trial and Error and Delayed Rewards.</p>

<p>The definition consists of three distinct parts: <strong>Learning</strong>, <strong>Trial and Error</strong> and <strong>Delayed Rewards</strong>. In order to understand the complete definition we will deconstruct the sentence and look at each part individually.</p> 

<h3>Learning</h3>

<p>Learning is probably the most obvious part of the definition. Usually in reinforcement learning when the agent starts to interact with the environment the agent does not know anything about that environment. The assumption in reinforcement learning that is always made is that the environment the agent interacts with contains some goal that the agent has to achieve.</p> 

<div class="flex-center">
    <Grid agentClass={RandomAgent}/>
</div>

<p>For example the agent is expected to move the circle from the starting cell position (top left corner) to the goal cell position (bottom left corner).</p>

<p>When we talk about learning, that means that the agent gets better at achieving that particular goal over time. It could start by moving in a random fashion and over time learn the best possible (meaning the shortest) route.</p> 

<p class="info"><strong>Learning</strong> means that the agent gets better at achieving the goal of the environment over time.</p>

<h3>Rewards</h3>

<p>The question still remains how exactly does the agent know what the goal of the environment actually is? The environment with which the agent interacts gives feedback about the behaviour of the agent by giving out a reward after each single step that the agent takes.</p>

<p>If the goal of the grid world is to move the circle to the cell with the triangle as fast as possible the environment could for example give a positive reward for getting to the cell with the triangle and punish the agent in any other case.</p>

<p>If the agent takes the direct route to the triangle it will get less negative rewards while an indirect route creates a lot of negative rewards. The agent needs to learn through the reward feedback that some sequences of actions are better than others.</p>

<p class="info">In reinforcement learning the agent learns to maximize <strong>rewards</strong>. The goal of the environment is therefore implicitly contained in the rewards.</p>

<h3>Trial and Error</h3>

<p>The problem with the rewards is that it is not clear from the very beginning what path produces the highest possible sum of rewards. In reinforcement learning there is only the reward signal and even if the agent receives a positive sum of rewards it never knows if it could have done better. Unlike in supervised learning, there is no teacher/supervisor to tell the agent what the best behaviour is. So how can the agent figure out what sequence of actions produces the highest sum of rewards? The only way it can, by trial and error.</p>

<p>The agent has to try out different behaviour to figure out which one produces optimal results. How long it takes the agent to find a good sequence of decisions depends on the complexity of the environment and the employed learning algorithm. It can be anything between a couple of seconds to many days. In some cases the agent can not solve an environment no matter how hard it tries.</p>

<p class="info">In the context of reinforcement learning, trial and error means trying out different sequences of decisions and comparing the resulting sum of rewards.</p>

<h3>Delayed</h3>

<p>In reinforcement learning the agent often needs to take dozens or even thousands of steps before a particular reward is achieved. In that case there has been a succession of many steps and the agent has to decide which step and in which proportion is responsible for the reward, so that the agent could select the decisions that lead to a good sequence of rewards more often.</p>

<p>Which of the steps is responsible for the positive reward? Is it the action just prior to the reward? Or the one before that? Or the one before that? Reinforcement Learning has no easy answer to the question which decision gets the credit for the reward. This problem is called <strong>the credit assignment problem</strong>.</p> 

<p class="info">In reinforcement learning rewards for an action are often delayed, which leads to the credit assignment problem.</p> 

<h3>Notes</h3>

<p>
.. [#] This definition is highly inspired by the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto.
.. [#] In reinforcement learning we do not actually differentiate between a reward and a punishment. We call it reward no matter if the reward is positive, negative or zero. 
</p>

<style>
    .flex-center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .info {
        position: relative;
    }

    .info::before {
        content: url(/icons/info-outline.svg);
        left: -70px;
        top: 0px;
        position: absolute;
        height: 40px;
        width: 40px;
    }
</style>