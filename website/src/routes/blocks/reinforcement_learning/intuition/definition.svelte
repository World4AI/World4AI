<script>
    import Question from '$lib/Question.svelte';
    import { RandomAgent } from '$lib/reinforcement_learning/common/RandomAgent';
    import { DeterministicAgent } from '$lib/reinforcement_learning/common/DeterministicAgent';
    import { GridEnvironment } from '$lib/reinforcement_learning/common/GridEnvironment';
    import { gridMap } from '$lib/reinforcement_learning/common/maps';
    import Grid from '$lib/reinforcement_learning/intuition/applications/Grid.svelte';
    import Sequence from '$lib/reinforcement_learning/intuition/definition/Sequence.svelte';
    import CreditAssignment from '$lib/reinforcement_learning/intuition/definition/CreditAssignment.svelte';

    let env_1 = new GridEnvironment(gridMap);
    let agent_1 = new RandomAgent(env_1.getObservationSpace(), env_1.getActionSpace());

    let env_2 = new GridEnvironment(gridMap);
    let agent_2 = new DeterministicAgent(env_2.getObservationSpace(), env_2.getActionSpace());

    let env_3 = new GridEnvironment(gridMap);
    let agent_3 = new DeterministicAgent(env_3.getObservationSpace(), env_3.getActionSpace());
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | Definition</title>
    <meta name="description" content="Reinforcement learning is defined as learning through trial and error and delayed rewards.">
</svelte:head>

<h1>Definition Of Reinforcement Learning</h1>
<Question>What are the key characteristics of reinforcement learning?</Question>
<div class="separator"></div>

<p>There are probably dozens of formal definitions of reinforcement learning. These definitions do not necessarily contradict each other, but rather explain something similar when we look a little deeper at what the definitions are trying to convey. In this section we are going to look at the one definition that should capture the essence of reinforcement learning in a very clear way.</p>
<p class="info">Reinforcement Learning is Learning through Trial and Error and Delayed Rewards <sup>[1]</sup>.</p>
<p>The definition consists of three distinct parts: <strong>Learning</strong>, <strong>Trial and Error</strong> and <strong>Delayed Rewards</strong>. In order to understand the complete definition we will deconstruct the sentence and look at each part individually.</p> 
<div class="separator"></div>

<h2>Learning</h2>
<p>Learning is probably the most obvious part of the definition. Usually in reinforcement learning when the agent starts to interact with the environment the agent does not know anything about that environment. The assumption in reinforcement learning that is always made is that the environment the agent interacts with contains some goal that the agent has to achieve.</p> 
<div class="flex-center">
    <Grid env={env_1} agent={agent_1}/>
</div>

<p>For example the agent is expected to move the circle from the starting cell position (top left corner) to the goal cell position (bottom left corner).</p>
<p>When we talk about learning, that means that the agent gets better at achieving that particular goal over time. It could start by moving in a random fashion and over time learn the best possible (meaning the shortest) route.</p>

<div class="flex-center">
    <Grid env={env_2} agent={agent_2}/>
</div>

<p>The agent above is a lot better at taking the shortest route between the starting point and the goal. Learning would basically mean transforming an agent from a random state to the one above.</p>

<p class="info"><strong>Learning</strong> means that the agent gets better at achieving the goal of the environment over time.</p>

<div class="separator"></div>

<h2>Rewards</h2>

<p>The question still remains how exactly does the agent know what the goal of the environment actually is? The environment with which the agent interacts gives feedback about the behaviour of the agent by giving out a reward after each single step that the agent takes.</p>

<p>If the goal of the grid world is to move the circle to the cell with the triangle as fast as possible the environment could for example give a positive reward for getting to the cell with the triangle and punish the agent in any other case.</p>

<div class="flex-center">
    <Grid env={env_3} agent={agent_3} isColoredReward=true/>
</div>

<p>The above animation shows color-coded rewards. The red grid cells give a reward of -1. The blue grid cell gives a reward of +1.</p>

<p>If the agent takes a random route to the triangle, then the sum of rewards is going to be very negative. If on the other hand like in the animation above the agent takes the direct route to the triangle, the sum of rewards is going to be larger (but still negative). The agent learns through the reward feedback that some sequences of actions are better than others. Generally speaking the agent needs to find the routes that produce high sum of rewards.</p>

<p class="info">In reinforcement learning the agent learns to maximize <strong>rewards</strong>. The goal of the environment is implicitly contained in the rewards.</p>

<div class="separator"></div>

<h2>Trial and Error</h2>

<p>The problem with rewards is that it is not clear from the very beginning what path produces the highest possible sum of rewards. It is therefore not clear which sequence of actions the agent needs to take. In reinforcement learning there is only the reward signal and even if the agent receives a positive sum of rewards it never knows if it could have done better. Unlike in supervised learning, there is no teacher/supervisor to tell the agent what the best behaviour is. So how can the agent figure out what sequence of actions produces the highest sum of rewards? The only way it can, by trial and error.</p>

<p>The agent has to try out different behaviour and produce different sequences of rewards to figure out which one produces optimal results. How long it takes the agent to find a good sequence of decisions depends on the complexity of the environment and the employed learning algorithm. It can be anything between a couple of seconds to many days. In some cases the agent can not solve an environment no matter how hard it tries.</p>

<p>Trial Nr. 1</p>
<Sequence sequenceLength=10/>
<p>Trial Nr. 2</p>
<Sequence sequenceLength=15/>
<p>Trial Nr. 3</p>
<Sequence sequenceLength=20/>

<p>The above animation shows how the sequences of actions might look like in the gridworld. Trial Nr. 1 takes the shortest route and has therefore the highest sum of rewards. Therefore it might be a good idea to follow the first sequence of actions more often that the sequence of actions taken in the third trial.</p>

<p class="info">In the context of reinforcement learning, trial and error means trying out different sequences of decisions and comparing the resulting sum of rewards to learn optimal behaviour.</p>

<div class="separator"></div>

<h2>Delayed</h2>

<p>In reinforcement learning the agent often needs to take dozens or even thousands of steps before a particular reward is achieved. In that case there has been a succession of many steps and the agent has to decide which step and in which proportion is responsible for the reward, so that the agent could select the decisions that lead to a good sequence of rewards more often.</p>

<p>Which of the steps is responsible for a particular reward? Is it the action just prior to the reward? Or the one before that? Or the one before that? Reinforcement Learning has no easy answer to the question which decision gets the credit for the reward. This problem is called <strong>the credit assignment problem</strong>.</p> 

<div class="flex-center">
    <CreditAssignment />
</div>

<p>In the grid world for example the first reward can only be assigned to the first action. The second reward can be assigned to the first and the second action. And so on. The last reward can theoretically be assigned to any of the actions taken prior to the reward.</p>

<p class="info">In reinforcement learning rewards for an action are often delayed, which leads to the credit assignment problem.</p> 

<div class="separator"></div>

<h2>Notes</h2>

<p>
    [1] This definition is highly inspired by the book "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto.
</p>

<div class="separator"></div>
