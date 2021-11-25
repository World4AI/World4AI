<script>
  import Question from '$lib/Question.svelte';
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | MDP Introduction</title>
    <meta name="description" content="In reinforcement learning the Markov decison process is the mathematical formalization of the agent environment interaction.">
</svelte:head>

<h1>Markov Decision Process</h1>
<Question>Why do we need a Markov Decicion Process?</Question>
<div class="separator"></div>

<p>In order to find an optimal solution to a reinforcement learning problem it is essential to formalize the problem in a mathematical framework. This allows researchers to study the properties of the problem and to develop algorithms. In reinforcement learning the tool that is commonly used for those purposes is the Markov decision process (often abbreviated as MDP).</p>
<p>Many of the components of the Markov decision process were already covered in the previous chaptes, but while the focus of the previous chapter was the intuition of reinforcement learning this chapter is going to develop the necessary mathematical foundation.</p>
<p>There are several ways to introduce MDPs, each way has it's own advantages and a different level of mathematical rigour. We are going to look at three different viewpoints of MDPs, while increasing the mathematical complexity one step at a time. At the end of the capter we will have a look at what it means to find a solution to a Markov decision process.</p>
<div class="separator"></div>
