<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Generalized Policy Iteration</title
  >
  <meta
    name="description"
    content="Policy Iteration and value iteration algorithms are implementations of generalized policy iteration algorithms. Many reinforcement learning algorithms follow a generalized policy iteration approach by switching between policy evaluation and policy improvement."
  />
</svelte:head>

<h1>Generalized Policy Iteration</h1>
<Question>What is generalized policy iteration?</Question>
<div class="separator" />
<p>
  Policy iteration and value iteration are both versions of generalized policy
  iteration (GPI) algorithms. Switching between policy evaluation and policy
  improvement is the core of GPI regardless of the implementation details of the
  algorithm.
</p>
<p>
  Many of the reinforcement learning algorithms yet to come are based on
  generalized policy iteration. Evaluation and improvement alternate to
  eventually find optimal value and policy functions.
</p>
<div class="separator" />
