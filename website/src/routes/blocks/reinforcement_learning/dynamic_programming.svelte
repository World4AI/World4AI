<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Dynamic Programming</title>
  <meta
    name="description"
    content="In Dynamic programming we have access to the model of the finite Markov decision process and can use iterative planning techniques to find the optimal value function and optimal policy."
  />
</svelte:head>

<h1>Dynamic Programming</h1>
<Question
  >Are dynamic programming algorithms learning or planning algorithms?</Question
>
<div class="separator" />
<p>
  The algorithms that we are going to cover in this section are not those that
  are commonly used to solve reinforcement learning tasks. In fact there is no
  learning involved at all. Dynamic programming (DP) requires the full knowledge
  of the model of the environment and calculates the optimal value function and
  optimal policy through the knowledge of that model. The interaction between
  the agent and the environment is not necessary. Nevertheless dynamic
  programming is the usual introduction to the solution of MDPs, because the
  knowledge gained by studying DP algorithms is transferable to reinforcement
  learning.
</p>
<div class="separator" />
