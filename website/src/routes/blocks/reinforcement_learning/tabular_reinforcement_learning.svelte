<script>
  import Question from "$lib/Question.svelte";
</script>

<svelte:head>
    <title>World4AI | Reinforcement Learning | Tabular Reinforcement Learning</title>
    <meta name="description" content="Tabular reinforcement learning deals with finding optimal value functions and policies for finite markov decision processes.">
</svelte:head>

<h1>Tabular Reinforcement Learning</h1>
<Question>What is tabular reinforcement learning?</Question>
<div class="separator" />

<p>
  Dynamic programming offers us valuable theoretical foundations, but generally
  speaking we do not have access to the model of the environment. The only
  viable solution that remains is learning through interaction. This chapter is
  dedicated to learning with finite MDPs. Finite Markov Decision Processes have
  finite State, Action and Reward sets. This allows for the implementation of
  state-value and action-value functions that can be stored in lists or tables.
  Those implementations are appropriately called tabular methods. The current
  state of the art reinforcement learning rarely deals with tabular methods any
  more, but it is still more convenient to start the exploration of
  reinforcement learning techniques with those, as the general ideas are
  extremely relevant to modern (approximative) methods which are going to be
  introduced in future sections.
</p>
<div class="separator" />
