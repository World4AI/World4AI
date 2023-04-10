<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
</script>

<svelte:head>
  <title>Dynamic Programming - World4AI</title>
  <meta
    name="description"
    content="In dynamic programming we have access to the model of the finite Markov decision process and can use iterative planning techniques to find the optimal value function and optimal policy."
  />
</svelte:head>

<h1>Dynamic Programming</h1>
<div class="separator" />

<Container>
  <p>
    The algorithms that we are going to cover in this section are known as <Highlight
      >dynamic programming</Highlight
    >. Dynamic programming is not commonly used to solve reinforcement learning
    tasks. In fact there is no learning involved at all. Instead of interacting
    with the environment to find an optimal policy, dynamic programming utilizes <Highlight
      >planning</Highlight
    >.
  </p>
  <Alert type="info"
    >Planning utilizes a model of the environment to improve a policy.</Alert
  >
  <p>
    Dynamic programming requires the full knowledge of the model of the
    environment and calculates the optimal value function and optimal policy
    through the knowledge of that model. The interaction between the agent and
    the environment is not necessary. While the access to the model of the
    environment is an unrealistic assumtion, the knowledge tha you will gain by
    studying dynamic programming algorithms is transferable to reinforcement
    learning.
  </p>
  <div class="separator" />
</Container>
