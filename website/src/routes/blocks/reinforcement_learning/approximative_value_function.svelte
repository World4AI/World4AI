<script>
  import Question from "$lib/Question.svelte";
  import Table from "$lib/Table.svelte";

  let discreteHeader = ["State", "Action 1", "Action 2"];
  let discreteData = [
    [0, 0, 1],
    [1, 0, 1],
  ];

  let discreteInfiniteHeader = ["State", "Action 1", "Action 2"];
  let discreteInfiniteData = [
    [0, 0, 1],
    [1, 0, 1],
    [2, 0, 1],
    [3, 2, 2],
    [4, -1, 1],
    ["...", "...", "..."],
    [1_000_000_000, 22, 25],
  ];

  let continuousHeader = ["State Representation", "Action 1", "Action 2"];
  let continuousData = [
    [1.1, 1, 2],
    [1.3, 1.2, 1.8],
    [1.5, 1.5, 1.2],
    [1.7, "?", "?"],
    [2.1, 1.8, 1.3],
    [2.2, 1.7, 1.8],
    [2.5, 1.5, 2],
  ];
</script>

<h1>Approximative Value Function</h1>
<Question>Why are approximative value functions useful?</Question>
<div class="separator" />

<p>
  So far we have dealt with tabular reinforcement learning and finite Markov
  decision processes.
</p>
<Table header={discreteHeader} data={discreteData} />

<p>
  The number of rows and columns in the Q-Table was finite. This allowed us to
  loop over all state-action pairs and apply Monte Carlo or temporal difference
  learning. Given enough iterations we were guaranteed to arrive at the optimal
  solution.
</p>
<p>
  Most interesting reinforcement learning problems do not have such nice
  properties. In case state or action sets are infinite or extremely large it
  becomes impossible to store the value function as a table.
</p>

<Table header={discreteInfiniteHeader} data={discreteInfiniteData} />
<p>
  The above table shows action-values for 1,000,000,000 discrete states. Even if
  we possessed a computer which could efficiently store a high amount of states,
  we still need to loop over all these states and thus convergence would be
  extremely slow.
</p>
<p>
  Q-Tables become almost impossible when the agent has to deal with continuous
  variables. When the agent encounters a continuous observation it is extremely
  unlikely that the same exact value will be seen again. Yet the agent will need
  to learn how to deal with future unseen observations. We expect from the agent
  to find a policy that is “good” across many different observations. The key
  word that we are looking for is generalization.
</p>
<Table header={continuousHeader} data={continuousData} />
<p>
  The example above shows how generalization might look like. The state is
  represented by 1 single continuous variable and there are only 2 discrete
  actions available (left and right). If you look at the state representation
  with the value of 1.7, could you approximate the action-values and determine
  which action has the higher value? You will probably not get the exact correct
  answer but the value for the left action should probably be somewhere between
  1.5 and 1.8. Real reinforcement learning tasks might be a lot more complex,
  but I it is not a bad mental model to imagine the agent interpolating between
  states that were already encountered.
</p>
<p>
  In the case when the state/action sets are extremely large and/or continous,
  lookup tables for each state-action pair become impossible. As the number of
  states and/or actions is possibly infinite, the function has to generalize and
  we can assume that it is impossible to create a function that generates
  optimal values for each state-action pair. Thus it becomes necessary to create
  value functions that are not exact, but approximative, meaning that the value
  function does not return the true value of a policy but a value that is
  hopefully close enough. Finding the optimal policy and value function is often
  not possible, but the general idea is to find a policy that still performs in
  a way that generates a relatively high expected sum of rewards.
</p>
<p>
  We can look at it this way. Humans most definitely don’t have optimal policies
  for complex tasks they have to perform. If they did, then chess for example
  would be extremely boring and would not have survived for many hundred years.
  Still you can appreciate the complexity of the game and the extremely high
  level of professional players. We are going to have similar expectations for
  our agents. Even though we will not be able to find optimal value functions
  for some of the environments we are going to generate extremely impressive
  results.
</p>
<div class="separator" />
