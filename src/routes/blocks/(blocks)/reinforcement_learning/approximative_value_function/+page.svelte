<script>
  import Container from "$lib/Container.svelte";
  import Table from "$lib/Table.svelte";
  import CartPole from "$lib/reinforcement_learning/CartPole.svelte";

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

<svelte:head>
  <title>World4AI | Reinforcement Learning | Value Function Approximation</title
  >
  <meta
    name="description"
    content="When the state space is too large or continuous, it gets impossible to represent a value function as a table. In that case the agent needs to use an approximate value function."
  />
</svelte:head>

<h1>Approximative Value Function</h1>
<div class="separator" />

<Container>
  <p>
    So far we have dealt with tabular reinforcement learning and finite Markov
    decision processes.
  </p>
  <Table header={discreteHeader} data={discreteData} />
  <p>
    The number of rows and columns in the Q-Table was finite. This allowed us to
    loop over all state-action pairs and apply Monte Carlo or temporal
    difference learning. Given enough iterations we were guaranteed to arrive at
    the optimal solution.
  </p>
  <p>
    Most interesting reinforcement learning problems do not have such nice
    properties. In case state or action sets are infinite or extremely large it
    becomes impossible to store the value function as a table.
  </p>
  <Table header={discreteInfiniteHeader} data={discreteInfiniteData} />
  <p>
    The above table shows action-values for 1,000,000,000 discrete states. Even
    if we possessed a computer which could efficiently store a high amount of
    states, we still need to loop over all these states to improve the
    estimation and thus convergence would be extremely slow.
  </p>
  <p>
    Representing a value function in Q-tables become almost impossible when the
    agent has to deal with continuous variables. When the agent encounters a
    continuous observation it is extremely unlikely that the same exact value
    will be seen again. Yet the agent will need to learn how to deal with future
    unseen observations. We expect from the agent to find a policy that is
    “good” across many different observations. The key word that we are looking
    for is generalization.
  </p>
  <Table header={continuousHeader} data={continuousData} />
  <p>
    The example above shows how generalization might look like. The state is
    represented by 1 single continuous variable and there are only 2 discrete
    actions available (left and right). If you look at the state representation
    with the value of 1.7, could you approximate the action-values and determine
    which action has the higher value? You will probably not get the exact
    correct answer but the value for the left action should probably be
    somewhere between 1.5 and 1.8. Real reinforcement learning tasks might be a
    lot more complex, but I it is not a bad mental model to imagine the agent
    interpolating between states that were already encountered.
  </p>
  <p>
    In the following sections we are going to deal with Markov decision
    processes that have a large or continuous state space, but a small discrete
    number of actions. Due to the unlimited number of states it becomes
    necessary to create value functions that are not exact, but approximative,
    meaning that the value function does not return the true value of a policy
    but a value that is hopefully close enough. Finding the optimal policy and
    value function is often not possible, but generalally we will attempt to
    find a policy that still performs in a way that generates a relatively high
    expected sum of rewards.
  </p>
  <p>
    In the cart pole environment below the task is to balance the pole and not
    to move too far away from the center of the screen. At each timestep the
    agent can either move left or right and gets a reward of +1. If the angle of
    the pole is too large or the cart goes offscreen the game ends. The agent
    can observe the car position, the cart velocity, the angle of the pole and
    the angular velocity. Each of the variables is continuous and it is
    therefore becomes necessary to approximate the value function.
  </p>
  <CartPole />
  <p>
    Before we move on to the next section, we should mention the limitation of
    approximative value based methods. Value based approximative methods are
    able to deal with a continuous state space, but not continuous actions. Yet
    as you can imagine many of the tasks a robot for example would need to
    perform require continuous actions. In later chapters we will discuss how we
    can deal with these limitations.
  </p>
  <div class="separator" />
</Container>
