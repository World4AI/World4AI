<script>
  import Container from "$lib/Container.svelte";
  import Interaction from "$lib/reinforcement_learning/intuition/states_actions_rewards/Interaction.svelte";
  import Table from "$lib/reinforcement_learning/grid_world/Table.svelte";
  import Function from "../_components/Function.svelte";
  import { Interaction as InteractionClass } from "$lib/reinforcement_learning/grid_world/Interaction";
  import { DeterministicAgent } from "$lib/reinforcement_learning/grid_world/DeterministicAgent";
  import { RandomAgent } from "$lib/reinforcement_learning/grid_world/RandomAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import {
    gridMap,
    gridCorridor,
  } from "$lib/reinforcement_learning/grid_world/maps";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  let env = new GridEnvironment(gridMap);
  let agent = new DeterministicAgent(
    env.getObservationSpace(),
    env.getActionSpace()
  );
  let interaction = new InteractionClass(agent, env, 2);

  let cellsStore = env.cellsStore;
  let playerStore = interaction.observationStore;

  $: cells = $cellsStore;
  $: player = $playerStore;

  let corridor = new GridEnvironment(gridCorridor);
  let corridorAgent = new RandomAgent(
    corridor.getObservationSpace(),
    corridor.getActionSpace()
  );
  let corridorInteraction = new InteractionClass(corridorAgent, corridor, 2);

  let corridorCellsStore = corridor.cellsStore;
  let corridorPlayerStore = corridorInteraction.observationStore;

  $: corridorCells = $corridorCellsStore;
  $: corridorPlayer = $corridorPlayerStore;

  let optimalPolicy = {
    0: { 0: 2, 1: 2, 2: 2, 3: 2, 4: 2 },
    1: { 0: 1, 1: 1, 2: 1, 3: 2, 4: 2 },
    2: { 0: 0, 1: 0, 2: 0, 3: 2, 4: 2 },
    3: { 0: 2, 1: 3, 2: 3, 3: 3, 4: 2 },
    4: { 0: 0, 1: 3, 2: 3, 3: 3, 4: 3 },
  };

  const modelHeader = ["State", "Action 0", "Action 1"];
  const modelData = [
    [
      0,
      [{ state: 0, probability: "100%", reward: -1 }],
      [
        { state: 0, probability: "50%", reward: -1 },
        { state: 1, probability: "50%", reward: -1 },
      ],
    ],
    [
      1,
      [
        { state: 0, probability: "50%", reward: -1 },
        { state: 1, probability: "50%", reward: -1 },
      ],
      [
        { state: 1, probability: "50%", reward: -1 },
        { state: 2, probability: "50%", reward: 1 },
      ],
    ],
    [
      2,
      [{ state: 2, probability: "100%", reward: 0 }],
      [{ state: 2, probability: "100%", reward: 0 }],
    ],
  ];

  const policyHeader = ["State", "Action"];
  const policyData = [
    [0, 1],
    [1, 1],
    [2, 1],
  ];

  const valueHeader = ["State", "Value"];
  const valueData = [
    [0, 1],
    [1, 2],
    [2, 0],
  ];
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | Value, Policy, Model</title>
  <meta
    name="description"
    content="In reinforcement learning the value function, the policy and the model are essential components of agent. The environment has only one component, the model."
  />
</svelte:head>

<h1>Value, Policy, Model</h1>
<div class="separator" />

<Container>
  <Interaction />
  <p>
    The agent and the environment provide data (state, action, reward) to each
    other through the interaction process. That data is calculated using the
    respective internal components. The environment utilizes only one component,
    the <strong>model</strong>, while the agent might use up to three
    components: the
    <strong>value function</strong>, the
    <strong>policy</strong>
    and the
    <strong>model</strong>.
  </p>
  <p class="info">
    The environment has one component called model, while the agent might have a
    value function, a policy and a model as components.
  </p>
  <p>
    These components are basically functions: they take inputs and generate
    outputs. Oftentimes the word mapping is used in that context. A function
    that takes x as input and outputs y is said to map x to y. (See interactive
    example below).
  </p>
  <Function />
  <p>
    To simplify some of the explanations additionally to the grid world we are
    going to use a one dimensional grid world with three cells, the agent has
    the option to either move left or right and the goal is to reach the
    triangle in the right end of the corridor.
  </p>
  <Grid
    height={100}
    width={300}
    maxWidth={"300px"}
    player={corridorPlayer}
    cells={corridorCells}
  />
  <div class="separator" />

  <h2>Model of the Environment</h2>
  <p>
    The model (sometimes called the dynamics) is the only component of the
    environment, yet that component fulfills two tasks. The first task is to
    calculate the next state of the environment based on the current state and
    the action chosen by the agent. This process of moving from one state to the
    next state is called transitioning (into the next state). The second task is
    to calculate the reward based on the current state and the action.
  </p>
  <p class="info">
    The model consists of the transition function and the reward function.
  </p>
  <p>
    How exactly the model looks depends on the environment. Sometimes a simple
    table is all that is required.
  </p>
  <p>
    For a gridworld with 3 possible states and 2 possible actions a table with 3
    rows and 2 columns could be used to represent the model. The inner cells at
    the interaction between the current state and the action would contain the
    probabilities to transition into the next state and the corresponding
    reward. For example if the state of the environment is 1 and the agent takes
    action number 1, with 50% probability the next state is going to be 1 again
    and the reward will correspond to -1 and with 50% probability the
    environment will transition into the state 2 with reward of 1.
  </p>
  <div class="separator" />
  <Grid
    height={100}
    width={300}
    maxWidth={"300px"}
    player={corridorPlayer}
    cells={corridorCells}
  />
  <div class="separator" />
  <Table header={modelHeader} data={modelData} />

  <p>
    More complex environments like the atari games would have their game engine
    and game logic that would calculate the transitions and rewards.
  </p>

  <p>
    In reinforcement learning the model of the environment is usually not
    something that the agent has access to. The agent has to learn to navigate
    in an environment where the rules of the game are not known.
  </p>

  <p>
    In most cases reinforcement learning practitioners do not deal with the
    creation of new environments. There are already hundreds of ready made
    environments that they can access. This reduces development speed and allows
    comparisons among different researchers and algorithms.
  </p>

  <div class="separator" />

  <h2>Components of the Agent</h2>
  <p>
    The agent has up to three main components. The policy function, the value
    function and a model. Generally only the policy is actually required for the
    agent to work. Nevertheless, the model and the value function are major
    parts of many modern reinforcement learning algorithms. Especially the value
    function is often considered to be a necessary component of a successful
    agent.
  </p>
  <div class="separator" />

  <h3>Policy</h3>
  <p>
    The first component is the policy. The policy calculates the action directly
    based the current state of the environment.
  </p>

  <Function inputs={["S1", "S2", "S3"]} outputs={["A1", "A2", "A3"]} />
  <p class="info">The policy of the agent maps states to actions.</p>
  <p>
    For very simple environments the policy function might also be a table that
    contains all possible states and for each state there is a corresponding
    action. In more complex environments it is not possible to construct a
    mapping table like the one above, as the number of states is extremely high.
    In that case other solutions like neural networks are used.
  </p>
  <div class="separator" />
  <Grid
    height={100}
    width={300}
    maxWidth={"300px"}
    player={corridorPlayer}
    cells={corridorCells}
  />
  <div class="separator" />
  <Table header={policyHeader} data={policyData} />

  <p>
    In the corridor example above the optimal policy is to always move right
    (action 1).
  </p>
  <p>
    The policy of the 5x5 grid world we used so far would also be contained in a
    mapping table, where the corresponding optimal policy might look as follows.
  </p>
  <Grid {player} {cells} policy={optimalPolicy} />
  <div class="separator" />

  <h3>Value Function</h3>
  <p>
    The second component of the agent is the so-called value function. The value
    function gets a state as an input and generates a single scalar value. The
    value function plays an important role in most state of the art
    reinforcement learning algorithms. Intuitively speaking the agent looks at
    the state of the environment and assigns a value of "goodness" to the state.
    The higher the value, the better the state. With the help of the value
    function the agent tries to locate and move towards better and better
    states.
  </p>
  <Function inputs={["S1", "S2", "S3"]} outputs={[0, 1, 2]} />
  <p class="info">The value function of the agent maps states to values.</p>

  <p>
    Similar to the policy for simple environments the value function can be
    calculated with the help of a table or in more complex environments using a
    neural network.
  </p>
  <Grid
    height={100}
    width={300}
    maxWidth={"300px"}
    player={corridorPlayer}
    cells={corridorCells}
  />
  <Table header={valueHeader} data={valueData} />
  <p>
    The grid world example below shows color coded values in the grid world
    environment. The orange value in the top left corner is the farthers away
    from the goal. The blue value in the bottom left corner is the goal that
    provides a positive reward. The colors inbetween are interpolated based on
    the distance from the goal. The closer the agent is to the goal the higher
    the values are expected to be. Therefore the agent could theoretically look
    around the current state and look for states with more "blueish" values and
    move into that direction to arrive at the goal.
  </p>
  <Grid {player} {cells} showColoredValues={true} />
  <div class="separator" />

  <h3>Model</h3>
  <p>
    The third and last component is the model. The model of the environment is
    something that the agent generally has no access to, but the agent can
    theoretically learn about the model by interacting with the environment.
    Essentially the agent creates some sort of an approximation of the true
    model of the environment. Each interaction allows the agent to improve his
    knowledge regarding the transition probabilities from one state to the next
    and the corresponding rewards. The model can for example be used to improve
    the policy. This is especially useful when interacting with the environment
    is for some reason costly. Additionally the model can connect to the policy
    in generate better actions.
  </p>
  <p class="info">
    The model of the agent is an approximation of the true model of the
    environment.
  </p>
  <div class="separator" />
</Container>
