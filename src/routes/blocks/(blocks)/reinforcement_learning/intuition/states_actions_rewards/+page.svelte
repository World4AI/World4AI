<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Latex from "$lib/Latex.svelte";

  import { RandomAgent } from "$lib/reinforcement_learning/grid_world/RandomAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";
  import { Interaction as InteractionClass } from "$lib/reinforcement_learning/grid_world/Interaction";
  let env = new GridEnvironment(gridMap);
  let agent = new RandomAgent(env.getObservationSpace(), env.getActionSpace());
  let interaction = new InteractionClass(agent, env, 2);

  let cellsStore = env.cellsStore;
  let playerStore = interaction.observationStore;
  let actionStore = interaction.actionStore;
  let rewardStore = interaction.rewardStore;
  $: cells = $cellsStore;
  $: player = $playerStore;
  $: action = $actionStore;
  $: reward = $rewardStore;

  import Interaction from "./Interaction.svelte";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  import Action from "$lib/reinforcement_learning/grid_world/Action.svelte";
  import State from "$lib/reinforcement_learning/grid_world/State.svelte";
  import Reward from "$lib/reinforcement_learning/grid_world/Reward.svelte";
</script>

<svelte:head>
  <title>States, Actions, Rewards - World4AI</title>
  <meta
    name="description"
    content="In reinforcement learning the agent and the environment exchange three distinct signals between each other: the state, the action and reward."
  />
</svelte:head>

<h1>States, Actions, Rewards</h1>
<div class="separator" />

<Container>
  <p>
    In reinforcement learning the agent and the environment interact with each
    other. In this context interaction means that signals flows sequentially
    between the two. The agent and the environment interact continuously, each
    reacting to the data sent by the other.
  </p>
  <Alert type="info">
    In reinforcement learning the sequential information flow between the agent
    and the environment is called interaction.
  </Alert>
  <p>
    It is important to understand that this stream of data is exchanged in a
    strictly sequential way. When the environment sends a signal for example, it
    has to wait until it receives the response signal from the agent. Only then
    can the environment generate a new batch of data. Reinforcement learning
    works in discrete timesteps. Each iteration where the environment and the
    agent exchanged their data constitutes a <Highlight>timestep</Highlight>.
  </p>
  <Alert type="info">
    In reinforcement learning there are just 3 types of data that need to be
    send between the agent and the environment: <Highlight>states</Highlight>,
    <Highlight>actions</Highlight>
    and <Highlight>rewards</Highlight>.
  </Alert>
  <Interaction />
  <p>
    The interaction cycle starts with the the agent receiving the initial state
    <Latex>S_0</Latex>
    from the environment. Based on that state the agent generates the action <Latex
      >A_0</Latex
    > it would like to take, which is transmitted to the environment. The environment
    transitions into the new state<Latex>S_1</Latex> and calculates the reward <Latex
      >R_1</Latex
    >. The new state and the reward are finally transmitted to the agent. The
    agent can use the reward as a feedback to learn, while the new state is used
    to generates the action <Latex>A_1</Latex> and the cycle keeps repeating, ponentially
    forever.
  </p>
  <div class="separator" />

  <h2>State</h2>
  <Alert type="info">
    The state is the representation of the current condition of the environment.
  </Alert>
  <p>
    The state describes how the environment actually looks like. It is the
    condition that the agent is facing and the one parameter that the agent
    bases its decisions on.
  </p>
  <p>
    In our simple gridworld example all the agent needs to know to make the
    decisions is the location of the circle in the environment. In the starting
    position the state would be row=0 and column=0. The state to the right of
    the starting position would be row equals to 0 and column equals to 1,
    meaning (0, 1). Based on the position the agent can choose the path towards
    the triangle.
  </p>
  <div class="mx-auto mb-4 max-w-sm">
    <State state={player} />
  </div>
  <Grid {cells} {player} />
  <p>
    This is not the only way to represent the the state of the environment. The
    state can be represented by a scalar, a vector, a matrix or a tensor and can
    be either discrete or continuous. In future chapters we will see more
    complex environments and learn how to deal with those. For now it is
    sufficient to know what role the state plays in the action-environment
    interaction.
  </p>
  <div class="separator" />

  <h2>Action</h2>
  <Alert type="info">
    The action is the representation of the decision of the agent.
  </Alert>
  <p>
    The action is the behaviour the agent chooses based on the state of the
    environment. Like the state the action can be a scalar, a vector, a matrix
    or a tensor of discrete or continuous values.
  </p>
  <p>
    In the gridworld example the agent can move north, east, south and west.
    Each action is encoded by a discrete scalar value, where north equals 0,
    east equals 1, south equals 2 and west equals 3.
  </p>
  <div class="mx-auto mb-4 max-w-xs">
    <Action {action} />
  </div>
  <Grid {cells} {player} />
  <div class="separator" />

  <h2>Reward</h2>
  <Alert type="info">
    The reward is the scalar signal to reinforce certain behaviour of the agent.
  </Alert>
  <p>
    The reward is what the agent receives from the environment for an action. It
    is the value that the environment uses to reinforce a behaviour and it is
    the value that the agent uses to improve it's behaviour.
  </p>
  <p>
    Unlike the action or the state the reward has to be a scalar, one single
    number, it is not possible for the reward to be a vector, matrix or tensor.
    As expected larger numbers represent larger or better rewards so that the
    reward of 1 is higher than the reward of -1.
  </p>
  <p>
    In this gridworld example the agent receives a reward of -1 for each step
    taken with the exception of reaching the triangle, where the agent receives
    a reward of +1.
  </p>
  <div class="mx-auto mb-4 max-w-xs">
    <Reward {reward} />
  </div>
  <Grid {cells} {player} />
  <div class="separator" />
</Container>
