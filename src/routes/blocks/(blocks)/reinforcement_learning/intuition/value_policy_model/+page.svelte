<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";

  import { Interaction as InteractionClass } from "$lib/reinforcement_learning/grid_world/Interaction";

  import { DeterministicAgent } from "$lib/reinforcement_learning/grid_world/DeterministicAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";
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

  let optimalPolicy = {
    0: { 0: 2, 1: 2, 2: 2, 3: 2, 4: 2 },
    1: { 0: 1, 1: 1, 2: 1, 3: 2, 4: 2 },
    2: { 0: 0, 1: 0, 2: 0, 3: 2, 4: 2 },
    3: { 0: 2, 1: 3, 2: 3, 3: 3, 4: 2 },
    4: { 0: 0, 1: 3, 2: 3, 3: 3, 4: 3 },
  };
</script>

<svelte:head>
  <title>Value, Policy, Model - World4AI</title>
  <meta
    name="description"
    content="In reinforcement learning the value function, the policy and the model are essential components of agent. The environment on the other hand has a single component, the model."
  />
</svelte:head>

<h1>Value, Policy, Model</h1>
<div class="separator" />

<Container>
  <p>
    In order for the agent to determine the next action and for the environment
    to calculate the next state and the corresponding reward, the two make use
    of their respective internal components. The environment utilizes the <Highlight
      >model</Highlight
    >, while the agent might use up to three components: the
    <Highlight>value function</Highlight>, the
    <Highlight>policy</Highlight>
    and the
    <Highlight>model</Highlight>.
  </p>
  <Alert type="info">
    The environment has one component called model, while the agent might
    contain a value function, a policy and a model.
  </Alert>
  <p>
    The agent only requires the policy to work, nevertheless the model and the
    value function are major parts of many modern reinforcement learning
    algorithms, as these additional components can make the agent a lot more
    competent at solving a task.
  </p>
  <div class="separator" />

  <h2>The Model</h2>
  <p>
    The model is the only component of the environment. The model takes the
    current state and the action chosen by the agent as inputs and outputs the
    next state and the reward. Usually it consists of two distnict functions:
    the <Highlight>transition function</Highlight> calculates the next state, while
    the <Highlight>reward function</Highlight> calculates the corresponding reward.
  </p>
  <Alert type="info">
    The model consists of the transition function and the reward function.
  </Alert>
  <p>
    How exactly the model looks depends on the environment. Sometimes a simple
    table is all that is required. For our gridworld with 25 possible states and
    4 possible actions a table with 25 rows and 4 columns could be used to
    represent the model. The inner cells at the intersection between the current
    state and the action would contain the corresponding probabilities to
    transition into the next state and the reward. More complex environments
    like the Atari games would implement the model using a game engine.
  </p>
  <Alert type="info">
    The model of the agent is an approximation of the true model of the
    environment.
  </Alert>
  <p>
    In reinforcement learning the model of the environment is usually not
    something that the agent has access to. The agent has to learn to navigate
    in an environment where the rules of the game are not known. The agent can
    theoretically learn about the model by interacting with the environment.
    Essentially the agent creates some sort of an approximation of the true
    model of the environment. Each interaction allows the agent to improve its
    knowledge. Algorithms that implement a model for the agent are called <Highlight
      >model based</Highlight
    >, otherwise the algorithms are called
    <Highlight>model free</Highlight>.
  </p>
  <div class="separator" />

  <h2>Policy</h2>
  <Alert type="info">The policy of the agent maps states to actions.</Alert>
  <p>
    The policy calculates the action based the current state of the environment.
    For very simple environments the policy function might be simple a table
    that contains all possible states and the corresponding action for that
    state. The policy of the 5x5 grid world we used so far would also be
    contained in a mapping table, where the corresponding optimal policy might
    look as follows.
  </p>
  <Grid {player} {cells} policy={optimalPolicy} />
  <p>
    In more complex environments it is not possible to construct a mapping
    table, as the number of possible states is extremely high. In that case
    other solutions like neural networks are used.
  </p>
  <div class="separator" />

  <h2>Value Function</h2>
  <Alert type="info">
    The value function of the agent maps states to values.
  </Alert>
  <p>
    The value function gets a state as an input and generates a single scalar
    value. The value function plays an important role in most state of the art
    reinforcement learning algorithms. Intuitively speaking the agent looks at
    the state of the environment and assigns a value of "goodness" to the state.
    The higher the value, the better the state. With the help of the value
    function the agent tries to locate and move towards better and better
    states.
  </p>
  <p>
    Similar to the policy for simple environments the value function can be
    calculated with the help of a table or in more complex environments using a
    neural network. The grid world example below shows color coded values in the
    grid world environment. The orange value in the top left corner is the
    farthers away from the goal. The blue value in the bottom left corner is the
    goal that provides a positive reward. The colors inbetween are interpolated
    based on the distance from the goal. The closer the agent is to the goal the
    higher the values are expected to be. Therefore the agent could
    theoretically look around the current state and look for states with more
    "blueish" values and move into that direction to arrive at the goal.
  </p>
  <Grid {player} {cells} showColoredValues={true} />
  <p>
    Especially beginner level reinforcement learning agents have only a value
    function. In that case the policy of the agent is implicitly derived from
    the value function. Reinforcement learning algorithms that only utilize a
    value function are called <Highlight>value based methods</Highlight>. If on
    the other hand the agent derives the policy directly without using value
    functions the methods are called <Highlight>policy based methods</Highlight
    >. Most modern algorithms have agents with both components. Those are called
    <Highlight>actor-critic methods</Highlight>.
  </p>
  <div class="separator" />
</Container>
