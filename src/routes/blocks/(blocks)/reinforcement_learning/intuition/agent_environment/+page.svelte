<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Alert from "$lib/Alert.svelte";
  import Robot from "./Robot.svelte";

  import { RandomAgent } from "$lib/reinforcement_learning/grid_world/RandomAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";
  import { Interaction } from "$lib/reinforcement_learning/grid_world/Interaction";
  let env = new GridEnvironment(gridMap);
  let agent = new RandomAgent(env.getObservationSpace(), env.getActionSpace());
  let interaction = new Interaction(agent, env, 2);
  const cellsStore = env.cellsStore;
  const playerStore = interaction.observationStore;

  $: cells = $cellsStore;
  $: player = $playerStore;

  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
</script>

<svelte:head>
  <title>Agent and Environment - World4AI</title>
  <meta
    name="description"
    content="The agent and the environment are the two main components in reinforcement learning. The agent is the component that generates the decisons. The environment is everything else."
  />
</svelte:head>

<h1>Agent and Environment</h1>
<div class="separator" />

<Container>
  <p>
    All of reinforcement learning is based on two main components, the
    <Highlight>agent</Highlight>
    and the <Highlight>environment</Highlight>. Let's use the gridworld below to
    get familiar with the two components.
  </p>
  <Grid {cells} {player} />
  <p>
    In the gridworld the main player is represented by a circle. The player can
    move into 4 different directions: North, East, South and West. If the circle
    is against an outer wall or a barrier (red boxes) the circle can not move in
    that particular direction. The goal of the game is for the circle to reach
    the triangle in the bottom left corner in as few steps as possible. For the
    time being the circle moves randomly.
  </p>

  <p>
    Intuitively we could say that the circle is the agent and the grid world is
    the environment, but that definition would not be entirely correct. There is
    actually a relatively strict separation between the agent and the
    environment.
  </p>
  <Alert type="info">The agent is the decision maker.</Alert>
  <p>
    All the agent can actually do is to make the decisions. In the case of the
    above gridworld the agent chooses the direction. Whether the circle actually
    moves in that direction is outside of the influence of the agent.
  </p>
</Container>

<Container>
  <p>
    For example in the gridworld above the agent can decide to go north even
    when the circle is against a barrier or a wall in the northern direction.
    That decision might be legitimate in many grid worlds, but the position of
    the circle will not change.
  </p>
  <p>
    The agent is the program that generates the decision and the decision of the
    agent is then relayed to the environment. The environment processes the
    decision, but that is not something that the agent can influence. In the
    simple grid-world game if the agent decides to go north the circle might
    actually move north or it could move in a totally different direction or not
    move at all.
  </p>
  <Alert type="info">Anything outside of the agent is the environment.</Alert>
  <p>
    The environment on the other hand is everything that is not the agent. The
    enviroment in the grid world reacts to the decisions of the agent and
    calculates what position the circle moves to. Additionally the environment
    rewards the agent for its decisions.
  </p>
  <p>
    A different example that is often used to make the distinction between the
    agent and the environment is that of a robot that interacts with the real
    world.
  </p>
  <Robot />
  <p>
    The goal of the agent might be to move the robot from right to left to
    recharge its battery. The agent sends the decison to move the robot. But it
    is entirely possible that the robot does not even start moving because the
    battery is already completely depleted. In this example the agent is the
    code that makes the decision to move left. While the arms, the legs, the
    cooling system of the robot, the battery, the floor and everything else in
    the image is part of the environment.
  </p>
  <p>
    A similar argument can be made for a human or any other form of a biological
    machine. The neural network in our brain makes the decisions to move, to
    study or to sleep. How the body actually reacts is not really in our
    control. For example the movement could be stopped when you are parallyzed
    either because of an illness or through fear. The desire to study could be
    stopped through external triggers like the smell of food or a habit to
    listen to music. And the decision to go to bed to fall asleep is not always
    accompanied by actual sleep. We do not have full control of our body. For
    example, no matter how hard we try, we can not stop our heart though our
    will.
  </p>
  <div class="separator" />
</Container>
