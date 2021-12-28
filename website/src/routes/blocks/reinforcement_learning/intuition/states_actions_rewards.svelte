<script>
  import Question from "$lib/Question.svelte";
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

  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  import Interaction from "$lib/reinforcement_learning/intuition/states_actions_rewards/Interaction.svelte";
  import Action from "$lib/reinforcement_learning/grid_world/Action.svelte";
  import State from "$lib/reinforcement_learning/grid_world/State.svelte";
  import Reward from "$lib/reinforcement_learning/grid_world/Reward.svelte";
</script>

<svelte:head>
  <title>World4AI | Reinforcement Learning | States, Actions, Rewards</title>
  <meta
    name="description"
    content="In Reinforcement learning states, actions and rewards are send between the agent and the environment."
  />
</svelte:head>

<h1>States, Actions, Rewards</h1>
<Question
  >How does interaction between the agent and the environment look in
  reinforcement learning?</Question
>
<div class="separator" />

<h2>Interaction</h2>

<p>
  In essence interaction means that there is a communication channel between the
  agent and the environment, where data flows sequentially between the two. The
  agent and the environment interact continuously, each reacting to the data
  sent by the other.
</p>

<p class="info">
  In reinforcement learning the sequential information flow between the agent
  and the environment is called <strong>interaction</strong>.
</p>

<p>
  It is important to remember that this stream of data is sent in a strictly
  sequential way. When the environment sends the data for example, it has to
  wait until it receives the response from the agent. Only then can the
  environment generate a new batch of data and send it back to the agent again.
</p>

<div class="flex-center">
  <Interaction />
</div>

<p>
  The interaction cycle starts with the the agent receiving the current state of
  the environment. Based on that state the agent generates the action it would
  like to take, which is sent to the environment. The environment calculates the
  reward for that action and transitions into a new state, while taking the
  action of the agent and the current state into account. The new state and the
  reward is finally transmitted to the agent. The agent can use the reward as
  feedback to learn to make better decisions. The new state is once again used
  to generates a new action and the cycle repeats, ponentially forever.
</p>

<p class="info">
  In reinforcement learning there are just 3 types of data that need to be send
  between the agent and the environment: <strong>states</strong>,
  <strong>actions</strong>
  and <strong>rewards</strong>.
</p>

<div class="separator" />

<h2>State</h2>

<p>
  The state describes how the environment actually looks like. It is the
  situation the agent faces and needs to take into account to make good
  decisions.
</p>

<p class="info">
  The state is the representation of the current condition of the environment.
</p>

<div class="flex-space">
  <Grid {cells} {player} />
  <State state={player} />
</div>

<p>
  In our simple gridworld example all the agent needs to know to make the
  decisions is the location of the circle in the environment. In the starting
  position the state would be row=0 and column=0. The state to the right of the
  starting position would be row equals to 0 and column equals to 1, meaning (0,
  1). Based on the position the agent can choose the path towards the triangle.
  Above the animation you can observe how the state changes depending on the
  position of the circle.
</p>

<p>
  This is not the only way to represent the the state of the environment. The
  state can be represented by a scalar, a vector, a matrix or a tensor and can
  be either discrete or continuous. In future chapters we will see more complex
  environments and learn how to deal with those. For now it is sufficient to
  know what role the state plays in the action-environment interaction.
</p>

<div class="separator" />

<h2>Action</h2>

<p>
  The action is the behaviour the agent chooses based on the state of the
  environment (or sometimes randomly). Like the state the action can be a
  scalar, a vector, a matrix or a tensor of discrete or continuous values.
</p>

<p class="info">
  The action is the representation of the decision of the agent.
</p>

<div class="flex-space">
  <Grid {cells} {player} />
  <Action {action} />
</div>

<p>
  In the above gridworld example the agent can move north, east, south and west.
  Each action is encoded by a discrete scalar value, where north equals 0, east
  equals 1, south equals 2 and west equals 3.
</p>

<div class="separator" />

<h2>Reward</h2>

<p>
  The reward is what the agent receives from the environment for an action. It
  is the value that the environment uses to reinforce a behaviour to solve an
  environment and it is the value that the agent uses to improve it's behaviour.
</p>

<p>
  Unlike the action or the state the reward has to be a scalar, one single
  number, it is not possible for the reward to be a vector, matrix or tensor. As
  expected larger numbers represent larger or better rewards so that the reward
  of 1 is higher than the reward of -1.
</p>

<p class="info">
  The reward is the scalar signal to reinforce certain behaviour of the agent.
</p>

<div class="flex-space">
  <Grid {cells} {player} />
  <Reward {reward} />
</div>

<p>
  In this gridworld example the agent receives a reward of -1 for each step
  taken with the exception of reaching the triangle, where the agent receives a
  reward of +1.
</p>

<div class="separator" />
<h2>Timestep</h2>

<p>
  Reinforcement learning works in discrete timesteps. Each iteration where the
  environment and the agent each have sent their data constitutes a timestep.
</p>

<p class="info">
  In reinforcement learning each iteration of exchanging an action for a state
  and a reward is called a timestep.
</p>

<div class="separator" />
