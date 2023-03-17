<script>
  import Container from "$lib/Container.svelte";
  import Alert from "$lib/Alert.svelte";
  import Footer from "$lib/Footer.svelte";
  import InternalLink from "$lib/InternalLink.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import Sequence from "./Sequence.svelte";
  import CreditAssignment from "./CreditAssignment.svelte";

  import { RandomAgent } from "$lib/reinforcement_learning/grid_world/RandomAgent";
  import { DeterministicAgent } from "$lib/reinforcement_learning/grid_world/DeterministicAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";
  import { Interaction } from "$lib/reinforcement_learning/grid_world/Interaction";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";

  const notes = [
    "This definition is highly inspired by the book 'Reinforcement Learning: An Introduction' by Richard S. Sutton and Andrew G. Barto.",
  ];

  let env_1 = new GridEnvironment(gridMap);
  let agent_1 = new RandomAgent(
    env_1.getObservationSpace(),
    env_1.getActionSpace()
  );
  let randomInteraction = new Interaction(agent_1, env_1, 3);

  const randomCellsStore = env_1.cellsStore;
  const randomPlayerStore = randomInteraction.observationStore;

  $: randomCells = $randomCellsStore;
  $: randomPlayer = $randomPlayerStore;

  let env_2 = new GridEnvironment(gridMap);
  let agent_2 = new DeterministicAgent(
    env_2.getObservationSpace(),
    env_2.getActionSpace()
  );
  let deterministicInteraction = new Interaction(agent_2, env_2, 3);
  const deterministicCellsStore = env_2.cellsStore;
  const deterministicPlayerStore = deterministicInteraction.observationStore;

  $: deterministicCells = $deterministicCellsStore;
  $: deterministicPlayer = $deterministicPlayerStore;
</script>

<svelte:head>
  <title>Definition of Reinforcement Learning - World4AI</title>
  <meta
    name="description"
    content="Reinforcement learning is defined as learning through trial and error and delayed rewards."
  />
</svelte:head>

<h1>Definition Of Reinforcement Learning</h1>
<div class="separator" />

<Container>
  <p>
    There are probably dozens of formal definitions of reinforcement learning.
    These definitions do not necessarily contradict each other, but rather
    explain something similar when we look a little deeper at what the
    definitions are trying to convey. In this section we are going to look at
    the one definition that should capture the essence of reinforcement learning
    in a very clear way.
  </p>
  <Alert type="info">
    Reinforcement Learning is characterized by learning through trial and error
    and delayed rewards<InternalLink type="note" id={1} />.
  </Alert>
  <p>
    The definition consists of three distinct parts: <Highlight
      >Learning</Highlight
    >,
    <Highlight>Trial and Error</Highlight>
    and <Highlight>Delayed Rewards</Highlight>. In order to understand the
    complete definition we will deconstruct the sentence and look at each part
    individually.
  </p>
  <div class="separator" />

  <h2>Learning</h2>
  <p>
    Learning is probably the most obvious part of the definition. When the agent
    starts to interact with the environment the agent does not know anything
    about that environment, but the environment contains some goal that the
    agent has to achieve.
  </p>
  <Grid cells={randomCells} player={randomPlayer} />
  <p>
    In the example above the agent is expected to move the circle from the
    starting cell (top left corner) to the goal cell (bottom left corner).
  </p>
  <Alert type="info">
    Learning means that the agent gets better at achieving the goal of the
    environment over time.
  </Alert>
  <p>
    When we talk about learning we imply that the agent gets better at achieving
    that particular goal over time. The agent would probably move randomly at
    first, but over time learn the best possible (meaning the shortest) route.
  </p>
  <Grid cells={deterministicCells} player={deterministicPlayer} />
  <div class="separator" />

  <h2>Rewards</h2>
  <p>
    The question still remains how exactly does the agent figure out what the
    goal of the environment actually is? The environment with which the agent
    interacts gives feedback about the behaviour of the agent by giving out a
    <Highlight>reward</Highlight> after each single step that the agent takes.
  </p>
  <Alert type="info">
    In reinforcement learning the agent learns to maximize rewards. The goal of
    the environment has to be implicitly contained in the rewards.
  </Alert>
  <p>
    If the goal of the grid world environment is to move the circle to the cell
    with the triangle as fast as possible the environment could for example give
    a positive reward when the agent reaches the goal cell and punish the agent
    in any other case.
  </p>
  <Grid
    cells={deterministicCells}
    player={deterministicPlayer}
    showColoredReward={true}
  />
  <p>
    The above animation represents that idea by color-coding rewards. The red
    grid cells give a reward of -1. The blue grid cell gives a reward of +1. If
    the agent takes a random route to the triangle, then the sum of rewards is
    going to be very negative. If on the other hand like in the animation above
    the agent takes the direct route to the triangle, the sum of rewards is
    going to be larger (but still negative). The agent learns through the reward
    feedback that some sequences of actions are better than others. Generally
    speaking the agent needs to find the routes that produce high sum of
    rewards.
  </p>
  <div class="separator" />

  <h2>Trial and Error</h2>
  <p>
    The problem with rewards is that it is not clear from the very beginning
    what path produces the highest possible sum of rewards. It is therefore not
    clear which sequence of actions the agent needs to take. In reinforcement
    learning the only feedback the agent receives is the reward signal and even
    if the agent receives a positive sum of rewards it never knows if it could
    have done better. Unlike in supervised learning, there is no teacher (a.k.a.
    supervisor) to tell the agent what the best behaviour is. So how can the
    agent figure out what sequence of actions produces the highest sum of
    rewards? The only way it can: by <Highlight>trial and error</Highlight>.
  </p>
  <p>
    The agent has to try out different behaviour and produce different sequences
    of rewards to figure out which sequence of actions is the optimal one. How
    long it takes the agent to find a good sequence of decisions depends on the
    complexity of the environment and the employed learning algorithm.
  </p>
  <p class="flex justify-center items-center font-bold">Trial Nr. 1</p>
  <Sequence sequenceLength="13" />
  <p class="flex justify-center items-center font-bold">Trial Nr. 2</p>
  <Sequence sequenceLength="10" />
  <p class="flex justify-center items-center font-bold">Trial Nr. 3</p>
  <Sequence sequenceLength="15" />
  <p>
    The above figures show how the sequences of actions might look like in the
    gridworld environment after three trials. In the second trial the agend
    takes the shortest route and has therefore the highest sum of rewards. It
    might therefore be a good idea to follow the first sequence of actions more
    often that the sequence of actions taken in the first and third trial.
  </p>
  <Alert type="info">
    In the context of reinforcement learning, trial and error means trying out
    different sequences of actions and compare the resulting sum of rewards to
    learn optimal behaviour.
  </Alert>
  <div class="separator" />

  <h2>Delayed</h2>
  <p>
    In reinforcement learning the agent often needs to take dozens or even
    thousands of steps before a reward is achieved. In that case there has been
    a succession of many steps and the agent has to decide which step and in
    which proportion is responsible for the reward, so that the agent could
    select the decisions that lead to a good sequence of rewards more often.
  </p>
  <Alert type="info">
    In reinforcement learning rewards for an action are often delayed, which
    leads to the credit assignment problem.
  </Alert>
  <p>
    Which of the steps is responsible for a particular reward? Is it the action
    just prior to the reward? Or the one before that? Or the one before that?
    Reinforcement Learning has no easy answer to the question which decision
    gets the credit for the reward. This problem is called <Highlight>
      the credit assignment problem</Highlight
    >.
  </p>
  <CreditAssignment />
  <p>
    Let's assume that in the grid world example the agent took 10 steps to reach
    the goal. The first reward can only be assigned to the first action. The
    second reward can be assigned to the first and the second action. And so on.
    The last (and positive) reward can theoretically be assigned to any of the
    actions taken prior to the reward.
  </p>
  <Footer {notes} />
</Container>
