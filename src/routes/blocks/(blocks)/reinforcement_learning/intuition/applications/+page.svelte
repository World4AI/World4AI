<script>
  import Container from "$lib/Container.svelte";
  import Highlight from "$lib/Highlight.svelte";
  import { Interaction } from "$lib/reinforcement_learning/grid_world/Interaction";
  import { RandomAgent } from "$lib/reinforcement_learning/grid_world/RandomAgent";
  import { GridEnvironment } from "$lib/reinforcement_learning/grid_world/GridEnvironment";
  import { gridMap } from "$lib/reinforcement_learning/grid_world/maps";
  import Grid from "$lib/reinforcement_learning/grid_world/Grid.svelte";
  let env = new GridEnvironment(gridMap);
  let agent = new RandomAgent(env.getObservationSpace(), env.getActionSpace());
  let interaction = new Interaction(agent, env, 3);

  import Pong from "./Pong.svelte";
  import Go from "./Go.svelte";
  import Finance from "./Finance.svelte";
  import Robot from "./Robot.svelte";
  import AutonomousVehicles from "./AutonomousVehicles.svelte";

  const cellsStore = env.cellsStore;
  const playerStore = interaction.observationStore;

  $: cells = $cellsStore;
  $: player = $playerStore;
</script>

<svelte:head>
  <title>Reinforcement Learning Applications - World4AI</title>
  <meta
    name="description"
    content="Reinforcement learning is used in many different applications, like games, finance, robotics, healcare and autonomous vehicles."
  />
</svelte:head>

<h1>Reinforcement Learning Applications</h1>
<div class="separator" />

<Container>
  <p>
    Over the last decade research in reinforcement learning has skyrocketed.
    Initially most research projects focused on computer games and board games,
    but over the last couple of years more and more practical applications are
    being developed.
  </p>
  <p>
    In this section we will focus on a couple of areas, where reinforcement
    learning is known to be used, but this list is far from being exhaustive.
  </p>
  <div class="separator" />
  <h2>Grid Worlds</h2>
  <p>
    Most beginner reinforcement learning problems are grid world problems. They
    are easy enough to understand and do not require a lot of computational
    power to solve. A gridworld is (usually) a simple game. You have some sort
    of a player that can move from cell to cell, while avoiding some obstacles
    and trying to reach a goal. Of course there are grid worlds that are
    substantially more complex. These can for example include powerups, enemies
    and many different levels.
  </p>
  <Grid {cells} {player} />
  <p>
    In the above example the player has to move the circle from the top left
    corner to the goal (represented by a triangle) in bottom left corner, while
    avoiding the walls (represented by the three squares). This seems like a
    trivial task for a human being, but it gets nontrivial if you are not
    allowed to hardcode the solution. Instead you have to make your computer
    learn the goal of the game and the strategy to achieve the goal.
  </p>
  <p>
    That is exactly where reinforcement learning comes into play. By applying
    reinforcement learning algorithms it becomes possible to learn the optimal
    behaviour, where the circle arrives at the goal in as few steps as possible.
    In our example above the circle moves randomly and therefore it might take a
    while for the circle to get to the goal. In a few chapter we will implement
    our first algorithms and see how this simple task can be solved though
    reinforcement learning.
  </p>
  <div class="separator" />

  <h2>Computer Games</h2>
  <p>
    Computer games have become a testing ground for reinforcement learning
    algorithms. Most new algorithms are tested on the Atari 2600 games in order
    to show how efficient the algorithms are. For a human it is not especially
    hard to learn the rules of the game (although it might require some time to
    master the game), but for computers it is an entirely different story. Due
    to the vast number of possible pixel values on the screen the usual
    strategies that are used to solve the grid worlds break down. Good solutions
    require the use of neural networks.
  </p>
  <Pong />
  <p>
    Pong for example seems to be a relatively easy bask, but a computer program
    needs to learn to transform raw pixel values of the screen into actions.
    Nowadays we have the necessary tools to solve such problems, but just 10
    years ago this task seemed impossible.
  </p>
  <div class="separator" />

  <h2>Board Games</h2>
  <p>
    Board games, like backgammon, chess and Go used to be the frontier for AI.
    There was an assumption that a computer would require creativity and
    imagination to beat a professional player. This assumption implied that the
    computer needed to possess human characteristics in order to win against a
    professional player. Nevertheless in all three games professionals and even
    world champions were beaten by AI systems.
  </p>
  <p>
    Most cecently DeepMind's AlphaGo won against the Go world champion. For a
    number of years the challenge of winning against the world champion was
    considered impossible. The number of legal board positions in the game of go
    is far greater than the number atoms in the observable universe. Iterating
    through all positions is impossible. Nevertheless, not only did the
    algorithm win against the world champion, Lee Sedol, in the 4 of 5 games,
    but according to some Go experts AlphaGo showed creativity. In the second of
    the five games AlphaGo shocked the world with the now iconic move. This move
    has become known as <Highlight>Move 37</Highlight>.
  </p>
  <Go />
  <div class="separator" />

  <h2>Finance</h2>
  <p>
    Nowadays it seems that machine learning is taking over the financial
    industry in every aspect imaginable. From valuing financial products to
    chatbots that communicate with prospective clients.
  </p>
  <Finance />
  <p>
    The most exciting part still seems to be portfolio management though.
    Imagine an AI agent that decides what financial instrument to invest in and
    in what proportion. There is an abundance of financial data going back
    sometimes a hundred years. That data can be used to train a reinforcement
    learning agent to potentially perform better than humans over a long period
    of time. Even if the AI performs equally well to human portfolio managers,
    the banks could cut costs, as automated trading bots tend to be much cheaper
    than human portfolio managers.
  </p>
  <div class="separator" />

  <h2>Robotics</h2>
  <p>
    The field of robotics is vast. We could talk about robots on assembly lines,
    drones or bipedal robots. In all the above mentioned cases it is possible to
    apply reinforcement learning to learn the desired task for the robot.
  </p>
  <Robot />
  <p>
    A bipedal robot can for example be taught to walk on two legs through the
    means of reinforcement learning. Each step or fall of the robot can be used
    as a learning experience. At the moment many bipedal robots, like those made
    by Boston Dynamics, are not actually trained through reinforcement learning,
    but are hardcoded to solve their task. These results are great feats of pure
    engineering and not AI, but 10 years down the road and AI will probably
    replace a lot of hardcoded parts.
  </p>
  <div class="separator" />

  <h2>Autonomous Vehicles</h2>
  <AutonomousVehicles />
  <p>
    Autonomous vehicles (a.k.a. self-driving cars) are at the moment of writing
    the current frontier for reinforcement learning. There are many car
    companies that invest in self-driving cars. Newer car companies like Tesla
    and Google’s Waymo and old German car manufacturers like Volkswagen all
    invest an enormous amount of time and money in the development of autonomous
    vehicles. Research in the area has been going on since at least the 80’s,
    but the behaviour of these vehicles in edge cases made their use often
    dangerous for everyday use. Since the DARPA Grand Challenge (2007) great
    leaps have been made and reinforcement learning played a huge role in that
    success story.
  </p>
  <div class="separator" />
</Container>
